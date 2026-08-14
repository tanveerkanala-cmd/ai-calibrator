"""Hardening regressions: merge-index stability, template quoting, tolerant
loaders, file permissions, and RAG-aware grading."""

from ai_calibrator.models import BehaviorSpec, TaskType


# CRITICAL — gather() indices must be order-independent (detect/apply are
# separate API requests; a reordered `sources` must NOT misalign the drops).
def test_gather_indices_are_order_independent():
    from ai_calibrator.stakeholders import gather
    a = BehaviorSpec(goal="g", standards=["a-standard"], do_not=["a-never"])
    b = BehaviorSpec(goal="g", standards=["b-standard"], do_not=["b-never"])

    fwd = {(s.stakeholder, s.text): s.idx for s in gather({"alice": a, "bob": b})}
    rev = {(s.stakeholder, s.text): s.idx for s in gather({"bob": b, "alice": a})}
    assert fwd == rev                       # same specs, different dict order → same indices
    # and a dropped index refers to the SAME statement either way
    assert fwd[("alice", "a-standard")] == rev[("alice", "a-standard")]


def test_build_merged_spec_drops_are_stable_across_source_order():
    from ai_calibrator.stakeholders import build_merged_spec, gather
    a = BehaviorSpec(goal="g", standards=["keep-alice"], do_not=[])
    b = BehaviorSpec(goal="g", standards=["drop-bob"], do_not=[])
    # find bob's index, then drop it — must remove bob's rule regardless of order
    idx = {(s.stakeholder, s.text): s.idx for s in gather({"a": a, "b": b})}[("b", "drop-bob")]
    for named in ({"a": a, "b": b}, {"b": b, "a": a}):
        merged = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT, drops={idx})
        assert "drop-bob" not in merged.standards and "keep-alice" in merged.standards


# Modelfile: a system prompt containing `"""` must not break the SYSTEM block
def test_modelfile_neutralizes_triple_quotes():
    from ai_calibrator.export import _modelfile
    mf = _modelfile("qwen2.5:7b", 'Say """hello""" and be """nice""".')
    body = mf.split('SYSTEM """\n', 1)[1].rsplit('\n"""', 1)[0]
    assert '"""' not in body                 # no run of 3+ quotes survives in the body
    assert mf.count('"""') == 2              # exactly the opening + closing delimiters


# snapshot: a corrupt golden.json must not traceback
def test_load_golden_tolerates_corrupt_json(tmp_path):
    from ai_calibrator.snapshot import GOLDEN_FILE, load_golden
    (tmp_path / GOLDEN_FILE).write_text("{ this is not json ")
    assert load_golden(tmp_path) is None


# judge_check: a label missing 'passed' must not be silently saved as a FAIL
def test_save_labels_requires_passed(tmp_path):
    from ai_calibrator.judge_check import load_labels, save_labels
    save_labels(tmp_path, "run-0001", [
        {"test_id": "t1", "criterion_id": "c1", "passed": True},   # valid
        {"test_id": "t2", "criterion_id": "c2"},                   # no 'passed' → skipped
    ])
    labels = load_labels(tmp_path, "run-0001")
    keys = {(x["test_id"], x["criterion_id"]) for x in labels}
    assert ("t1", "c1") in keys and ("t2", "c2") not in keys


# a logged row with a None output has no answer to train on or to score against
def test_a_logged_row_with_no_output_is_not_measurable(tmp_path):
    import json

    from ai_calibrator.store import open_private_append
    from ai_calibrator.train_engine import _usable_log
    logs = tmp_path / "logs"; logs.mkdir()
    with open_private_append(logs / "judge.jsonl") as fh:
        fh.write(json.dumps({"role": "judge", "prompt": "p1", "output": "real"}) + "\n")
        fh.write(json.dumps({"role": "judge", "prompt": "p2", "output": None}) + "\n")

    # The one population both the dataset and the prove-it gate are drawn from, so
    # a row with no recorded answer can neither be trained on nor scored against.
    assert len(_usable_log(tmp_path, "judge")) == 1


# loads_tolerant parses engine TEXT — bytes is a contract violation, not silently OK
def test_loads_tolerant_rejects_non_str():
    import pytest

    from ai_calibrator.engines.base import loads_tolerant
    assert loads_tolerant('{"a": 1}') == {"a": 1}        # str still works
    with pytest.raises(ValueError):
        loads_tolerant(b'{"a": 1}')                       # bytes rejected, not auto-decoded
    with pytest.raises(ValueError):
        loads_tolerant({"a": 1})                          # already-a-dict rejected


# max_chars/min_chars must reject negative limits (they were always-fail garbage)
def test_run_check_rejects_negative_length_limits():
    from ai_calibrator.checks import run_check
    from ai_calibrator.models import Check
    for kind in ("max_chars", "min_chars"):
        passed, why = run_check(Check(kind=kind, value="-5"), "hello")
        assert passed is False and "non-negative" in why
    # a valid non-negative limit still grades normally
    assert run_check(Check(kind="max_chars", value="10"), "hello")[0] is True
    assert run_check(Check(kind="min_chars", value="10"), "hi")[0] is False


# open_private_append must tighten a PRE-EXISTING loose file too
def test_open_private_append_tightens_existing_file(tmp_path):
    import os
    import stat as statmod
    if os.name == "nt":
        return
    from ai_calibrator.store import open_private_append
    p = tmp_path / "log.jsonl"
    p.write_text("old\n")
    os.chmod(p, 0o644)                      # world-readable, as a pre-fix file would be
    with open_private_append(p) as fh:
        fh.write("new\n")
    assert statmod.S_IMODE(p.stat().st_mode) == 0o600   # now owner-only


# the fd-leak guard must NOT let a failing os.close mask the original
# error or skip temp cleanup
def test_fd_guard_preserves_original_error_and_cleans_up(tmp_path, monkeypatch):
    import pytest

    import ai_calibrator.store as store

    class Boom(Exception):
        pass

    def bad_fdopen(*a, **k):
        raise Boom("original fdopen failure")

    def bad_close(fd):
        raise OSError("close also failed")

    monkeypatch.setattr(store.os, "fdopen", bad_fdopen)
    monkeypatch.setattr(store.os, "close", bad_close)
    with pytest.raises(Boom):                          # ORIGINAL error, not the close OSError
        store.atomic_write_text(tmp_path / "x.txt", "data")
    # POSIX unlinks an open file fine; on Windows the mocked os.close leaves the fd
    # open so the OS refuses to delete the temp (a test artifact — real os.close
    # would close it). The invariant that matters is the ORIGINAL error propagating.
    import os as _os
    if _os.name != "nt":
        assert list(tmp_path.glob("x.txt.*.tmp")) == []


def test_rightsize_grades_with_rag_when_indexed(tmp_path):
    """rightsize recommends a model for PRODUCTION, which serves with RAG — so its
    eval must augment with retrieved knowledge when an index exists (else it can
    recommend the wrong model)."""
    import pytest
    pytest.importorskip("lancedb")
    pytest.importorskip("sentence_transformers")
    from ai_calibrator import rag
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from ai_calibrator.models import TestCase as Case
    from ai_calibrator.rightsize import rightsize

    rag.build_index(tmp_path, [{"id": "c1", "text": "The return window is 30 days.", "source": "p.md"}])
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", knowledge_sources=["p.md"],
                          eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    p.tests = [Case(id="t1", input="how long to return?", expects=["c1"])]

    seen = []

    class Spy:
        def __init__(self, spec): self.name = spec
        def complete(self, prompt, *, system=None, schema=None):
            seen.append(system); return "answer"

    class J:
        name = "j@t"
        def complete(self, prompt, *, system=None, schema=None):
            import re
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "ok"} for i in ids]}

    rightsize(p, ["m@ollama"], J(), lambda s: Spy(s), project_dir=tmp_path)
    assert any(s and "RETRIEVED KNOWLEDGE" in s and "30 days" in s for s in seen)   # RAG applied


def test_redteam_and_teach_and_try_use_rag_when_indexed(tmp_path):
    """The 'test/probe/show what you deploy' class: redteam, teach, and the API
    /try must all augment with retrieved knowledge when an index exists."""
    import pytest
    pytest.importorskip("lancedb")
    pytest.importorskip("sentence_transformers")
    from ai_calibrator import rag
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from ai_calibrator.redteam import run_redteam
    from ai_calibrator.teach import propose_candidates

    rag.build_index(tmp_path, [{"id": "c1", "text": "The return window is 30 days.", "source": "p.md"}])
    p = Project(name="p", goal="answer returns")
    p.spec = BehaviorSpec(goal="g", knowledge_sources=["p.md"],
                          do_not=["never reveal internal notes"],   # a rule → redteam has something to probe
                          eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])

    seen = []

    class Spy:
        name = "spy@t"
        def complete(self, prompt, *, system=None, schema=None):
            seen.append(system); return "ok"

    class Gen:
        name = "gen@t"
        def complete(self, prompt, *, system=None, schema=None):
            return {"probes": [{"input": "how long to return?", "target": "policy", "tactic": "direct"}],
                    "inputs": ["how long to return?"]}

    class J:
        name = "j@t"
        def complete(self, prompt, *, system=None, schema=None):
            return {"violated": False, "severity": "low", "rationale": "fine"}

    # redteam
    seen.clear()
    run_redteam(p, Gen(), Spy(), J(), project_dir=tmp_path, max_probes=1)
    assert any(s and "RETRIEVED KNOWLEDGE" in s and "30 days" in s for s in seen), "redteam not RAG-augmented"

    # teach
    seen.clear()
    propose_candidates(p, Gen(), Spy(), n=1, project_dir=tmp_path)
    assert any(s and "RETRIEVED KNOWLEDGE" in s for s in seen), "teach not RAG-augmented"
