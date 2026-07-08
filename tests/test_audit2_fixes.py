"""Regression tests for the 2nd line-by-line audit fixes."""

from calibrator.models import BehaviorSpec, TaskType


# CRITICAL — gather() indices must be order-independent (detect/apply are
# separate API requests; a reordered `sources` must NOT misalign the drops).
def test_gather_indices_are_order_independent():
    from calibrator.stakeholders import gather
    a = BehaviorSpec(goal="g", standards=["a-standard"], do_not=["a-never"])
    b = BehaviorSpec(goal="g", standards=["b-standard"], do_not=["b-never"])

    fwd = {(s.stakeholder, s.text): s.idx for s in gather({"alice": a, "bob": b})}
    rev = {(s.stakeholder, s.text): s.idx for s in gather({"bob": b, "alice": a})}
    assert fwd == rev                       # same specs, different dict order → same indices
    # and a dropped index refers to the SAME statement either way
    assert fwd[("alice", "a-standard")] == rev[("alice", "a-standard")]


def test_build_merged_spec_drops_are_stable_across_source_order():
    from calibrator.stakeholders import build_merged_spec, gather
    a = BehaviorSpec(goal="g", standards=["keep-alice"], do_not=[])
    b = BehaviorSpec(goal="g", standards=["drop-bob"], do_not=[])
    # find bob's index, then drop it — must remove bob's rule regardless of order
    idx = {(s.stakeholder, s.text): s.idx for s in gather({"a": a, "b": b})}[("b", "drop-bob")]
    for named in ({"a": a, "b": b}, {"b": b, "a": a}):
        merged = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT, drops={idx})
        assert "drop-bob" not in merged.standards and "keep-alice" in merged.standards


# Modelfile: a system prompt containing `"""` must not break the SYSTEM block
def test_modelfile_neutralizes_triple_quotes():
    from calibrator.export import _modelfile
    mf = _modelfile("qwen2.5:7b", 'Say """hello""" and be """nice""".')
    body = mf.split('SYSTEM """\n', 1)[1].rsplit('\n"""', 1)[0]
    assert '"""' not in body                 # no run of 3+ quotes survives in the body
    assert mf.count('"""') == 2              # exactly the opening + closing delimiters


# snapshot: a corrupt golden.json must not traceback
def test_load_golden_tolerates_corrupt_json(tmp_path):
    from calibrator.snapshot import GOLDEN_FILE, load_golden
    (tmp_path / GOLDEN_FILE).write_text("{ this is not json ")
    assert load_golden(tmp_path) is None


# judge_check: a label missing 'passed' must not be silently saved as a FAIL
def test_save_labels_requires_passed(tmp_path):
    from calibrator.judge_check import load_labels, save_labels
    save_labels(tmp_path, "run-0001", [
        {"test_id": "t1", "criterion_id": "c1", "passed": True},   # valid
        {"test_id": "t2", "criterion_id": "c2"},                   # no 'passed' → skipped
    ])
    labels = load_labels(tmp_path, "run-0001")
    keys = {(x["test_id"], x["criterion_id"]) for x in labels}
    assert ("t1", "c1") in keys and ("t2", "c2") not in keys


# prove_engine: rows with a None logged output must be skipped, not scored
def test_prove_engine_skips_none_output(tmp_path):
    import json

    from calibrator.store import open_private_append
    from calibrator.train_engine import prove_engine
    logs = tmp_path / "logs"; logs.mkdir()
    with open_private_append(logs / "judge.jsonl") as fh:
        fh.write(json.dumps({"role": "judge", "prompt": "p1", "output": "real"}) + "\n")
        fh.write(json.dumps({"role": "judge", "prompt": "p2", "output": None}) + "\n")

    class Echo:
        name = "echo@test"
        def complete(self, prompt, *, system=None, schema=None):
            return "real"
    res = prove_engine(tmp_path, "judge", Echo())
    assert res.samples == 1        # only the row with a non-None output is measured


# loads_tolerant parses engine TEXT — bytes is a contract violation, not silently OK
def test_loads_tolerant_rejects_non_str():
    import pytest

    from calibrator.engines.base import loads_tolerant
    assert loads_tolerant('{"a": 1}') == {"a": 1}        # str still works
    with pytest.raises(ValueError):
        loads_tolerant(b'{"a": 1}')                       # bytes rejected, not auto-decoded
    with pytest.raises(ValueError):
        loads_tolerant({"a": 1})                          # already-a-dict rejected


# max_chars/min_chars must reject negative limits (they were always-fail garbage)
def test_run_check_rejects_negative_length_limits():
    from calibrator.checks import run_check
    from calibrator.models import Check
    for kind in ("max_chars", "min_chars"):
        passed, why = run_check(Check(kind=kind, value="-5"), "hello")
        assert passed is False and "non-negative" in why
    # a valid non-negative limit still grades normally
    assert run_check(Check(kind="max_chars", value="10"), "hello")[0] is True
    assert run_check(Check(kind="min_chars", value="10"), "hi")[0] is False
