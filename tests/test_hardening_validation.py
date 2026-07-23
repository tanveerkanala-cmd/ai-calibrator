"""Hardening regressions: name/path validation, locking, gitignore writes,
engine error handling, and label loading."""

import pytest

from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from ai_calibrator.models import TestCase as CaseModel


# #7 — name validator normalizes + enforces the cap on the STORED value
def test_project_name_stripped_and_capped():
    from pydantic import ValidationError
    assert Project(name="  spaced  ", goal="g").name == "spaced"
    # stripped to exactly 120 is valid AND stored stripped (not the padded 126)
    p = Project(name=" " * 3 + "a" * 120 + " " * 3, goal="g")
    assert p.name == "a" * 120 and len(p.name) == 120
    with pytest.raises(ValidationError):
        Project(name="a" * 121, goal="g")


# #8 — safe_token rejects traversal shapes the charset would otherwise allow
def test_safe_token_rejects_traversal():
    from ai_calibrator.coerce import safe_token
    assert safe_token("org/Model-7B.v2", "x") == "org/Model-7B.v2"
    for bad in ["../etc", "a/../b", "/abs", "trailing/", ".."]:
        with pytest.raises(ValueError):
            safe_token(bad, "base model")


# #9 — config_hash re-earns certification when the JUDGE changes
def test_config_hash_includes_judge():
    from ai_calibrator.ci import config_hash
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c", description="d", weight=Weight.HIGH)])
    before = config_hash(p)
    p.engines.judge = "some-other-judge@ollama"
    assert config_hash(p) != before


# #13 — FileLock is not re-entrant: a double acquire fails fast (no deadlock)
def test_filelock_double_acquire_raises(tmp_path):
    from ai_calibrator.locking import FileLock
    lock = FileLock(tmp_path / ".lock")
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="not re-entrant"):
            lock.acquire()
    finally:
        lock.release()


# #14 — empty project.yaml → friendly ValueError, not a cryptic pydantic wall
def test_empty_project_yaml_is_friendly(tmp_path):
    from ai_calibrator.store import load_project
    (tmp_path / "project.yaml").write_text("")
    with pytest.raises(ValueError, match="is empty"):
        load_project(tmp_path)


# #15 — write_project_gitignore never clobbers an existing file (atomic O_EXCL)
def test_gitignore_never_clobbers(tmp_path):
    from ai_calibrator.store import write_project_gitignore
    (tmp_path / ".gitignore").write_text("# mine\n")
    write_project_gitignore(tmp_path)
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == "# mine\n"


# #16 — OpenAI empty choices → friendly RuntimeError, not IndexError
def test_openai_empty_choices_is_friendly(monkeypatch):
    pytest.importorskip("openai")
    from ai_calibrator.engines.openai import OpenAIEngine

    class FakeResp:
        choices = []

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return FakeResp()

    for bad_choices in ([], None, 42, {"x": 1}):   # empty, None, non-subscriptable, dict
        class R:
            choices = bad_choices

        class Client:
            class chat:
                class completions:
                    create = staticmethod(lambda **kw: R())

        eng = OpenAIEngine.__new__(OpenAIEngine)
        eng.model = "gpt-x"; eng.name = "gpt-x@openai"; eng._client = Client()
        with pytest.raises(RuntimeError, match="no usable choices"):
            eng._chat([{"role": "user", "content": "hi"}])


# #17 — human ground truth overrides a logged row even when the log had no system msg
def test_ground_truth_overrides_systemless_logged_row(tmp_path):
    import json

    from ai_calibrator.eval import judge_prompt, save_scorecard
    from ai_calibrator.judge_check import save_labels
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult
    from ai_calibrator.store import save_project
    from ai_calibrator.train_engine import export_engine_bundle

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="cites policy", weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="can I return this?", expects=["c1"])]
    save_project(p, tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[TestResult(
        test_id="t1", output="the answer", criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])]))
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c1", "passed": False}])

    prompt = judge_prompt("can I return this?", "the answer", [("c1", "cites policy")])
    (tmp_path / "logs").mkdir(exist_ok=True)
    # logged judge row WITHOUT a system message (the case the old key missed)
    (tmp_path / "logs" / "judge.jsonl").write_text(json.dumps({
        "role": "judge", "prompt": prompt, "system": None,
        "output": {"results": [{"criterion_id": "c1", "passed": True, "score": 1.0, "rationale": "j"}]},
    }) + "\n")

    result = export_engine_bundle(tmp_path, "judge")
    lines = (tmp_path / "trained-engines" / "judge" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert result.examples == 1 and result.human_examples == 1     # logged row dropped, human kept
    target = json.loads(json.loads(lines[0])["messages"][-1]["content"])
    assert target["results"][0]["passed"] is False                 # the HUMAN verdict


# #18 — load_labels drops entries missing the verdict
def test_load_labels_requires_passed(tmp_path):
    import json

    from ai_calibrator.judge_check import load_labels
    d = tmp_path / "evals" / "run-0001"
    d.mkdir(parents=True)
    (d / "human-labels.json").write_text(json.dumps({"run_id": "run-0001", "labels": [
        {"test_id": "t1", "criterion_id": "c1", "passed": True},
        {"test_id": "t2", "criterion_id": "c1"},                   # no verdict → dropped
    ]}))
    assert load_labels(tmp_path, "run-0001") == [{"test_id": "t1", "criterion_id": "c1", "passed": True}]


# #20 — the corpus cap is never exceeded (header now counted against the budget)
def test_ingest_cap_is_respected():
    from pathlib import Path

    from ai_calibrator.ingest import _join_capped
    docs = [(Path(f"f{i}.md"), "x" * 50) for i in range(20)]
    out = _join_capped(docs, cap=200)
    assert len(out) <= 200


# OpenAI adapter: raw SDK errors (auth/404/rate/timeout) → friendly RuntimeError.
# Found by a LIVE run against the real API; tested here with a fake SDK (no key).
def test_openai_sdk_errors_are_friendly(monkeypatch):
    openai = pytest.importorskip("openai")
    from ai_calibrator.engines.openai import OpenAIEngine

    def _raiser(exc):
        class C:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        raise exc
        return C()

    def mk(cls):
        # Build a real SDK exception instance; fall back to __new__ for
        # constructor-signature stability across openai SDK versions.
        try:
            return cls("x", response=None, body=None)
        except Exception:
            return cls.__new__(cls)

    # A bad key surfaces the actionable missing-credentials guidance (not "401");
    # the other statuses still carry their code.
    for cls, expect in [(openai.NotFoundError, "404"),
                        (openai.AuthenticationError, "OPENAI_API_KEY"),
                        (openai.PermissionDeniedError, "403"), (openai.RateLimitError, "429")]:
        eng = OpenAIEngine.__new__(OpenAIEngine)
        eng.model = "gpt-x"; eng.name = "gpt-x@openai"; eng._client = _raiser(mk(cls))
        with pytest.raises(RuntimeError) as ei:
            eng._chat([{"role": "user", "content": "hi"}])
        assert expect in str(ei.value)


# Regression: O_EXCL gitignore write failure must not leave a 0-byte file that
# blocks the next retry.
def test_gitignore_write_failure_leaves_no_stub(tmp_path, monkeypatch):
    import os as _os

    from ai_calibrator import store

    real_fdopen = _os.fdopen

    def boom(fd, *a, **k):
        fh = real_fdopen(fd, *a, **k)
        def failing(_):
            raise OSError("disk full")
        fh.write = failing
        return fh

    monkeypatch.setattr(store.os, "fdopen", boom)
    with pytest.raises(OSError):
        store.write_project_gitignore(tmp_path)
    assert not (tmp_path / ".gitignore").exists()   # no 0-byte stub left behind

    monkeypatch.undo()
    store.write_project_gitignore(tmp_path)          # retry now succeeds
    assert "evals/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")
