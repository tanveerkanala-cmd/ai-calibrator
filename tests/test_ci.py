"""`calibrate ci` — the composed lint → eval → drift → snapshot gate."""

import re

from calibrator.ci import ci_dict, run_ci
from calibrator.eval import run_eval, save_scorecard
from calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from calibrator.models import TestCase as CaseModel
from calibrator.snapshot import save_golden


class Judge:
    """Passes iff the output contains GOOD."""
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        good = "GOOD" in prompt
        return {"results": [{"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0,
                             "rationale": "r"} for i in ids]}


class Subject:
    name = "subject@test"

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, *, system=None, schema=None):
        return self.text


def _project():
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["Always answer with the documented policy."],
                          refusal_policy="decline medical questions",
                          eval_criteria=[EvalCriterion(id="c1", description="answer matches the documented policy",
                                                       weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="a question", expects=["c1"])]
    return p


def _statuses(result):
    return {s.name: s.status for s in result.stages}


def test_ci_all_green_first_run(tmp_path):
    r = run_ci(_project(), Subject("GOOD answer"), Judge(), project_dir=tmp_path)
    assert _statuses(r) == {"lint": "pass", "eval": "pass",
                            "drift": "skip",       # no baseline yet
                            "snapshot": "skip"}    # no golden pinned
    assert r.ok and r.run_id == "run-0001" and r.pass_rate == 1.0
    d = ci_dict(r)
    assert d["ok"] is True and len(d["stages"]) == 4


def test_ci_lint_error_stops_before_any_engine_acquisition(tmp_path):
    """Lint failure must exit before engines are even ACQUIRED — a broken spec
    shouldn't demand credentials, and an engine problem shouldn't mask lint."""
    def exploding_factory():
        raise AssertionError("engine factory must not be called when lint fails")

    p = _project()
    p.spec.eval_criteria = []  # no criteria → lint error
    r = run_ci(p, exploding_factory, exploding_factory, project_dir=tmp_path)
    assert _statuses(r) == {"lint": "fail", "eval": "skip", "drift": "skip", "snapshot": "skip"}
    assert not r.ok and r.run_id is None


def test_ci_accepts_engine_factories(tmp_path):
    """Factories resolve lazily after lint passes (the CLI/API path)."""
    calls = []

    def subject_factory():
        calls.append("subject")
        return Subject("GOOD answer")

    def judge_factory():
        calls.append("judge")
        return Judge()

    r = run_ci(_project(), subject_factory, judge_factory, project_dir=tmp_path)
    assert r.ok and calls == ["subject", "judge"]


def test_ci_eval_below_threshold_fails_but_later_stages_still_run(tmp_path):
    p = _project()
    save_scorecard(tmp_path, run_eval(p, Subject("GOOD baseline"), Judge(), run_id="run-0001"))
    r = run_ci(p, Subject("BAD now"), Judge(), project_dir=tmp_path, threshold=0.8)
    st = _statuses(r)
    assert st["eval"] == "fail"
    assert st["drift"] == "fail"          # t1 flipped pass→fail vs run-0001
    assert not r.ok


def test_ci_drift_regression_fails_even_when_eval_passes_threshold(tmp_path):
    p = _project()
    p.tests.append(CaseModel(id="t2", input="another", expects=["c1"]))

    class SplitSubject:  # t1 GOOD, t2 BAD → 50% pass
        name = "s@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "GOOD" if "a question" in prompt else "BAD"

    save_scorecard(tmp_path, run_eval(p, Subject("GOOD everywhere"), Judge(), run_id="run-0001"))
    r = run_ci(p, SplitSubject(), Judge(), project_dir=tmp_path, threshold=0.5)
    st = _statuses(r)
    assert st["eval"] == "pass"           # 50% >= threshold 0.5
    assert st["drift"] == "fail"          # but t2 regressed vs baseline
    assert "t2" in next(s.detail for s in r.stages if s.name == "drift")


def test_ci_snapshot_drift_fails_and_match_passes(tmp_path):
    p = _project()
    save_golden(tmp_path, {"t1": "GOOD answer"})
    ok = run_ci(p, Subject("GOOD answer"), Judge(), project_dir=tmp_path)
    assert _statuses(ok)["snapshot"] == "pass"

    changed = run_ci(p, Subject("GOOD but reworded"), Judge(), project_dir=tmp_path)
    st = changed.stages[-1]
    assert st.status == "fail" and "t1" in st.detail
    assert not changed.ok


def test_ci_explicit_baseline(tmp_path):
    p = _project()
    save_scorecard(tmp_path, run_eval(p, Subject("BAD old"), Judge(), run_id="run-0001"))   # failing baseline
    save_scorecard(tmp_path, run_eval(p, Subject("GOOD mid"), Judge(), run_id="run-0002"))
    # default baseline would be run-0002; pin run-0001 → t1 goes fail→pass = no regression
    r = run_ci(p, Subject("GOOD now"), Judge(), project_dir=tmp_path, baseline="run-0001")
    drift = next(s for s in r.stages if s.name == "drift")
    assert drift.status == "pass" and "run-0001" in drift.detail and "fixed" in drift.detail

    # a bogus explicit baseline must fail loudly, not silently skip
    r2 = run_ci(p, Subject("GOOD"), Judge(), project_dir=tmp_path, baseline="nope")
    assert next(s for s in r2.stages if s.name == "drift").status == "fail"
