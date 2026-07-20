"""Behavior drift detection — scorecard comparison + fresh-eval drift run."""

import re

import pytest

from ai_calibrator.drift import compare_scorecards, load_scorecard, run_drift
from ai_calibrator.eval import next_run_id, run_eval, save_scorecard
from ai_calibrator.models import (
    BehaviorSpec,
    CriterionResult,
    EvalCriterion,
    Project,
    Scorecard,
    Weight,
)
from ai_calibrator.models import TestCase as Case
from ai_calibrator.models import TestResult as Result  # aliased: avoids pytest collecting the model


def _card(run_id, results):
    return Scorecard(run_id=run_id, results=[
        Result(test_id=tid, output="o", criteria=[CriterionResult(criterion_id="c", passed=p)])
        for tid, p in results
    ])


def test_compare_detects_regression_and_fix():
    base = _card("run-0001", [("t1", True), ("t2", True), ("t3", False)])
    cand = _card("run-0002", [("t1", True), ("t2", False), ("t3", True)])
    r = compare_scorecards(base, cand)
    assert r.regressed_tests == ["t2"]
    assert r.fixed_tests == ["t3"]
    assert r.regressed is True


def test_no_drift_when_stable():
    base = _card("run-0001", [("t1", True), ("t2", True)])
    cand = _card("run-0002", [("t1", True), ("t2", True)])
    r = compare_scorecards(base, cand)
    assert not r.regressed and r.delta == 0.0
    assert r.regressed_tests == [] and r.fixed_tests == []


def test_tolerance_absorbs_rate_drop_without_test_regression():
    base = _card("run-0001", [("a", True), ("b", True)])               # 100%
    cand = _card("run-0002", [("a", True), ("b", True), ("c", False)])  # 67%, but a/b held
    assert compare_scorecards(base, cand, tolerance=0.0).regressed is True
    assert compare_scorecards(base, cand, tolerance=0.5).regressed is False


class _Subj:
    def __init__(self, marker):
        self.marker = marker
        self.name = "subject@test"

    def complete(self, prompt, *, system=None, schema=None):
        return self.marker


class _Judge:
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        good = "GOOD" in prompt
        return {"results": [{"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0, "rationale": ""} for i in ids]}


def _project():
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    p.tests = [Case(id="t1", input="q", expects=["c1"])]
    return p


def test_run_drift_runs_eval_persists_and_flags_regression(tmp_path):
    project = _project()
    base = run_eval(project, _Subj("GOOD answer"), _Judge(), run_id=next_run_id(tmp_path))
    save_scorecard(tmp_path, base)  # baseline passes

    report, cand = run_drift(project, _Subj("BAD answer"), _Judge(), baseline=base, project_dir=tmp_path)
    assert report.regressed is True
    assert report.regressed_tests == ["t1"]
    assert (tmp_path / "evals" / cand.run_id / "scorecard.json").exists()
    assert load_scorecard(tmp_path, base.run_id).run_id == base.run_id


# --- input validation (stress findings) -------------------------------------

@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf"), "0.5", None, True])
def test_compare_rejects_bad_tolerance(bad):
    base, cand = _card("run-0001", [("t1", True)]), _card("run-0002", [("t1", True)])
    with pytest.raises((ValueError, TypeError)):
        compare_scorecards(base, cand, tolerance=bad)


@pytest.mark.parametrize("bad", [{}, [], None, "x", 5])
def test_compare_rejects_non_scorecard_args(bad):
    base = _card("run-0001", [("t1", True)])
    with pytest.raises(TypeError):
        compare_scorecards(base, bad)
    with pytest.raises(TypeError):
        compare_scorecards(bad, base)


@pytest.mark.parametrize("bad", ["", "   ", "../evil", "a/b", "x\\y"])
def test_load_scorecard_rejects_unsafe_run_id(tmp_path, bad):
    with pytest.raises(ValueError):
        load_scorecard(tmp_path, bad)


@pytest.mark.parametrize("bad", ["", "   ", "a/b", "../x", "x\\y", "a\x00b"])
def test_scorecard_rejects_unsafe_run_id(bad):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Scorecard(run_id=bad)


def test_scorecard_accepts_normal_run_ids():
    assert Scorecard(run_id="run-0001").run_id == "run-0001"
    assert Scorecard(run_id="redteam-0002").run_id == "redteam-0002"
