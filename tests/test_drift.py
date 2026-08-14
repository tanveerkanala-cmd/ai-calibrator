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


def _hashed_card(run_id, results):
    """Like `_card`, but each result records WHAT was asked — (id, passed, hash)."""
    return Scorecard(run_id=run_id, results=[
        Result(test_id=tid, output="o", input_hash=h,
               criteria=[CriterionResult(criterion_id="c", passed=p)])
        for tid, p, h in results
    ])


def test_compare_excludes_results_whose_input_hash_differs():
    """`compile` re-mints t1..tN positionally, so a shared id can name two
    different questions. Flipping a verdict across that pair invents a
    regression the model never caused — or, worse, hides a real one and
    reports a deleted failure as a fix."""
    base = _hashed_card("run-0001", [("t1", False, "aaaa000000000000"), ("t2", True, "cccc222222222222")])
    cand = _hashed_card("run-0002", [("t1", True, "bbbb111111111111"), ("t2", True, "cccc222222222222")])
    r = compare_scorecards(base, cand)
    # t1 asked a different question in each run: not a fix, not a regression.
    assert r.fixed_tests == [] and r.regressed_tests == []
    assert r.incomparable_tests == ["t1"]
    assert r.compared == 1          # only t2 was actually compared
    assert r.comparable is True


def test_compare_reports_nothing_comparable_when_the_whole_suite_was_reminted():
    base = _hashed_card("run-0001", [("t1", True, "aaaa000000000000")])
    cand = _hashed_card("run-0002", [("t1", False, "bbbb111111111111")])
    r = compare_scorecards(base, cand)
    assert r.incomparable_tests == ["t1"] and r.compared == 0
    assert r.comparable is False
    # The rate moved, but across two different exams — that is not a regression.
    assert r.regressed is False
    # And there is no delta to report at all: 0.0 would read as "nothing
    # changed", -100% as a collapse. Neither happened; nothing was measured.
    assert r.delta is None
    assert r.baseline_shared_rate is None and r.candidate_shared_rate is None


def test_drift_dict_publishes_the_absent_delta_rather_than_a_number():
    """The payload every non-Python surface reads. "Nothing was comparable" has
    to arrive as a fact in the payload, or each consumer invents its own reading
    of a Δ that means nothing."""
    from ai_calibrator.drift import drift_dict

    base = _hashed_card("run-0001", [("t1", True, "aaaa000000000000")])
    cand = _hashed_card("run-0002", [("t1", False, "bbbb111111111111")])
    d = drift_dict(compare_scorecards(base, cand))
    assert d["delta"] is None and d["comparable"] is False and d["compared"] == 0
    assert d["baseline_shared_rate"] is None and d["candidate_shared_rate"] is None
    # The whole-run rates stay: they describe each run honestly on its own.
    assert d["baseline_rate"] == 1.0 and d["candidate_rate"] == 0.0


def test_a_reminted_test_does_not_move_the_delta():
    """The ordinary state after answering one more interview question: `compile`
    re-mints one probe and leaves the rest. A test excluded from the flip lists
    must be excluded from the rates too — a delta driven by the very results the
    comparison refused to make fails the gate on behavior that never changed."""
    held = [(f"t{i}", True, f"{i:016d}") for i in range(1, 10)]
    base = _hashed_card("run-0001", [*held, ("t10", True, "aaaa000000000000")])
    cand = _hashed_card("run-0002", [*held, ("t10", False, "bbbb111111111111")])
    r = compare_scorecards(base, cand)
    assert r.compared == 9 and r.incomparable_tests == ["t10"]
    assert r.regressed_tests == [] and r.fixed_tests == []
    assert r.delta == 0.0            # over the 9 that were actually compared
    assert r.regressed is False


def test_the_shared_rates_are_the_pair_the_delta_subtracts():
    """Whatever a surface prints beside a Δ has to be the two numbers that Δ is
    the difference of, or the display contradicts its own arithmetic."""
    base = _hashed_card("run-0001", [("t1", True, "a" * 16), ("t2", True, "b" * 16), ("t3", True, "c" * 16)])
    cand = _hashed_card("run-0002", [("t1", True, "a" * 16), ("t2", False, "b" * 16), ("t3", True, "z" * 16)])
    r = compare_scorecards(base, cand)
    assert r.compared == 2 and r.incomparable_tests == ["t3"]
    assert r.baseline_shared_rate == 1.0 and r.candidate_shared_rate == 0.5
    assert r.delta == -0.5 and r.regressed_tests == ["t2"] and r.regressed is True


def test_a_test_the_baseline_never_graded_is_not_drift():
    """A new probe has no "before": the baseline never asked it, so it cannot
    have flipped. Folding it into the rate scores two different exams against
    each other — the same reason a partial run is refused as a baseline."""
    base = _card("run-0001", [("a", True), ("b", True)])                # 100%
    cand = _card("run-0002", [("a", True), ("b", True), ("c", False)])  # 67% of a different set
    r = compare_scorecards(base, cand)
    assert r.compared == 2 and r.delta == 0.0
    assert r.regressed is False and r.regressed_tests == []


def test_tolerance_never_excuses_a_test_that_flipped_pass_to_fail():
    """Tolerance bounds the rate drop; a probe that went pass→fail is drift at
    any tolerance."""
    base = _card("run-0001", [("a", True), ("b", True)])
    cand = _card("run-0002", [("a", True), ("b", False)])
    assert compare_scorecards(base, cand, tolerance=0.9).regressed is True


def test_compare_still_matches_by_id_when_either_hash_is_none():
    """Back-compat: scorecards written before the field records None, which
    means "unknown", never "matches". Those keep comparing exactly as before."""
    base = _hashed_card("run-0001", [("t1", True, None)])
    cand = _hashed_card("run-0002", [("t1", False, "bbbb111111111111")])
    r = compare_scorecards(base, cand)
    assert r.regressed_tests == ["t1"] and r.incomparable_tests == []


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


# --- input validation --------------------------------------------------------

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
