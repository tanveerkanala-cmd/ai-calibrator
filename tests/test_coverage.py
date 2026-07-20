"""Behavioral coverage — deterministic spec↔test analysis."""

from ai_calibrator.coverage import analyze_coverage, coverage_dict
from ai_calibrator.models import BehaviorSpec, EvalCriterion, Weight
from ai_calibrator.models import TestCase as Case


def _spec(crits, **kw):
    return BehaviorSpec(goal="g", eval_criteria=crits, **kw)


def test_full_coverage_no_warnings():
    spec = _spec([
        EvalCriterion(id="c1", description="d", weight=Weight.HIGH),
        EvalCriterion(id="c2", description="d", weight=Weight.LOW),
    ])
    tests = [Case(id="t1", input="x", expects=["c1"]), Case(id="t2", input="y", expects=["c2"])]
    r = analyze_coverage(spec, tests)
    assert r.coverage_rate == 1.0
    assert len(r.covered_criteria) == 2 and not r.uncovered_criteria
    assert not r.warnings


def test_uncovered_high_weight_warns():
    spec = _spec([
        EvalCriterion(id="c1", description="d", weight=Weight.HIGH),
        EvalCriterion(id="c2", description="d", weight=Weight.MEDIUM),
    ])
    r = analyze_coverage(spec, [Case(id="t1", input="x", expects=["c2"])])
    assert r.coverage_rate == 0.5
    assert [c.id for c in r.uncovered_criteria] == ["c1"]
    assert any("HIGH-weight" in w for w in r.warnings)


def test_broad_tests_are_weak_coverage():
    spec = _spec([EvalCriterion(id="c1", description="d", weight=Weight.LOW)])
    r = analyze_coverage(spec, [Case(id="t1", input="x", expects=[])])  # grade-all
    assert r.broad_tests == ["t1"]
    assert r.coverage_rate == 0.0  # broad grading is not targeted coverage
    assert any("broad" in w for w in r.warnings)


def test_orphan_expectation_flagged():
    spec = _spec([EvalCriterion(id="c1", description="d", weight=Weight.LOW)])
    r = analyze_coverage(spec, [Case(id="t1", input="x", expects=["c1", "ghost"])])
    assert "ghost" in r.orphan_expectations
    assert any("not in the spec" in w for w in r.warnings)


def test_under_measurement_heuristic():
    # Many behavioral rules, but only one criterion to check them.
    spec = _spec(
        [EvalCriterion(id="c1", description="d", weight=Weight.LOW)],
        standards=["a", "b", "c", "d"], do_not=["e", "f"],
    )
    r = analyze_coverage(spec, [Case(id="t1", input="x", expects=["c1"])])
    assert any("unmeasured" in w for w in r.warnings)


def test_coverage_dict_shape():
    spec = _spec([EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    d = coverage_dict(analyze_coverage(spec, [Case(id="t1", input="x", expects=["c1"])]))
    assert d["coverage_rate"] == 1.0 and d["total_criteria"] == 1
    assert d["criteria"][0]["covered"] is True and d["criteria"][0]["targeted_by"] == ["t1"]
