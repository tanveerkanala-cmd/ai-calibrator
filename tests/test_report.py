"""Calibration report — confidence score + nutrition-label rendering."""

from calibrator.coverage import analyze_coverage
from calibrator.models import (
    BehaviorSpec,
    CriterionResult,
    EvalCriterion,
    InterviewItem,
    Project,
    Scorecard,
    Weight,
)
from calibrator.models import TestCase as Case
from calibrator.models import TestResult as Result  # aliased: avoids pytest collecting the model
from calibrator.report import calibration_confidence, render_report, report_dict


def _project():
    p = Project(name="support", goal="answer support questions")
    p.spec = BehaviorSpec(
        goal="answer support questions",
        standards=["Be concise."],
        eval_criteria=[
            EvalCriterion(id="c1", description="cites policy", weight=Weight.HIGH),
            EvalCriterion(id="c2", description="warm tone", weight=Weight.LOW),
        ],
    )
    p.tests = [Case(id="t1", input="q", expects=["c1"])]  # c2 has no targeted test
    p.interview = [InterviewItem(id="q1", dimension="tone", question="Voice?", answer="warm")]
    return p


def test_confidence_zero_without_eval():
    assert calibration_confidence(0.5, 0.9, has_eval=False) == 0.0


def test_confidence_is_coverage_times_pass():
    assert calibration_confidence(0.5, 0.8, has_eval=True) == 0.4


def test_render_report_sections_and_provenance():
    p = _project()
    cov = analyze_coverage(p.spec, p.tests)
    md = render_report(p, cov, latest=None)
    assert "# Calibration Report — support" in md
    assert "Calibration Confidence: 0%" in md       # no eval yet
    assert "`c2`" in md                              # uncovered criterion surfaced
    assert "Provenance" in md and "Voice?" in md and "warm" in md


def test_render_report_with_eval_computes_confidence():
    p = _project()
    cov = analyze_coverage(p.spec, p.tests)  # coverage = 1/2 = 50%
    card = Scorecard(run_id="run-0003", results=[
        Result(test_id="t1", output="o", criteria=[CriterionResult(criterion_id="c1", passed=True)]),
    ])  # pass rate 100%
    md = render_report(p, cov, card)
    assert "Calibration Confidence: 50%" in md   # 50% coverage × 100% pass
    assert "run-0003" in md


def test_report_dict_shape():
    p = _project()
    cov = analyze_coverage(p.spec, p.tests)
    d = report_dict(p, cov, latest=None)
    assert d["confidence"] == 0.0 and d["pass_rate"] is None
    assert "c2" in d["uncovered_criteria"] and d["coverage_rate"] == 0.5
