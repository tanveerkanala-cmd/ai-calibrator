"""Calibration report — confidence score + nutrition-label rendering."""

from ai_calibrator.coverage import analyze_coverage
from ai_calibrator.models import (
    BehaviorSpec,
    CriterionResult,
    EvalCriterion,
    InterviewItem,
    Project,
    Scorecard,
    Weight,
)
from ai_calibrator.models import TestCase as Case
from ai_calibrator.models import TestResult as Result  # aliased: avoids pytest collecting the model
from ai_calibrator.report import calibration_confidence, render_report, report_dict


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


def test_badge_colors_track_certification(tmp_path):
    """Green ONLY for a passing gate that certifies the current config."""
    import re as _re

    from ai_calibrator.ci import run_ci
    from ai_calibrator.report import badge_dict

    class Judge:
        name = "j@test"

        def complete(self, prompt, *, system=None, schema=None):
            ids = _re.findall(r"^- (\S+):", prompt, _re.M)
            good = "GOOD" in prompt
            return {"results": [{"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0,
                                 "rationale": "r"} for i in ids]}

    class Subject:
        name = "s@test"

        def __init__(self, t):
            self.t = t

        def complete(self, prompt, *, system=None, schema=None):
            return self.t

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["Always answer with the documented policy."],
                          refusal_policy="decline medical questions",
                          eval_criteria=[EvalCriterion(id="c1", description="matches the documented policy",
                                                       weight=Weight.HIGH)])
    p.tests = [Case(id="t1", input="q", expects=["c1"])]

    assert badge_dict(p, tmp_path)["color"] == "lightgrey"           # nothing yet
    run_ci(p, Subject("GOOD"), Judge(), project_dir=tmp_path)
    b = badge_dict(p, tmp_path)
    assert b == {"schemaVersion": 1, "label": "calibrated", "message": "100% · 1 tests",
                 "color": "brightgreen"}
    p.spec.standards.append("changed")                                # stale
    assert badge_dict(p, tmp_path)["color"] == "orange"
    p.spec.standards.pop()
    run_ci(p, Subject("BAD"), Judge(), project_dir=tmp_path)          # red gate
    assert badge_dict(p, tmp_path) == {"schemaVersion": 1, "label": "calibrated",
                                       "message": "gate failing", "color": "red"}


def test_html_certificate_numbers_trace_to_computed_values(tmp_path):
    from ai_calibrator.coverage import analyze_coverage
    from ai_calibrator.models import Check, CriterionResult, Scorecard, TestResult
    from ai_calibrator.report import render_html_report

    p = Project(name="my-ai", goal="answer <returns> & stuff")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="cites the <30-day> window", weight=Weight.HIGH,
                      check=Check(kind="contains", value="30"))])
    p.tests = [Case(id="t1", input="q", expects=["c1"])]
    card = Scorecard(run_id="run-0007", results=[TestResult(test_id="t1", output="o", criteria=[
        CriterionResult(criterion_id="c1", passed=True, score=1.0, weight=Weight.HIGH)])])

    html = render_html_report(p, analyze_coverage(p.spec, p.tests), card, tmp_path)
    assert "100%" in html                       # coverage & pass rate & confidence
    assert "deterministic check (contains)" in html
    assert "&lt;30-day&gt;" in html             # HTML-escaped, not injected
    assert "run-0007" in html and "<script" not in html.lower()
