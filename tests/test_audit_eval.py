"""Grading and reporting must not credit behavior no run ever exercised.

Three seams where a number could claim more than happened: a calibration report
whose headline multiplies today's coverage by an obsolete run's pass rate, a
deterministic check graded against every assistant turn glued together instead of
each answer, and a failures file that lists never-graded tests as failures.
"""

import json
import re

from ai_calibrator.coverage import analyze_coverage
from ai_calibrator.eval import run_eval, save_scorecard
from ai_calibrator.models import (
    BehaviorSpec,
    Check,
    CriterionResult,
    EvalCriterion,
    Project,
    Scorecard,
    Weight,
)
# Aliased: pytest would otherwise try to collect these models as test classes.
from ai_calibrator.models import TestCase as CaseModel
from ai_calibrator.models import TestResult as ResultModel
from ai_calibrator.report import render_html_report, render_report, report_dict


class PassJudge:
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "r"} for i in ids]}


class ScriptedSubject:
    """Answers turn N of a conversation with the Nth scripted reply."""
    name = "subject@test"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def complete(self, prompt, *, system=None, schema=None):
        return self.replies.pop(0)


# --- the report may not headline a run that predates the suite ---------------

def _grown_project() -> Project:
    """8 criteria, 8 targeted tests — a suite that grew after its last eval."""
    p = Project(name="support", goal="answer return questions")
    p.spec = BehaviorSpec(
        goal="answer return questions",
        eval_criteria=[EvalCriterion(id=f"c{i}", description=f"rule {i}", weight=Weight.HIGH)
                       for i in range(1, 9)],
    )
    p.tests = [CaseModel(id=f"t{i}", input=f"q{i}", expects=[f"c{i}"]) for i in range(1, 9)]
    return p


def _card_grading(test_ids: list[str], run_id: str = "run-0001") -> Scorecard:
    return Scorecard(run_id=run_id, subject="s@test", judge="j@test", results=[
        ResultModel(test_id=tid, output="ok", criteria=[
            CriterionResult(criterion_id=tid.replace("t", "c"), passed=True, score=1.0,
                            weight=Weight.HIGH)])
        for tid in test_ids
    ])


def test_confidence_does_not_credit_tests_the_run_never_graded():
    p = _grown_project()
    cov = analyze_coverage(p.spec, p.tests)          # coverage is 100%: every criterion is targeted
    card = _card_grading(["t1", "t2", "t3"])         # but only 3 of 8 tests were ever run

    assert card.pass_rate == 1.0 and not card.partial
    assert report_dict(p, cov, card)["confidence"] == 0.375
    assert "Calibration Confidence: 100%" not in render_report(p, cov, card)


def test_report_dict_lists_the_tests_the_run_never_graded():
    p = _grown_project()
    d = report_dict(p, analyze_coverage(p.spec, p.tests), _card_grading(["t1", "t2", "t3"]))
    assert d["ungraded_tests"] == ["t4", "t5", "t6", "t7", "t8"]


def test_report_flags_ungraded_tests_and_does_not_claim_no_failures():
    p = _grown_project()
    md = render_report(p, analyze_coverage(p.spec, p.tests), _card_grading(["t1", "t2", "t3"]))
    assert "`t4`" in md
    assert "- ✓ No failing tests." not in md


def test_html_certificate_reports_how_much_of_the_suite_the_run_graded(tmp_path):
    p = _grown_project()
    html = render_html_report(p, analyze_coverage(p.spec, p.tests),
                              _card_grading(["t1", "t2", "t3"]), tmp_path)
    assert "3/8 current test(s) graded" in html
    assert ">100%<" not in html                       # the headline is no longer a perfect score
    # The subtitle must name the rate the headline was built from, or a reader
    # multiplying the rows above gets a different answer than the number shown.
    assert "coverage × pass rate over the CURRENT test suite" in html


def test_current_run_is_not_falsely_flagged_stale(tmp_path):
    """A run that graded the whole suite must still read 100% — no spurious staleness."""
    p = _grown_project()
    cov = analyze_coverage(p.spec, p.tests)
    card = _card_grading([f"t{i}" for i in range(1, 9)])

    assert report_dict(p, cov, card)["confidence"] == 1.0
    assert report_dict(p, cov, card)["ungraded_tests"] == []
    md = render_report(p, cov, card)
    assert "Calibration Confidence: 100%" in md
    assert "- Confidence = coverage × pass rate." in md
    assert "- ✓ No failing tests." in md
    assert "never graded" not in md.lower()
    assert ">100%<" in render_html_report(p, cov, card, tmp_path)


def test_absorbed_regression_test_is_not_reported_as_passing(tmp_path):
    """`absorb` pins the exchange a human flagged; until it runs, it isn't passing."""
    from ai_calibrator.flywheel import absorb_feedback, append_feedback

    p = Project(name="support", goal="answer return questions")
    p.spec = BehaviorSpec(goal="answer return questions",
                          eval_criteria=[EvalCriterion(id="c1", description="on policy",
                                                       weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="q1", expects=["c1"])]
    card = _card_grading(["t1"])

    append_feedback(tmp_path, {"turns": ["Can I return after 40 days?"], "output": "Sure, any time!",
                               "verdict": "down", "correction": "No — the window is 30 days.",
                               "reason": "invented policy"})
    absorb_feedback(p, tmp_path)

    d = report_dict(p, analyze_coverage(p.spec, p.tests), card)
    assert "fb_1" in d["ungraded_tests"]
    assert d["confidence"] == 0.5                     # 1 of the 2 pinned behaviors is proven


# --- deterministic checks grade each answer, not every answer glued together --

def _checked(kind: str, value: str = "") -> BehaviorSpec:
    return BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="chk", description="graded by code", weight=Weight.HIGH,
                      check=Check(kind=kind, value=value))])


def _multi_turn_verdict(spec: BehaviorSpec, *replies: str) -> CriterionResult:
    p = Project(name="p", goal="g")
    p.spec = spec
    p.tests = [CaseModel(id="t1", input="q1", follow_ups=[f"q{i}" for i in range(2, len(replies) + 1)],
                         expects=["chk"])]
    card = run_eval(p, ScriptedSubject(*replies), PassJudge())
    return card.results[0].criteria[0]


def test_max_chars_applies_to_each_reply_not_their_concatenation():
    reply = "x" * 49
    cr = _multi_turn_verdict(_checked("max_chars", "50"), reply, reply, reply)
    assert cr.passed
    assert "149" not in (cr.rationale or "")


def test_min_chars_fails_the_turn_that_falls_short():
    cr = _multi_turn_verdict(_checked("min_chars", "25"), "Yes, our 30-day window applies.", "Sure.")
    assert not cr.passed
    assert "turn 2" in (cr.rationale or "")


def test_non_empty_fails_when_one_reply_says_nothing():
    cr = _multi_turn_verdict(_checked("non_empty"), "fine", "", "fine")
    assert not cr.passed


def test_regex_does_not_match_across_the_seam_between_two_replies():
    cr = _multi_turn_verdict(_checked("regex", r"re\nfund"), "I cannot discuss the re", "fund policy here.")
    assert not cr.passed


def test_a_forbidden_term_in_any_reply_fails_the_conversation():
    cr = _multi_turn_verdict(_checked("not_contains", "cure"), "No claims here.", "It is a cure.")
    assert not cr.passed


def test_a_required_term_is_satisfied_by_the_turn_that_carries_it():
    """Term checks read the conversation as a whole — a closing pleasantry is fine."""
    cr = _multi_turn_verdict(_checked("contains", "30-day"), "Our 30-day window applies.", "Happy to help!")
    assert cr.passed


def test_single_turn_rationales_are_unchanged_by_turn_aware_grading():
    from ai_calibrator.checks import run_check, run_check_turns

    cases = [("contains", "day", "within 30 days"), ("not_contains", "cure", "no claims"),
             ("regex", r"\d+ days", "within 30 days"), ("max_chars", "10", "short"),
             ("min_chars", "10", "hi"), ("non_empty", "", "hello")]
    for kind, value, text in cases:
        chk = Check(kind=kind, value=value)
        assert run_check_turns(chk, [text]) == run_check(chk, text)


# --- failures.jsonl is a list of failures ------------------------------------

def test_failures_file_records_only_tests_that_were_actually_graded(tmp_path):
    card = Scorecard(run_id="run-0001", results=[
        ResultModel(test_id="t-ok", output="fine",
                    criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)]),
        ResultModel(test_id="t-fail", output="bad",
                    criteria=[CriterionResult(criterion_id="c1", passed=False, score=0.0,
                                              rationale="wrong")]),
        ResultModel(test_id="t-ungraded", output="", criteria=[]),
    ])
    d = save_scorecard(tmp_path, card)

    rows = [json.loads(line) for line in
            (d / "failures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["test_id"] for r in rows] == ["t-fail"]
