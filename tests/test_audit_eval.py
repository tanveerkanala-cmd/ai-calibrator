"""Grading and reporting must not credit behavior no run ever exercised.

Three seams where a number could claim more than happened: a calibration report
whose headline multiplies today's coverage by an obsolete run's pass rate, a
deterministic check graded against every assistant turn glued together instead of
each answer, and a failures file that lists never-graded tests as failures.
"""

import json
import re

from ai_calibrator.coverage import analyze_coverage
from ai_calibrator.drift import load_scorecard
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


# --- the gate cannot certify what the judge never graded --------------------

class _PartialJudge:
    """Grades the criteria it is asked about, except for one test it stays silent
    on — a malformed reply, a dropped criterion, a judge that lost the id."""

    name = "judge@test"

    def __init__(self, silent_on: str):
        self.silent_on = silent_on

    def complete(self, prompt, *, system=None, schema=None):
        import re as _re
        if self.silent_on in prompt:
            return {"results": []}
        ids = _re.findall(r"^- (\S+):", prompt, _re.M)
        return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "r"}
                            for i in ids]}


class _Subject:
    name = "subject@test"

    def complete(self, prompt, *, system=None, schema=None):
        return "an answer that follows the documented policy"


def _gate_project():
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from ai_calibrator.models import TestCase as CaseModel

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(
        goal="g",
        standards=["Always answer with the documented policy."],
        refusal_policy="decline medical questions",
        eval_criteria=[EvalCriterion(id="c1", description="answer matches the documented policy",
                                     weight=Weight.HIGH)],
    )
    p.tests = [CaseModel(id="t1", input="the first question", expects=["c1"]),
               CaseModel(id="t2", input="the second question", expects=["c1"])]
    return p


def test_a_silent_judge_fails_the_criterion_instead_of_dropping_the_test(tmp_path):
    """This is what keeps the gate honest, and it is worth pinning explicitly.

    A judge that answers about t1 and says nothing about t2 must not make t2
    disappear: an ungraded test leaves the pass rate's denominator, so the
    surviving rate would be a true number about half the suite, and the gate
    would certify on it. Instead every requested criterion comes back with a
    verdict, defaulting to a fail — so t2 stays counted, the rate is 50%, and
    the gate refuses."""
    from ai_calibrator.ci import run_ci

    result = run_ci(_gate_project(), _Subject(), _PartialJudge("the second question"),
                    project_dir=tmp_path, threshold=0.8)

    assert result.pass_rate == 0.5           # t2 counted, not dropped
    stage = next(s for s in result.stages if s.name == "eval")
    assert stage.status == "fail" and not result.ok
    card = load_scorecard(tmp_path, result.run_id)
    t2 = next(r for r in card.results if r.test_id == "t2")
    assert t2.criteria and not t2.passed     # graded, and graded as a failure


def test_a_fully_graded_run_certifies(tmp_path):
    """The companion case: nothing silent, so nothing to hold back."""
    from ai_calibrator.ci import run_ci

    result = run_ci(_gate_project(), _Subject(), _PartialJudge("nothing matches this"),
                    project_dir=tmp_path, threshold=0.8)

    stage = next(s for s in result.stages if s.name == "eval")
    assert stage.status == "pass" and result.ok and result.pass_rate == 1.0


# --- a score that goes up is a claim, and needs a receipt ------------------

def _report_project(test_ids):
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from ai_calibrator.models import TestCase as CaseModel

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="answers from the documented policy", weight=Weight.HIGH)])
    p.tests = [CaseModel(id=t, input=f"q {t}", expects=["c1"]) for t in test_ids]
    return p


def _card(outcomes):
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult

    return Scorecard(run_id="run-0001", results=[
        TestResult(test_id=tid, output="o",
                   criteria=[CriterionResult(criterion_id="c1", passed=ok, score=1.0 if ok else 0.0)])
        for tid, ok in outcomes])


def test_deleting_a_failing_test_raises_the_score_and_the_report_says_so():
    """The confidence divides by the CURRENT suite, so dropping a test the run
    failed lifts the headline with no new evidence. That can be a legitimate edit,
    but it cannot be silent."""
    card = _card([("t1", True), ("t2", False)])
    before = _report_project(["t1", "t2"])
    after = _report_project(["t1"])                       # t2 deleted

    cov_before = analyze_coverage(before.spec, before.tests)
    cov_after = analyze_coverage(after.spec, after.tests)
    d_before = report_dict(before, cov_before, card)
    d_after = report_dict(after, cov_after, card)

    assert d_after["confidence"] > d_before["confidence"]  # the number really does move
    assert d_after["dropped_tests"] == ["t2"] and d_before["dropped_tests"] == []

    text = render_report(after, cov_after, card)
    assert "no longer in the" in text and "`t2`" in text and "FAILED" in text


def test_an_unchanged_suite_reports_nothing_dropped():
    card = _card([("t1", True), ("t2", False)])
    project = _report_project(["t1", "t2"])
    cov = analyze_coverage(project.spec, project.tests)

    assert report_dict(project, cov, card)["dropped_tests"] == []
    assert "no longer in the" not in render_report(project, cov, card)


def test_a_banning_regex_fails_the_turn_that_breaks_it():
    """`regex` is the only kind that can express a pattern-based ban — a literal
    `not_contains` cannot. Grading it as "any turn carries it" would pass a
    conversation in which one reply emitted the forbidden pattern, and would
    certify green what `calibrate run --guard` flags on the same replies."""
    banned = _checked("regex", r"^(?!.*\bguarantee\b).*$")
    cr = _multi_turn_verdict(banned, "Our policy is a 30-day window.",
                             "I guarantee a full refund.", "Anything else?")
    assert not cr.passed
    assert "turn 2" in (cr.rationale or "")


def test_a_required_term_is_still_satisfied_by_the_turn_that_says_it():
    """The companion: `contains` stays conversation-level, so a closing pleasantry
    does not fail a criterion the substantive turn satisfied."""
    cr = _multi_turn_verdict(_checked("contains", "30-day"),
                             "Our policy is a 30-day window.", "Happy to help!")
    assert cr.passed


# --- a test id is a slot, not an identity ----------------------------------

def test_recompiled_tests_are_not_credited_with_the_old_run_verdicts(tmp_path):
    """`compile` mints t1..tN positionally and regenerates the whole range, so the
    ordinary loop — compile, eval, answer more questions, compile again — replaces
    every probe with different text under the SAME id. Crediting the old run's
    passes to them reports behavior that has never been executed as proven."""
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from ai_calibrator.models import TestCase as CaseModel

    def _project(inputs):
        p = Project(name="p", goal="g")
        p.spec = BehaviorSpec(goal="g", eval_criteria=[
            EvalCriterion(id="c1", description="on policy", weight=Weight.HIGH)])
        p.tests = [CaseModel(id=f"t{i}", input=text, expects=["c1"])
                   for i, text in enumerate(inputs, start=1)]
        return p

    before = _project(["what is the return window?", "do you price match?"])
    card = run_eval(before, ScriptedSubject("a", "b"), PassJudge())
    assert card.pass_rate == 1.0
    assert all(r.input_hash for r in card.results)      # the run records WHAT it asked

    cov = analyze_coverage(before.spec, before.tests)
    assert report_dict(before, cov, card)["ungraded_tests"] == []

    # Same ids, different questions — what a second `compile` produces.
    after = _project(["can I return a gift?", "is shipping free?"])
    d = report_dict(after, analyze_coverage(after.spec, after.tests), card)

    assert d["ungraded_tests"] == ["t1", "t2"]
    assert d["suite_pass_rate"] == 0.0 and d["confidence"] == 0.0
    assert d["dropped_tests"] == ["t1", "t2"]           # and the old ones are named as gone


def test_a_scorecard_without_content_hashes_still_reports_as_it_did(tmp_path):
    """Existing projects must not have their reports invalidated: a result written
    before the hash existed is matched by id, exactly as before."""
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult

    project = _report_project(["t1", "t2"])
    legacy = Scorecard(run_id="run-0001", results=[
        TestResult(test_id=t, output="o",
                   criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])
        for t in ("t1", "t2")])
    assert all(r.input_hash is None for r in legacy.results)

    d = report_dict(project, analyze_coverage(project.spec, project.tests), legacy)
    assert d["ungraded_tests"] == [] and d["suite_pass_rate"] == 1.0


def test_an_unreadable_golden_fails_the_gate_instead_of_skipping_it(tmp_path):
    """`load_golden` answers None for "absent" and for "corrupt" alike, so the gate
    could skip a pinned check that a bad edit or an unresolved merge conflict had
    silently disabled — turning a FAILING snapshot gate into a passing one."""
    from ai_calibrator.ci import run_ci
    from ai_calibrator.snapshot import GOLDEN_FILE

    (tmp_path / GOLDEN_FILE).write_text("<<<<<<< HEAD\n{}\n=======\n", encoding="utf-8")
    result = run_ci(_gate_project(), _Subject(), _PartialJudge("nothing matches"),
                    project_dir=tmp_path, threshold=0.8)

    stage = next(s for s in result.stages if s.name == "snapshot")
    assert stage.status == "fail" and "could not be read" in stage.detail
    assert not result.ok


def test_no_golden_at_all_still_skips(tmp_path):
    from ai_calibrator.ci import run_ci

    result = run_ci(_gate_project(), _Subject(), _PartialJudge("nothing matches"),
                    project_dir=tmp_path, threshold=0.8)
    stage = next(s for s in result.stages if s.name == "snapshot")
    assert stage.status == "skip" and result.ok


# --- the report must not launder an unreviewed guess as calibration --------

def _project_with_interview(sources):
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, InterviewItem, Project, Weight

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="on policy", weight=Weight.HIGH)])
    p.interview = [InterviewItem(id=f"q{i}", dimension=f"dim{i}", question="?",
                                 answer="an answer", answer_source=src)
                   for i, src in enumerate(sources, start=1)]
    return p


def test_a_spec_built_from_unreviewed_drafts_says_so_on_every_surface(tmp_path):
    """The tool cannot know what the materials leave unstated, so an auto-accepted
    draft can assert policy nobody wrote — and compile turns it into a standard
    and a graded criterion. A high score then measures agreement with the guess."""
    project = _project_with_interview(["engine", "engine", "human"])
    cov = analyze_coverage(project.spec, project.tests)

    d = report_dict(project, cov, None)
    assert d["unratified_answers"] == ["dim1", "dim2"]

    md = render_report(project, cov, None)
    assert "nobody reviewed" in md and "dim1" in md

    html = render_html_report(project, cov, None, tmp_path)
    assert "Spec provenance" in html and "unreviewed" in html


def test_a_ratified_spec_carries_no_such_warning(tmp_path):
    """Typing your own answer and accepting a draft you read are both decisions a
    person made; neither is the tool answering itself."""
    project = _project_with_interview(["human", "human_ratified", None])
    cov = analyze_coverage(project.spec, project.tests)

    assert report_dict(project, cov, None)["unratified_answers"] == []
    assert "nobody reviewed" not in render_report(project, cov, None)
    assert "Spec provenance" not in render_html_report(project, cov, None, tmp_path)


# --- the judge grades with the instructions the AI was given ----------------

def test_the_judge_sees_the_instructions_the_ai_was_given():
    """Without them a judge cannot grade any criterion that refers to them, and it
    does not abstain: in a live run "cites only policies stated in this spec"
    failed answers for stating policy the materials really did contain, with
    confident rationales and a 0% headline."""
    from ai_calibrator.eval import JUDGE_SYSTEM, judge_system

    assert judge_system() == JUDGE_SYSTEM       # nothing to add, nothing added
    assert judge_system("   ") == JUDGE_SYSTEM  # and blank is nothing

    withrules = judge_system("Refunds are issued within 5 business days.")
    assert JUDGE_SYSTEM in withrules            # still told how to grade
    assert "Refunds are issued within 5 business days." in withrules
    assert "NOT invented" in withrules          # and told what that means
    assert "not criteria" in withrules          # without mistaking them for criteria


def test_the_trainer_reproduces_the_system_message_the_judge_graded_under(tmp_path):
    """A ground-truth row must carry the identical system message, or the local
    judge trains on a distribution the cloud judge never graded under."""
    from ai_calibrator.compile import render_system_prompt
    from ai_calibrator.eval import judge_system
    from ai_calibrator.judge_check import save_labels
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project
    from ai_calibrator.store import save_project
    from ai_calibrator.train_engine import human_judge_rows

    p = Project(name="p", goal="answer returns questions")
    p.spec = BehaviorSpec(goal="answer returns questions",
                          standards=["Always cite the documented 30-day window."],
                          eval_criteria=[EvalCriterion(id="c1", description="cites the window",
                                                       weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="can I return this?", expects=["c1"])]
    save_project(p, tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[
        ResultModel(test_id="t1", output="the answer",
                    criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])]))
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c1", "passed": False}])

    rows = human_judge_rows(tmp_path)
    assert rows, "a human label should produce a ground-truth row"
    recorded = rows[0]["messages"][0]["content"]
    assert recorded == judge_system(render_system_prompt(p.spec))
    assert "30-day window" in recorded
