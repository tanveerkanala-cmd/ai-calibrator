"""M4 eval + refine loop, verified with mocked subject/judge/refiner engines."""

import re

import pytest

from ai_calibrator.eval import low_confidence_results, next_run_id, run_eval, save_scorecard
from ai_calibrator.models import (
    BehaviorSpec,
    EvalCriterion,
    Project,
    Weight,
)
from ai_calibrator.models import TestCase as CaseModel  # aliased: avoids pytest collecting the model
from ai_calibrator.pipeline import calibrate_loop, refine_spec

# Marker convention: the subject emits "ACCEPTABLE" for a good answer; the judge
# passes a criterion iff the output it sees contains that marker.


class PassJudge:
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        good = "ACCEPTABLE" in prompt
        return {
            "results": [
                {"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0,
                 "rationale": "meets it" if good else "does not meet it"}
                for i in ids
            ]
        }


class GoodSubject:
    name = "subject@test"
    def complete(self, prompt, *, system=None, schema=None):
        return "ACCEPTABLE answer"


class EmptySubject:
    name = "subject@test"
    def complete(self, prompt, *, system=None, schema=None):
        return ""


class RefineAwareSubject:
    """Bad until a 'REFINED' standard shows up in the (re-rendered) system prompt."""
    name = "subject@test"
    def complete(self, prompt, *, system=None, schema=None):
        return "ACCEPTABLE answer" if system and "REFINED" in system else "POOR answer"


class Refiner:
    name = "refiner@test"
    def complete(self, prompt, *, system=None, schema=None):
        return {"new_standards": ["REFINED: always do the acceptable thing"]}


def _project(standards=None):
    p = Project(name="t", goal="g")
    p.spec = BehaviorSpec(
        goal="g",
        standards=standards or [],
        eval_criteria=[EvalCriterion(id="c1", description="answer is on-policy", weight=Weight.HIGH)],
    )
    p.tests = [CaseModel(id="t1", input="a question", expects=["c1"])]
    return p


def test_run_eval_all_pass():
    card = run_eval(_project(), GoodSubject(), PassJudge(), run_id="run-0001")
    assert card.pass_rate == 1.0
    assert card.results[0].passed


def test_run_eval_records_provenance():
    card = run_eval(_project(), GoodSubject(), PassJudge(), run_id="run-0001")
    # provenance so the prove-it gate can verify which models produced a run
    assert card.subject == GoodSubject().name and card.judge == PassJudge().name
    assert card.created_at is not None and card.tool_version is not None
    assert card.partial is False


def test_run_eval_max_tests_marks_partial_and_progress():
    proj = _project()
    proj.tests = proj.tests * 3  # a few tests
    seen = []
    card = run_eval(proj, GoodSubject(), PassJudge(), run_id="r",
                    max_tests=1, on_progress=lambda d, t, tid: seen.append((d, t)))
    assert len(card.results) == 1
    assert card.partial is True          # a capped run is not a full pass
    assert seen == [(1, 1)]


def test_pass_rate_excludes_ungradeable_tests():
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult
    card = Scorecard(run_id="r", results=[
        TestResult(test_id="graded", output="x", criteria=[CriterionResult(criterion_id="c", passed=True)]),
        TestResult(test_id="ungradeable", output="x", criteria=[]),  # no criteria → not counted
    ])
    assert card.pass_rate == 1.0  # the ungradeable test must NOT drag this to 0.5


def test_empty_output_fails_without_calling_judge():
    class SpyJudge(PassJudge):
        called = False
        def complete(self, *a, **k):
            SpyJudge.called = True
            return super().complete(*a, **k)

    card = run_eval(_project(), EmptySubject(), SpyJudge(), run_id="r")
    assert card.pass_rate == 0.0
    assert SpyJudge.called is False  # deterministic guard skipped the judge


def test_refine_spec_parses_new_standards():
    project = _project()
    card = run_eval(project, RefineAwareSubject(), PassJudge(), run_id="r")  # fails (POOR)
    assert card.pass_rate == 0.0
    assert refine_spec(project, card, Refiner()) == ["REFINED: always do the acceptable thing"]


def test_calibrate_loop_improves_pass_rate():
    project = _project()
    cards = calibrate_loop(
        project, RefineAwareSubject(), PassJudge(), Refiner(),
        threshold=1.0, max_rounds=3, project_dir=None,
    )
    assert len(cards) == 2
    assert cards[0].pass_rate == 0.0   # round 1: subject is POOR
    assert cards[1].pass_rate == 1.0   # round 2: refined standard flowed into the prompt
    assert any("REFINED" in s for s in project.spec.standards)


def test_save_scorecard_and_run_ids(tmp_path):
    card = run_eval(_project(), GoodSubject(), PassJudge(), run_id=next_run_id(tmp_path))
    assert card.run_id == "run-0001"
    d = save_scorecard(tmp_path, card)
    assert (d / "scorecard.json").exists()
    assert (d / "failures.jsonl").exists()
    assert next_run_id(tmp_path) == "run-0002"


def test_run_eval_tolerates_non_string_subject_output():
    """A misbehaving subject that returns a non-string must not crash run_eval —
    it's coerced to empty and caught by the empty-output guard."""
    class WeirdSubject:
        name = "weird@test"

        def complete(self, prompt, *, system=None, schema=None):
            return 12345  # truthy non-string

    card = run_eval(_project(), WeirdSubject(), PassJudge(), run_id="run-0001")  # must NOT raise
    assert card.results[0].output == "" and card.pass_rate == 0.0


class FlakyJudge:
    """Returns pass, fail, pass on successive calls — a split (noisy) judge."""
    name = "flaky@test"

    def __init__(self):
        self.n = 0

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        self.n += 1
        good = self.n % 2 == 1
        return {"results": [{"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0, "rationale": "x"}
                            for i in ids]}


def test_judge_consensus_majority_vote_and_confidence():
    card = run_eval(_project(), GoodSubject(), FlakyJudge(), run_id="r", judge_passes=3)
    cr = card.results[0].criteria[0]
    assert cr.passed is True                    # verdicts pass/fail/pass → 2/3 majority pass
    assert cr.confidence == round(2 / 3, 3)     # 2/3 agreement
    low = low_confidence_results(card, threshold=0.9)
    assert low and low[0][1].criterion_id == "c1"  # 0.67 < 0.9 → flagged for review


def test_single_pass_has_no_confidence():
    card = run_eval(_project(), GoodSubject(), PassJudge(), run_id="r")  # judge_passes default 1
    assert card.results[0].criteria[0].confidence is None
    assert low_confidence_results(card) == []


def test_run_eval_rejects_bad_judge_passes():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            run_eval(_project(), GoodSubject(), PassJudge(), judge_passes=bad)


def test_deterministic_check_is_graded_by_code_not_judge():
    from ai_calibrator.models import Check

    class SpyJudge(PassJudge):
        called = False

        def complete(self, *a, **k):
            SpyJudge.called = True
            return super().complete(*a, **k)

    class Subject:
        name = "subject@test"

        def __init__(self, text):
            self.text = text

        def complete(self, prompt, *, system=None, schema=None):
            return self.text

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="mentions 30-day", weight=Weight.HIGH,
                      check=Check(kind="contains", value="30-day"))])
    p.tests = [CaseModel(id="t1", input="q", expects=["c1"])]

    card = run_eval(p, Subject("our 30-day return policy"), SpyJudge(), run_id="r")
    assert card.results[0].criteria[0].passed is True
    assert SpyJudge.called is False  # graded deterministically — the judge was never called

    card2 = run_eval(p, Subject("no policy here"), PassJudge(), run_id="r2")
    assert card2.results[0].criteria[0].passed is False


def test_mixed_criterion_order_is_preserved():
    """Results must follow test.expects order even when checked + judged criteria mix
    (regression: the deterministic/judged split used to reorder to [checked, judged])."""
    from ai_calibrator.models import Check

    class Subject:
        name = "subject@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "ACCEPTABLE — our 30-day return policy is friendly"

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="is friendly", weight=Weight.MEDIUM),    # judged
        EvalCriterion(id="c2", description="mentions 30-day", weight=Weight.HIGH,    # checked
                      check=Check(kind="contains", value="30-day")),
        EvalCriterion(id="c3", description="is concise", weight=Weight.LOW),         # judged
    ])
    p.tests = [CaseModel(id="t1", input="q", expects=["c1", "c2", "c3"])]

    card = run_eval(p, Subject(), PassJudge(), run_id="r")
    assert [c.criterion_id for c in card.results[0].criteria] == ["c1", "c2", "c3"]
    c2 = next(c for c in card.results[0].criteria if c.criterion_id == "c2")
    assert c2.passed is True and "30-day" in (c2.rationale or "")  # graded exactly by code


def test_run_eval_multi_turn_conversation():
    prompts_seen = []

    class ConvoSubject:
        name = "subject@test"

        def complete(self, prompt, *, system=None, schema=None):
            prompts_seen.append(prompt)
            return f"reply{len(prompts_seen)}"

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="first?", expects=["c1"], follow_ups=["second?", "third?"])]

    card = run_eval(p, ConvoSubject(), PassJudge(), run_id="r")
    assert len(prompts_seen) == 3                              # 3 user turns → 3 subject calls
    assert "first?" in prompts_seen[1] and "reply1" in prompts_seen[1]  # history carried forward
    assert "third?" in prompts_seen[2]
    out = card.results[0].output                              # graded output is the full transcript
    assert "User: first?" in out and "Assistant: reply1" in out and "User: third?" in out


def test_weighted_score_hand_math_and_weight_stamping():
    """Weighted score = Σ(w·score)/Σ(w), and each verdict records the weight it
    was graded under (scorecard stays honest if spec weights change later)."""
    class Judge:
        name = "j@test"

        def complete(self, prompt, *, system=None, schema=None):
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            # c_high fails (score 0), everything else passes (score 1)
            return {"results": [
                {"criterion_id": i, "passed": i != "c_high", "score": 0.0 if i == "c_high" else 1.0,
                 "rationale": "r"} for i in ids]}

    class Subject:
        name = "s@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "output"

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c_high", description="d", weight=Weight.HIGH),
        EvalCriterion(id="c_med", description="d", weight=Weight.MEDIUM),
        EvalCriterion(id="c_low", description="d", weight=Weight.LOW),
    ])
    p.tests = [CaseModel(id="t1", input="q", expects=["c_high", "c_med", "c_low"])]

    card = run_eval(p, Subject(), Judge(), run_id="r")
    r = card.results[0]
    assert [c.weight for c in r.criteria] == [Weight.HIGH, Weight.MEDIUM, Weight.LOW]  # stamped
    # hand math: (3*0 + 2*1 + 1*1) / (3+2+1) = 3/6 = 0.5
    assert r.weighted_score == pytest.approx(0.5)
    assert card.weighted_score == pytest.approx(0.5)
    assert r.passed is False                     # binary pass/fail unchanged
    # scorecard round-trips the weight
    from ai_calibrator.models import Scorecard
    again = Scorecard.model_validate_json(card.model_dump_json())
    assert again.results[0].criteria[0].weight == Weight.HIGH
    assert again.weighted_score == pytest.approx(0.5)


def test_weighted_score_backward_compat_unweighted_scorecard():
    """Old scorecards (no recorded weight) score as all-medium — same relative math."""
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult
    old = Scorecard(run_id="r", results=[TestResult(test_id="t", output="o", criteria=[
        CriterionResult(criterion_id="a", passed=True, score=1.0),   # weight=None
        CriterionResult(criterion_id="b", passed=False, score=0.0),
    ])])
    assert old.results[0].weighted_score == pytest.approx(0.5)  # (2*1+2*0)/4
    assert Scorecard(run_id="r", results=[]).weighted_score == 0.0


def test_judge_consensus_even_split_is_not_a_majority():
    """'yes * 2 > passes' vs '>=' — a 1-of-2 tie must FAIL (strict majority),
    or a flaky judge could certify on a coin flip."""
    card = run_eval(_project(), GoodSubject(), FlakyJudge(), run_id="r", judge_passes=2)
    cr = card.results[0].criteria[0]
    assert cr.passed is False            # pass/fail tie → not a majority → fail
    assert cr.confidence == 0.5


def test_refine_loop_never_duplicates_standards():
    """A refiner proposing the same standard every round (or twice in one
    batch, or one that already exists as a never-rule) must not bloat the spec."""
    class RepeatingRefiner:
        name = "r@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"new_standards": ["SAME_STD", "SAME_STD", "ALREADY_A_NEVER_RULE"]}

    project = _project()
    project.spec.do_not = ["ALREADY_A_NEVER_RULE"]
    cards = calibrate_loop(project, RefineAwareSubject(), PassJudge(), RepeatingRefiner(),
                           threshold=1.0, max_rounds=4, project_dir=None)
    assert project.spec.standards.count("SAME_STD") == 1          # once, ever
    assert "ALREADY_A_NEVER_RULE" not in project.spec.standards   # cross-list guard
    # round 2's refine returns nothing NEW → the loop stops instead of spinning
    assert len(cards) == 2


class StringFalseJudge:
    """A non-compliant judge that emits the STRING 'false' for the verdict —
    bool('false') is True, which must NOT grade the criterion as passing."""
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        return {"results": [{"criterion_id": i, "passed": "false", "score": 0.0} for i in ids]}


def test_judge_string_false_does_not_pass():
    # Regression: the pass/fail boolean must be strictly coerced (as_bool), not
    # bool()-truthy — else "false" inflates the score to a perfect pass.
    card = run_eval(_project(), GoodSubject(), StringFalseJudge(), run_id="r")
    assert card.pass_rate == 0.0
    assert card.results[0].criteria[0].passed is False


def test_duplicate_expects_do_not_multiply_weight():
    # A test that lists the same criterion id 3× plus a distinct failing one must
    # weight each criterion ONCE, not once-per-occurrence.
    from ai_calibrator.eval import run_eval as _re
    p = Project(name="t", goal="g")
    p.spec = BehaviorSpec(
        goal="g",
        eval_criteria=[
            EvalCriterion(id="lo", description="d", weight=Weight.LOW),
            EvalCriterion(id="hi", description="d", weight=Weight.HIGH),
        ],
    )
    # "lo" repeated 3×; grading: lo passes (marker present logic below), hi fails.
    p.tests = [CaseModel(id="t1", input="q", expects=["lo", "lo", "lo", "hi"])]

    class SplitJudge:
        name = "judge@test"
        def complete(self, prompt, *, system=None, schema=None):
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            return {"results": [
                {"criterion_id": i, "passed": (i == "lo"), "score": 1.0 if i == "lo" else 0.0}
                for i in ids
            ]}

    card = _re(p, GoodSubject(), SplitJudge(), run_id="r")
    tr = card.results[0]
    # exactly two criteria recorded (lo once, hi once), not four
    assert [c.criterion_id for c in tr.criteria] == ["lo", "hi"]
