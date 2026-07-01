"""M4 eval + refine loop, verified with mocked subject/judge/refiner engines."""

import re

import pytest

from calibrator.eval import low_confidence_results, next_run_id, run_eval, save_scorecard
from calibrator.models import (
    BehaviorSpec,
    EvalCriterion,
    Project,
    Weight,
)
from calibrator.models import TestCase as CaseModel  # aliased: avoids pytest collecting the model
from calibrator.pipeline import calibrate_loop, refine_spec

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


def test_pass_rate_excludes_ungradeable_tests():
    from calibrator.models import CriterionResult, Scorecard, TestResult
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
    it's coerced to empty and caught by the empty-output guard. (stress finding)"""
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
    from calibrator.models import Check

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
