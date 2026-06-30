"""M4 eval + refine loop, verified with mocked subject/judge/refiner engines."""

import re

from calibrator.eval import next_run_id, run_eval, save_scorecard
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
