"""Spec-lint — proactive quality checks on a behavior spec."""

from calibrator.lint import lint_contradictions, lint_spec
from calibrator.models import BehaviorSpec, EvalCriterion, Weight
from calibrator.models import TestCase as Case


def test_clean_spec_has_no_errors():
    spec = BehaviorSpec(
        goal="g", standards=["Always cite the 30-day return window."],
        do_not=["Never promise refunds we don't offer."], refusal_policy="Decline politely and redirect.",
        eval_criteria=[EvalCriterion(id="cite", description="cites the policy window", weight=Weight.HIGH)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["cite"])])
    assert r.ok and not r.errors


def test_no_criteria_is_error():
    r = lint_spec(BehaviorSpec(goal="g", standards=["Be concise and clear always."]), [])
    assert not r.ok and any(i.code == "no_criteria" for i in r.errors)


def test_untested_criterion_warns():
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="something objectively testable", weight=Weight.HIGH)])
    r = lint_spec(spec, [])  # nothing targets c1
    assert any(i.code == "untested_criterion" and i.where == "c1" for i in r.issues)


def test_vague_and_short_standards_flagged():
    spec = BehaviorSpec(goal="g", standards=["ok", "Be helpful and appropriate as needed"],
                        eval_criteria=[EvalCriterion(id="c1", description="x is clearly satisfied", weight=Weight.HIGH)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["c1"])])
    assert "vague_standard" in {i.code for i in r.issues}  # "ok" too short + weasel words


def test_duplicate_criterion_is_error():
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="dup", description="first definition here", weight=Weight.HIGH),
        EvalCriterion(id="dup", description="second definition here", weight=Weight.LOW)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["dup"])])
    assert any(i.code == "duplicate_criterion" for i in r.errors)


def test_never_rules_without_refusal_policy_is_info():
    spec = BehaviorSpec(goal="g", do_not=["Never give medical advice."],
                        eval_criteria=[EvalCriterion(id="c1", description="gives no medical advice", weight=Weight.HIGH)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["c1"])])
    assert any(i.code == "no_refusal_policy" for i in r.issues)


def test_lint_contradictions_reuses_conflict_detector():
    class ConflictEngine:
        name = "ce@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"conflicts": [{"a": 1, "b": 2, "explanation": "cannot both hold", "severity": "high"}]}

    spec = BehaviorSpec(goal="g", standards=["Always be brief.", "Always explain in great detail."])
    issues = lint_contradictions(spec, ConflictEngine())
    assert len(issues) == 1 and issues[0].code == "self_contradiction" and issues[0].severity == "error"
