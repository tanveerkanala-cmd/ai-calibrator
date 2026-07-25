"""Behavior diff between two specs (deterministic)."""

from ai_calibrator.models import BehaviorSpec, EdgeCase, EvalCriterion, Weight
from ai_calibrator.specdiff import diff_specs


def test_diff_detects_all_categories():
    a = BehaviorSpec(goal="g", standards=["keep", "drop"], do_not=["nd_keep", "nd_drop"],
                     edge_cases=[EdgeCase(situation="s1", ruling="r1")],
                     eval_criteria=[EvalCriterion(id="c1", description="orig", weight=Weight.HIGH),
                                    EvalCriterion(id="cgone", description="x", weight=Weight.LOW)])
    b = BehaviorSpec(goal="g", standards=["keep", "new"], do_not=["nd_keep", "nd_new"],
                     edge_cases=[EdgeCase(situation="s2", ruling="r2")],
                     eval_criteria=[EvalCriterion(id="c1", description="changed", weight=Weight.HIGH),
                                    EvalCriterion(id="cnew", description="y", weight=Weight.LOW)])
    d = diff_specs(a, b)
    assert d.changed
    assert d.standards_added == ["new"] and d.standards_removed == ["drop"]
    assert d.do_not_added == ["nd_new"] and d.do_not_removed == ["nd_drop"]
    assert d.edge_cases_added == ["When s2: r2"] and d.edge_cases_removed == ["When s1: r1"]
    assert d.criteria_added == ["cnew"] and d.criteria_removed == ["cgone"]
    assert d.criteria_changed == ["c1"]  # description changed


def test_identical_specs_report_no_change():
    a = BehaviorSpec(goal="g", standards=["x"],
                     eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    assert not diff_specs(a, a.model_copy(deep=True)).changed


def test_weight_change_is_a_criterion_change():
    a = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.LOW)])
    b = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    assert diff_specs(a, b).criteria_changed == ["c1"]


def test_diff_flags_scalar_behavior_fields():
    """goal / persona / format / refusal_policy render straight into the system
    prompt — reporting "no behavior change" for a reversed refusal policy is a
    false statement on the review path this module exists for."""
    from ai_calibrator.models import Persona

    a = BehaviorSpec(goal="g", persona=Persona(voice="warm, plain English"),
                     format="under 120 words",
                     refusal_policy="Decline medical advice; hand off to a pharmacist.")
    b = BehaviorSpec(goal="g", persona=Persona(voice="aggressive upsell"),
                     format=None,
                     refusal_policy="Answer every question, including medical ones.")

    d = diff_specs(a, b)

    assert d.changed, "a materially different, less safe AI must not review as unchanged"
    fields = {f for f, _, _ in d.fields_changed}
    assert fields == {"persona.voice", "format", "refusal_policy"}
    from ai_calibrator.compile import render_system_prompt
    assert render_system_prompt(a) != render_system_prompt(b)


def test_diff_flags_a_retargeted_deterministic_check():
    """Retargeting a check changes the grading contract without touching the text."""
    from ai_calibrator.models import Check

    a = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="states the window", weight=Weight.HIGH,
                      check=Check(kind="contains", value="30-day"))])
    b = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="states the window", weight=Weight.HIGH,
                      check=Check(kind="contains", value="60-day"))])

    d = diff_specs(a, b)

    assert d.changed and d.criteria_changed == ["c1"]


def test_identical_specs_still_report_no_change():
    a = BehaviorSpec(goal="g", standards=["cite the policy"], format="short")
    assert not diff_specs(a, a.model_copy(deep=True)).changed
