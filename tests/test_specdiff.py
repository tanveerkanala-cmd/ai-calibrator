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
