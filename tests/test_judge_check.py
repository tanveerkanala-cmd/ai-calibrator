"""Judge calibration — measure judge↔human agreement from saved verdicts."""

from calibrator.judge_check import agreement_dict, gradings, judge_agreement
from calibrator.models import CriterionResult, Scorecard
from calibrator.models import TestResult as Result


def _card():
    return Scorecard(run_id="r", results=[
        Result(test_id="t1", output="o1", criteria=[
            CriterionResult(criterion_id="c1", passed=True, rationale="ok"),
            CriterionResult(criterion_id="c2", passed=False, rationale="no")]),
        Result(test_id="t2", output="o2", criteria=[CriterionResult(criterion_id="c1", passed=True)]),
    ])


def test_gradings_flattens_all_verdicts():
    g = gradings(_card())
    assert len(g) == 3
    assert {(x["test_id"], x["criterion_id"]) for x in g} == {("t1", "c1"), ("t1", "c2"), ("t2", "c1")}


def test_judge_agreement_overall_and_per_criterion():
    labels = [
        {"test_id": "t1", "criterion_id": "c1", "passed": True},    # judge True  → agree
        {"test_id": "t1", "criterion_id": "c2", "passed": True},    # judge False → disagree
        {"test_id": "t2", "criterion_id": "c1", "passed": True},    # judge True  → agree
    ]
    ag = judge_agreement(_card(), labels)
    assert ag.total == 3 and ag.agreed == 2
    assert abs(ag.agreement_rate - 2 / 3) < 1e-9
    assert ag.by_criterion["c1"] == (2, 2) and ag.by_criterion["c2"] == (0, 1)
    assert ag.unreliable_criteria() == ["c2"]
    assert ag.disagreements[0]["criterion_id"] == "c2" and ag.disagreements[0]["judge"] is False


def test_unknown_labels_are_ignored():
    assert judge_agreement(_card(), [{"test_id": "nope", "criterion_id": "x", "passed": True}]).total == 0


def test_agreement_dict_shape():
    d = agreement_dict(judge_agreement(_card(), [{"test_id": "t1", "criterion_id": "c1", "passed": True}]))
    assert d["agreement_rate"] == 1.0 and d["by_criterion"]["c1"]["rate"] == 1.0 and d["unreliable_criteria"] == []
