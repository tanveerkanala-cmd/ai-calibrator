"""Honest percentage formatting — displayed numbers must never contradict truth."""

from ai_calibrator.fmt import pct, pct_delta


def test_pct_poles_are_exact_only():
    assert pct(1.0) == "100%" and pct(0.0) == "0%"
    assert pct(249 / 250) == ">99%"     # 1 failing test must not show 100%
    assert pct(0.999) == ">99%"
    assert pct(1 / 250) == "<1%"        # and 1 passing test must not show 0%
    assert pct(0.5) == "50%" and pct(0.83) == "83%"


def test_pct_delta_never_masks_a_real_change():
    assert pct_delta(0.0) == "±0%"
    assert pct_delta(-0.004) == "-0.4%"      # -0.4% must not show ±0%
    assert pct_delta(0.02) == "+2.0%"
    assert pct_delta(-0.0004) == "-<0.1%"    # tiny but real
    assert pct_delta(0.0004) == "+<0.1%"


def test_ci_eval_detail_is_honest_at_249_of_250(tmp_path):
    """Integration: a scorecard with one failure among 250 must render >99%, and a
    just-below-threshold fail must disambiguate the boundary."""
    from ai_calibrator.ci import _drift_stage, run_ci  # noqa: F401  (drift smoke below)
    from ai_calibrator.ci import CiResult
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import (BehaviorSpec, CriterionResult, EvalCriterion, Project,
                                   Scorecard, TestCase, TestResult, Weight)

    import re

    class Judge:
        name = "j@test"

        def complete(self, prompt, *, system=None, schema=None):
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            fail = "UNLUCKY" in prompt
            return {"results": [{"criterion_id": i, "passed": not fail,
                                 "score": 0.0 if fail else 1.0, "rationale": "r"} for i in ids]}

    class Subject:
        name = "s@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "UNLUCKY answer" if prompt == "q249" else "fine"

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["Always answer with the documented policy."],
                          refusal_policy="decline medical questions",
                          eval_criteria=[EvalCriterion(id="c1", description="matches the documented policy",
                                                       weight=Weight.HIGH)])
    p.tests = [TestCase(id=f"t{i}", input=f"q{i}", expects=["c1"]) for i in range(250)]

    result: CiResult = run_ci(p, Subject(), Judge(), project_dir=tmp_path, threshold=0.8)
    eval_stage = next(s for s in result.stages if s.name == "eval")
    assert ">99%" in eval_stage.detail and "100%" not in eval_stage.detail.split("threshold")[0]
    assert "(249/250)" in eval_stage.detail

    # badge honesty on the same card
    from ai_calibrator.report import badge_dict
    b = badge_dict(p, tmp_path)
    assert b["message"].startswith(">99%")

    # boundary disambiguation: 199/250 = 79.6% vs threshold 80% must show the real numbers
    card = Scorecard(run_id="run-9998", results=[
        TestResult(test_id=f"t{i}", output="o",
                   criteria=[CriterionResult(criterion_id="c1", passed=i < 199, score=1.0 if i < 199 else 0.0)])
        for i in range(250)])
    save_scorecard(tmp_path, card)

    class Judge199:
        name = "j@test"

        def complete(self, prompt, *, system=None, schema=None):
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            fail = any(f"q{i} " in prompt.split("AI OUTPUT")[0] or prompt.split("\n")[1] == f"q{i}"
                       for i in range(199, 250))
            return {"results": [{"criterion_id": i2, "passed": not fail,
                                 "score": 0.0 if fail else 1.0, "rationale": "r"} for i2 in ids]}

    class Subject199:
        name = "s@test"

        def complete(self, prompt, *, system=None, schema=None):
            n = int(prompt[1:]) if prompt.startswith("q") and prompt[1:].isdigit() else 0
            return "UNLUCKY" if n >= 199 else "fine"

    r2 = run_ci(p, Subject199(), Judge(), project_dir=tmp_path, threshold=0.8)
    stage = next(s for s in r2.stages if s.name == "eval")
    assert stage.status == "fail"
    assert "below threshold (79.6% < 80%)" in stage.detail
