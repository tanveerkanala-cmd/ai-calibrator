"""The gate must never report success it did not earn.

A behavior spec is easy to produce; a spec the model demonstrably obeys is the
product. Every test here pins one way a pass rate, a drift verdict, a badge, or a
red-team result could claim more than actually happened — the failure class that
matters most, because the number is the user's only reason to trust the output.
"""

import json
import re

import pytest

from ai_calibrator.ci import run_ci
from ai_calibrator.eval import latest_run_id, run_eval, save_scorecard
from ai_calibrator.lint import lint_spec
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


class BoomJudge:
    """Any judge call is a failure of the test's premise."""
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        raise AssertionError("the judge must not be called here")


class PassJudge:
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "r"} for i in ids]}


class SilentSubject:
    """A content filter, a quantized local model, or a bad day: no answer at all."""
    name = "subject@test"

    def complete(self, prompt, *, system=None, schema=None):
        return ""


def _spec_with_checks() -> BehaviorSpec:
    return BehaviorSpec(
        goal="answer billing questions",
        eval_criteria=[
            EvalCriterion(id="no_claim", description="makes no medical claim", weight=Weight.HIGH,
                          check=Check(kind="not_contains", value="cure")),
            EvalCriterion(id="short", description="stays brief", weight=Weight.LOW,
                          check=Check(kind="max_chars", value="400")),
        ],
    )


def test_empty_output_fails_even_when_every_criterion_is_deterministic():
    """An AI that says nothing violates no negative-form check — and must still fail.

    not_contains and max_chars are both trivially satisfied by "", so without a
    blank-output guard ahead of the deterministic layer a silent subject scores a
    certified 100%.
    """
    p = Project(name="p", goal="g")
    p.spec = _spec_with_checks()
    p.tests = [CaseModel(id="t1", input="hi", expects=["no_claim"]),
               CaseModel(id="t2", input="yo", expects=["no_claim", "short"])]

    card = run_eval(p, SilentSubject(), BoomJudge())

    assert card.pass_rate == 0.0
    assert card.weighted_score == 0.0
    for r in card.results:
        assert not r.passed
        assert all(c.rationale == "empty output" for c in r.criteria)


def test_multi_turn_checks_grade_the_ai_not_the_user():
    """A check asks what the AI said. The user's words must never satisfy it."""
    class EchoUserSubject:
        name = "subject@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "understood"

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(
        goal="g",
        eval_criteria=[EvalCriterion(id="no_claim", description="no medical claim", weight=Weight.HIGH,
                                     check=Check(kind="not_contains", value="cure"))],
    )
    # The USER says "cure"; the assistant never does.
    p.tests = [CaseModel(id="t1", input="does this cure anything?",
                         follow_ups=["and is that a cure too?"], expects=["no_claim"])]

    card = run_eval(p, EchoUserSubject(), PassJudge())

    assert card.results[0].criteria[0].passed, "the user's word must not fail the AI's check"
    # The transcript is still what gets recorded and judged — context is preserved.
    assert "User: does this cure anything?" in card.results[0].output


def test_multi_turn_silent_assistant_still_fails():
    """The transcript is non-empty even when every reply is — grade the replies."""
    p = Project(name="p", goal="g")
    p.spec = _spec_with_checks()
    p.tests = [CaseModel(id="t1", input="hi", follow_ups=["still there?"], expects=["no_claim", "short"])]

    card = run_eval(p, SilentSubject(), BoomJudge())

    assert card.pass_rate == 0.0
    assert all(c.rationale == "empty output" for c in card.results[0].criteria)


def test_single_turn_eval_sends_the_prompt_the_runtime_serves():
    """What you certify must be what you deploy, down to the encoding."""
    seen: list[str] = []

    class Recorder:
        name = "subject@test"

        def complete(self, prompt, *, system=None, schema=None):
            seen.append(prompt)
            return "an answer"

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="is helpful", weight=Weight.MEDIUM)])
    p.tests = [CaseModel(id="t1", input="how long is the return window?", expects=["c1"])]

    run_eval(p, Recorder(), PassJudge())

    from ai_calibrator.runtime import encode_messages
    assert seen == ["User: how long is the return window?\nAssistant:"]
    assert seen[0] == encode_messages([{"role": "user", "content": "how long is the return window?"}])


def test_orphan_expectation_is_a_lint_error():
    """A test expecting a criterion the spec lost runs, costs money, and grades
    nothing — it must block certification rather than shrink the denominator."""
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="is accurate and cites the policy", weight=Weight.HIGH)])
    tests = [CaseModel(id="t1", input="a", expects=["c1"]),
             CaseModel(id="t2", input="b", expects=["c_safety"])]

    report = lint_spec(spec, tests)
    orphans = [i for i in report.issues if i.code == "orphan_expectation"]

    assert len(orphans) == 1
    assert orphans[0].severity == "error"
    assert "c_safety" in orphans[0].message and "t2" in orphans[0].message


def _full_card(run_id: str, n: int, passing: int) -> Scorecard:
    return Scorecard(run_id=run_id, results=[
        ResultModel(test_id=f"t{i}", output="o",
                   criteria=[CriterionResult(criterion_id="c1", passed=i < passing,
                                             score=1.0 if i < passing else 0.0)])
        for i in range(n)])


def test_partial_scorecard_never_becomes_the_drift_baseline(tmp_path):
    """A --max-tests smoke run as the baseline compares two different test sets,
    so every regression on a test it skipped reads as 'no regressions'."""
    save_scorecard(tmp_path, _full_card("run-0001", 10, 10))
    smoke = _full_card("run-0002", 2, 1)
    smoke.partial = True
    save_scorecard(tmp_path, smoke)

    assert latest_run_id(tmp_path) == "run-0002"
    assert latest_run_id(tmp_path, full_only=True) == "run-0001", "the smoke run is not a reference point"

    class Subject:
        name = "subject@test"

        def complete(self, prompt, *, system=None, schema=None):
            # Regress the tests the smoke run never covered.
            m = re.search(r"^User: q(\d+)$", prompt, re.M)
            return "BAD" if m and int(m.group(1)) >= 7 else "fine"

    class Judge:
        name = "judge@test"

        def complete(self, prompt, *, system=None, schema=None):
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            bad = "BAD" in prompt.split("CRITERIA")[0]
            return {"results": [{"criterion_id": i, "passed": not bad,
                                 "score": 0.0 if bad else 1.0, "rationale": "r"} for i in ids]}

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["Always cite the documented policy line."],
                          eval_criteria=[EvalCriterion(id="c1", description="cites the documented policy",
                                                       weight=Weight.HIGH)])
    p.tests = [CaseModel(id=f"t{i}", input=f"q{i}", expects=["c1"]) for i in range(10)]

    result = run_ci(p, Subject(), Judge(), project_dir=tmp_path, threshold=0.0)
    drift = next(s for s in result.stages if s.name == "drift")

    assert drift.status == "fail", "regressions vs the FULL baseline must surface"
    assert "run-0001" in drift.detail


def test_explicitly_pinned_partial_baseline_is_skipped_not_compared(tmp_path):
    save_scorecard(tmp_path, _full_card("run-0001", 10, 10))
    smoke = _full_card("run-0002", 2, 2)
    smoke.partial = True
    save_scorecard(tmp_path, smoke)

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["Always cite the documented policy line."],
                          eval_criteria=[EvalCriterion(id="c1", description="cites the documented policy",
                                                       weight=Weight.HIGH)])
    p.tests = [CaseModel(id=f"t{i}", input=f"q{i}", expects=["c1"]) for i in range(10)]

    class Subject:
        name = "subject@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "fine"

    result = run_ci(p, Subject(), PassJudge(), project_dir=tmp_path, threshold=0.0, baseline="run-0002")
    drift = next(s for s in result.stages if s.name == "drift")

    assert drift.status == "skip"
    assert "PARTIAL" in drift.detail


def test_badge_carries_the_certified_number_not_the_latest_smoke_run(tmp_path):
    """A green badge must show what the gate certified. A --max-tests run after a
    passing gate would otherwise publish its own 100% in the gate's colour."""
    from ai_calibrator.report import badge_dict

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["Always cite the documented policy line."],
                          eval_criteria=[EvalCriterion(id="c1", description="cites the documented policy",
                                                       weight=Weight.HIGH)])
    p.tests = [CaseModel(id=f"t{i}", input=f"q{i}", expects=["c1"]) for i in range(10)]

    class Subject:
        name = "subject@test"

        def complete(self, prompt, *, system=None, schema=None):
            m = re.search(r"^User: q(\d+)$", prompt, re.M)
            return "BAD" if m and int(m.group(1)) == 9 else "fine"

    class Judge:
        name = "judge@test"

        def complete(self, prompt, *, system=None, schema=None):
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            bad = "BAD" in prompt.split("CRITERIA")[0]
            return {"results": [{"criterion_id": i, "passed": not bad,
                                 "score": 0.0 if bad else 1.0, "rationale": "r"} for i in ids]}

    gate = run_ci(p, Subject(), Judge(), project_dir=tmp_path, threshold=0.8)
    assert gate.ok and gate.pass_rate == pytest.approx(0.9)

    # A one-test smoke run lands afterwards and becomes the newest scorecard.
    smoke = _full_card("run-9999", 1, 1)
    smoke.partial = True
    save_scorecard(tmp_path, smoke)

    badge = badge_dict(p, tmp_path)
    assert badge["color"] == "brightgreen"
    assert badge["message"].startswith("90%"), badge
    assert "1 tests" not in badge["message"]

    persisted = json.loads((tmp_path / "evals" / "last-gate.json").read_text(encoding="utf-8"))
    assert persisted["pass_rate"] == pytest.approx(0.9)


def test_zero_probes_is_not_a_perfect_red_team_hold():
    """Nothing was attacked, so nothing held."""
    from ai_calibrator.redteam import RedTeamReport

    assert RedTeamReport(run_id="redteam-0001", results=[]).hold_rate == 0.0
