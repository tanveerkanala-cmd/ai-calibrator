"""Model rightsizing — cheapest model that meets the bar, across the suite."""

import re

from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from ai_calibrator.models import TestCase as Case
from ai_calibrator.rightsize import DEFAULT_LADDER, rightsize


class MarkerSubject:
    def __init__(self, marker):
        self.marker = marker
        self.name = "subject@test"

    def complete(self, prompt, *, system=None, schema=None):
        return self.marker


class PassJudge:
    """Passes a criterion iff the AI output (echoed into the prompt) says GOOD."""
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        good = "GOOD" in prompt
        return {"results": [
            {"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0, "rationale": ""}
            for i in ids
        ]}


def _project():
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="on-policy", weight=Weight.HIGH)])
    p.tests = [Case(id="t1", input="q", expects=["c1"])]
    return p


def _factory(passing_models: set):
    def make(spec):
        model = spec.split("@")[0]
        return MarkerSubject("GOOD answer" if model in passing_models else "BAD answer")
    return make


def test_recommends_cheapest_passing(tmp_path):
    factory = _factory({"claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"})  # all pass
    report = rightsize(_project(), list(DEFAULT_LADDER), PassJudge(), factory, threshold=0.8, project_dir=tmp_path)
    assert all(r.pass_rate == 1.0 for r in report.results)
    assert report.recommended.model == "claude-haiku-4-5"  # cheapest of the passers
    assert (tmp_path / "evals" / "rightsize.json").exists()


def test_respects_the_bar(tmp_path):
    factory = _factory({"claude-opus-4-8", "claude-sonnet-4-6"})  # haiku fails
    report = rightsize(_project(), list(DEFAULT_LADDER), PassJudge(), factory, threshold=0.8)
    assert report.recommended.model == "claude-sonnet-4-6"  # cheapest that still meets the bar
    assert "claude-haiku-4-5" not in {r.model for r in report.passing}


def test_records_engine_errors_without_aborting():
    def bad(spec):
        raise RuntimeError("no creds")
    report = rightsize(_project(), ["x@anthropic"], PassJudge(), bad, threshold=0.8)
    assert report.results[0].error == "no creds"
    assert report.recommended is None  # nothing passed → no recommendation


def test_unknown_model_priced_as_none_but_still_rankable():
    factory = _factory({"mystery"})
    report = rightsize(_project(), ["mystery@ollama"], PassJudge(), factory, threshold=0.5)
    r = report.results[0]
    assert r.in_price is None and r.cost_score is None
    assert report.recommended.model == "mystery"  # passes → recommended despite unknown cost




def test_free_local_candidate_beats_a_paid_one_that_also_met_the_bar():
    """"Cheapest that meets the bar" has to mean it: a local model bills nothing
    per token, so no priced cloud model can be the cheaper recommendation."""
    factory = _factory({"claude-haiku-4-5", "local-7b"})
    report = rightsize(_project(), ["claude-haiku-4-5@anthropic", "local-7b@ollama"],
                       PassJudge(), factory, threshold=0.8)
    assert {r.model for r in report.passing} == {"claude-haiku-4-5", "local-7b"}
    assert report.recommended.model == "local-7b"
    assert report.results[0].local is False and report.results[1].local is True




def test_free_local_model_beats_a_paid_one_that_also_passes():
    """A local candidate has no per-token bill, so nothing on a price list can
    undercut it — recommending the paid model bills the owner for nothing."""
    factory = _factory({"claude-haiku-4-5", "llama3"})   # both clear the bar
    report = rightsize(_project(), ["claude-haiku-4-5@anthropic", "llama3@ollama"],
                       PassJudge(), factory, threshold=0.8)
    assert {r.model for r in report.passing} == {"claude-haiku-4-5", "llama3"}
    assert report.recommended.model == "llama3"
    assert report.recommended.local is True
