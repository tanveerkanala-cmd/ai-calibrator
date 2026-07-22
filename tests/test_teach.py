"""Calibrate-by-example — candidate proposal, standard inference, application."""

from ai_calibrator.models import BehaviorSpec, Project
from ai_calibrator.models import TestCase as Case
from ai_calibrator.teach import Judged, apply_learned, infer_standards, propose_candidates


class GenEngine:
    """Generates inputs (schema has 'inputs') and infers standards (schema has 'standards')."""
    name = "gen@test"

    def complete(self, prompt, *, system=None, schema=None):
        props = (schema or {}).get("properties", {})
        if "inputs" in props:
            return {"inputs": ["scenario A?", "scenario B?", "scenario C?"]}
        if "standards" in props:
            return {"standards": ["Always cite the policy."], "do_not": ["Never promise refunds."]}
        return "fallback"


class Subject:
    name = "subject@test"

    def complete(self, prompt, *, system=None, schema=None):
        return f"answer to: {prompt}"


def test_propose_reuses_existing_test_inputs():
    p = Project(name="p", goal="g")
    p.tests = [Case(id="t1", input="real one"), Case(id="t2", input="real two")]
    cands = propose_candidates(p, GenEngine(), Subject(), n=2)
    assert [c.input for c in cands] == ["real one", "real two"]
    assert all(c.output.startswith("answer to:") for c in cands)


def test_propose_generates_when_no_tests():
    cands = propose_candidates(Project(name="p", goal="g"), GenEngine(), Subject(), n=3)
    assert len(cands) == 3 and cands[0].input == "scenario A?"


def test_propose_tops_up_existing_with_generated():
    p = Project(name="p", goal="g")
    p.tests = [Case(id="t1", input="real one")]
    cands = propose_candidates(p, GenEngine(), Subject(), n=3)
    assert cands[0].input == "real one" and len(cands) == 3  # 1 real + 2 generated


def test_infer_standards_from_judgments():
    judged = [Judged(input="i", output="o", approved=True, reason="good"),
              Judged(input="i2", output="o2", approved=False, reason="bad")]
    learned = infer_standards("g", judged, GenEngine())
    assert learned["standards"] == ["Always cite the policy."]
    assert learned["do_not"] == ["Never promise refunds."]


def test_apply_learned_bootstraps_spec_and_records_examples():
    p = Project(name="p", goal="g")  # no spec
    judged = [Judged(input="i", output="good out", approved=True, reason="nice"),
              Judged(input="i2", output="bad out", approved=False, reason="wrong")]
    result = apply_learned(p, judged, {"standards": ["S1"], "do_not": ["D1"]})

    assert p.spec is not None
    assert "S1" in p.spec.standards and "D1" in p.spec.do_not
    assert result.standards_added == 1 and result.do_not_added == 1 and result.examples_recorded == 2
    # approved → good_output; rejected → bad_output
    assert p.spec.examples[0].good_output == "good out" and p.spec.examples[0].bad_output is None
    assert p.spec.examples[1].bad_output == "bad out" and p.spec.examples[1].good_output is None


def test_apply_learned_dedupes_existing():
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["S1"])
    result = apply_learned(p, [], {"standards": ["S1", "S2"], "do_not": []})
    assert result.standards_added == 1  # S1 already present, only S2 is new
    assert p.spec.standards == ["S1", "S2"]


class NonStrSubject:
    name = "subject@test"

    def complete(self, prompt, *, system=None, schema=None):
        return {"not": "a string"}


def test_propose_tolerates_non_string_subject_output():
    p = Project(name="p", goal="g")
    p.tests = [Case(id="t1", input="real")]
    cands = propose_candidates(p, GenEngine(), NonStrSubject(), n=1)
    assert isinstance(cands[0].output, str) and cands[0].output == ""  # coerced, no crash


def test_apply_learned_coerces_non_string_judged_fields():
    """Judged is an unvalidated dataclass; non-string input/output must be coerced,
    never raise a Pydantic ValidationError building the Example."""
    p = Project(name="p", goal="g")
    judged = [Judged(input=123, output=456, approved=True),
              Judged(input=None, output=["x"], approved=False, reason=7)]
    result = apply_learned(p, judged, {"standards": [], "do_not": []})  # must NOT raise
    assert result.examples_recorded == 2
    exs = p.spec.examples
    assert exs[0].input == "" and exs[0].good_output == ""   # 123 / 456 → ""
    assert exs[1].input == "" and exs[1].bad_output == ""     # None / list → ""
    assert exs[1].why is None                                 # non-string reason → None


def test_apply_learned_dedups_within_batch_and_across_lists():
    """The same sentence must never land in both standards and do_not,
    and within-batch duplicates collapse to one."""
    from ai_calibrator.models import BehaviorSpec
    from ai_calibrator.teach import apply_learned

    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(goal="g", do_not=["CONFLICT"])
    result = apply_learned(project, [], {
        "standards": ["CONFLICT", "NEW", "NEW"],       # cross-list + within-batch dupe
        "do_not": ["NEW", "NEVER", "NEVER"],           # 'NEW' already claimed by standards
    })
    assert project.spec.standards == ["NEW"]
    assert project.spec.do_not == ["CONFLICT", "NEVER"]
    assert result.standards == ["NEW"] and result.do_not == ["NEVER"]
