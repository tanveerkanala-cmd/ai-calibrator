"""Multi-stakeholder calibration — gather, conflict detection, merge."""

from typer.testing import CliRunner

from ai_calibrator.cli import app
from ai_calibrator.models import BehaviorSpec, EdgeCase, EvalCriterion, Example, TaskType, Weight
from ai_calibrator.stakeholders import (
    build_merged_spec,
    detect_conflicts,
    gather,
    merged_project,
)


def _spec(standards=None, do_not=None, **kw):
    return BehaviorSpec(goal="g", standards=standards or [], do_not=do_not or [], **kw)


def test_gather_tags_and_indexes():
    named = {"legal": _spec(standards=["disclaim"], do_not=["no slang"]),
             "sales": _spec(standards=["be punchy"])}
    stmts = gather(named)
    assert [s.idx for s in stmts] == [1, 2, 3]
    assert (stmts[0].stakeholder, stmts[0].kind, stmts[0].text) == ("legal", "standard", "disclaim")
    assert stmts[1].kind == "do_not"
    assert stmts[2].stakeholder == "sales"


class ConflictEngine:
    name = "ce@test"

    def __init__(self, a=1, b=2):
        self.a, self.b = a, b

    def complete(self, prompt, *, system=None, schema=None):
        return {"conflicts": [{"a": self.a, "b": self.b, "explanation": "they contradict", "severity": "high"}]}


def test_detect_conflicts_maps_indices_to_statements():
    named = {"legal": _spec(standards=["always disclaim"]), "sales": _spec(standards=["never disclaim"])}
    conflicts = detect_conflicts(gather(named), ConflictEngine(1, 2))
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.a.stakeholder == "legal" and c.b.stakeholder == "sales" and c.severity == "high"


def test_detect_conflicts_filters_bad_and_duplicate_pairs():
    named = {"a": _spec(standards=["x"]), "b": _spec(standards=["y"])}

    class E:
        name = "e"

        def complete(self, prompt, *, system=None, schema=None):
            return {"conflicts": [
                {"a": 1, "b": 99, "explanation": "", "severity": "low"},   # out of range
                {"a": 1, "b": 1, "explanation": "", "severity": "low"},    # self-pair
                {"a": 1, "b": 2, "explanation": "ok", "severity": "bogus"},  # valid (severity coerced)
                {"a": 2, "b": 1, "explanation": "dup", "severity": "low"},  # duplicate pair
            ]}

    conflicts = detect_conflicts(gather(named), E())
    assert len(conflicts) == 1 and conflicts[0].severity == "medium"  # bogus severity → medium


def test_detect_conflicts_needs_two_statements():
    assert detect_conflicts(gather({"a": _spec(standards=["only one"])}), ConflictEngine()) == []


def test_build_merged_keep_a_drops_b():
    named = {"legal": _spec(standards=["always disclaim"]), "sales": _spec(standards=["never disclaim"])}
    spec = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT, drops={2})
    assert "always disclaim" in spec.standards and "never disclaim" not in spec.standards


def test_build_merged_merge_drops_both_and_adds():
    named = {"legal": _spec(standards=["always disclaim"]), "sales": _spec(standards=["never disclaim"])}
    spec = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT,
                             drops={1, 2}, additions=["disclaim only for medical topics"])
    assert spec.standards == ["disclaim only for medical topics"]


def test_build_merged_unions_other_dimensions_with_dedup():
    named = {
        "a": _spec(standards=["s1"],
                   eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)],
                   edge_cases=[EdgeCase(situation="x", ruling="y")], examples=[Example(input="i1")]),
        "b": _spec(standards=["s2"],
                   eval_criteria=[EvalCriterion(id="c1", description="dup-id", weight=Weight.LOW),
                                  EvalCriterion(id="c2", description="d2", weight=Weight.LOW)],
                   examples=[Example(input="i1"), Example(input="i2")]),
    }
    spec = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT)
    assert set(spec.standards) == {"s1", "s2"}
    # A shared id with a DIFFERENT meaning is namespaced, never dropped: criterion
    # ids are engine-generated labels, so two specs colliding on `c1` is routine
    # and silently discarding one stakeholder's criterion loses their check too.
    by_id = {c.id: c for c in spec.eval_criteria}
    assert set(by_id) == {"c1", "c2", "b_c1"}
    assert by_id["c1"].description == "d"
    assert by_id["b_c1"].description == "dup-id"
    assert {e.input for e in spec.examples} == {"i1", "i2"}        # dedup by input
    assert len(spec.edge_cases) == 1


def test_merged_project():
    named = {"a": _spec(standards=["s1"]), "b": _spec(standards=["s2"])}
    p = merged_project("org", named, goal="org goal", task_type=TaskType.SUPPORT_ASSISTANT)
    assert p.name == "org" and p.goal == "org goal" and p.spec is not None
    assert set(p.spec.standards) == {"s1", "s2"}


def test_cli_merge_requires_two_sources(tmp_path):
    r = CliRunner().invoke(app, ["merge", str(tmp_path / "out"), "--from", str(tmp_path / "a")])
    assert r.exit_code == 1 and "at least two" in r.output
