"""M3 compile logic, verified with a mocked engine (no network)."""

import json

import yaml

from calibrator.compile import (
    SPEC_SCHEMA,
    TESTS_SCHEMA,
    compile_project,
    generate_tests,
    rag_config,
    render_system_prompt,
    synthesize_spec,
)
from calibrator.models import BehaviorSpec, EvalCriterion, InterviewItem, Material, Project, TaskType, Weight

SPEC_PAYLOAD = {
    "persona": {"voice": "warm, concise", "reading_level": "8th grade"},
    "standards": ["Always confirm the account before sharing specifics."],
    "do_not": ["Never promise timelines we don't control."],
    "edge_cases": [
        {"situation": "customer asks for a medical claim", "ruling": "decline and cite policy"}
    ],
    "format": "<=120 words",
    "refusal_policy": "Decline medical/legal advice; redirect to a human.",
    "eval_criteria": [
        {"id": "cites_policy", "description": "Refund claims cite a policy line.", "weight": "high"}
    ],
    "examples": [
        {"input": "Can I return this?", "good_output": "Yes, within 30 days…",
         "bad_output": "Maybe", "why": "cites policy"}
    ],
}

TESTS_PAYLOAD = {
    "tests": [
        {"id": "t1", "input": "Does the serum cure eczema?", "expects": ["cites_policy"], "notes": "banned claim"},
        {"id": "t2", "input": "How do returns work?", "expects": ["cites_policy"], "notes": ""},
    ]
}


class SeqEngine:
    """Returns canned payloads in call order (spec, then tests)."""

    name = "fake@test"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.schemas = []

    def complete(self, prompt, *, system=None, schema=None):
        self.schemas.append(schema)
        return self.payloads.pop(0)


def _project():
    return Project(
        name="t",
        goal="answer product questions",
        task_type=TaskType.SUPPORT_ASSISTANT,
        materials=[Material(path="faq.md", kind="md")],
        facts=["We sell skincare."],
        interview=[InterviewItem(id="q1", dimension="tone", question="Voice?", answer="warm")],
    )


def test_synthesize_spec_maps_fields():
    engine = SeqEngine([SPEC_PAYLOAD])
    spec = synthesize_spec(_project(), engine)

    assert isinstance(spec, BehaviorSpec)
    assert spec.goal == "answer product questions"
    assert spec.persona.voice == "warm, concise"
    assert spec.standards and spec.do_not and spec.edge_cases
    assert spec.knowledge_sources == ["faq.md"]  # filled from materials, not the LLM
    assert spec.eval_criteria[0].id == "cites_policy"
    assert spec.eval_criteria[0].weight.value == "high"
    assert engine.schemas[0] is SPEC_SCHEMA


def test_generate_tests_maps_fields():
    engine = SeqEngine([TESTS_PAYLOAD])
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="cites_policy", description="cites a policy line", weight=Weight.HIGH),
    ])
    tests = generate_tests(spec, engine)

    assert [t.id for t in tests] == ["t1", "t2"]
    assert tests[0].expects == ["cites_policy"]
    assert engine.schemas[0] is TESTS_SCHEMA


def test_generate_tests_drops_unknown_expects():
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="x", weight=Weight.HIGH),
    ])
    engine = SeqEngine([{
        "tests": [
            {"id": "t1", "input": "q1", "expects": ["c1", "bogus"], "notes": ""},
            {"id": "t2", "input": "q2", "expects": ["nope"], "notes": ""},
        ]
    }])
    tests = generate_tests(spec, engine)
    assert tests[0].expects == ["c1"]   # invented id dropped
    assert tests[1].expects == []        # all-invalid → empty → falls back to all criteria in eval


def test_render_system_prompt_includes_key_sections():
    spec = synthesize_spec(_project(), SeqEngine([SPEC_PAYLOAD]))
    sp = render_system_prompt(spec)
    assert "answer product questions" in sp
    assert "warm, concise" in sp
    assert "Never promise timelines" in sp
    assert "decline and cite policy" in sp


def test_compile_project_writes_bundle(tmp_path):
    engine = SeqEngine([SPEC_PAYLOAD, TESTS_PAYLOAD])
    project = _project()
    result = compile_project(project, engine, project_dir=tmp_path)

    assert project.spec is not None
    assert len(project.tests) == 2
    assert result.criteria == 1 and result.tests == 2

    build = tmp_path / "build"
    for name in ["spec.yaml", "system_prompt.txt", "rubric.yaml", "rag.config.yaml", "tests.jsonl"]:
        assert (build / name).exists(), name

    rub = yaml.safe_load((build / "rubric.yaml").read_text())
    assert rub["criteria"][0]["id"] == "cites_policy"

    lines = [ln for ln in (build / "tests.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "t1"


def test_compile_preserves_taught_standards(tmp_path):
    """Standards added via teach must survive a later compile (was: silent data loss)."""
    from calibrator.compile import compile_project
    from calibrator.teach import Judged, apply_learned

    project = _project()  # has an interview answer + facts
    apply_learned(project, [Judged(input="q", output="good", approved=True, reason="r")],
                  {"standards": ["TAUGHT: always cite the policy"], "do_not": ["TAUGHT: never guess"]})
    assert "TAUGHT: always cite the policy" in project.spec.standards

    compile_project(project, SeqEngine([SPEC_PAYLOAD, TESTS_PAYLOAD]), project_dir=tmp_path)
    # taught rules carried forward AND the synthesized standard is present
    assert "TAUGHT: always cite the policy" in project.spec.standards
    assert "TAUGHT: never guess" in project.spec.do_not
    assert any("confirm the account" in s for s in project.spec.standards)  # from SPEC_PAYLOAD


def test_compile_preserves_spec_when_no_interview_answers(tmp_path):
    """With no interview answers (e.g. a teach-bootstrapped project), compile must
    preserve the existing spec, not overwrite it with one synthesized from nothing."""
    from calibrator.compile import compile_project

    project = Project(name="t", goal="g")  # no interview answers
    project.spec = BehaviorSpec(goal="g", standards=["BOOTSTRAPPED"],
                                eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    # Only generate_tests should call the engine — synthesize must be skipped.
    compile_project(project, SeqEngine([TESTS_PAYLOAD]), project_dir=tmp_path)
    assert project.spec.standards == ["BOOTSTRAPPED"]  # preserved, not overwritten


def test_rag_config_shape():
    cfg = rag_config(BehaviorSpec(goal="g", knowledge_sources=["a.md"]))
    assert cfg["knowledge_sources"] == ["a.md"]
    assert cfg["top_k"] == 5 and cfg["table"]
