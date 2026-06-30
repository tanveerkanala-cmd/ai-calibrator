"""Reverse-calibrate — extract a tested spec from an existing system prompt."""

from calibrator.models import BehaviorSpec, TaskType
from calibrator.reverse import reverse_project, reverse_spec
from calibrator.store import load_project

PROMPT = ("You are a concise research assistant. Always cite sources. Never speculate. "
          "If a query is ambiguous, ask to clarify.")


class SpecEngine:
    """Returns a SPEC payload for the reverse pass, a TESTS payload for generate_tests."""
    name = "rev@test"

    def __init__(self):
        self.calls = []

    def complete(self, prompt, *, system=None, schema=None):
        self.calls.append((system, schema))
        props = (schema or {}).get("properties", {})
        if "tests" in props:
            return {"tests": [{"id": "t1", "input": "an ambiguous query", "expects": ["clarity"], "notes": ""}]}
        return {
            "persona": {"voice": "concise", "reading_level": "plain"},
            "standards": ["Be clear.", "Cite sources."],
            "do_not": ["Never speculate."],
            "edge_cases": [{"situation": "ambiguous query", "ruling": "ask to clarify"}],
            "format": "short paragraphs", "refusal_policy": "decline harmful requests",
            "eval_criteria": [{"id": "clarity", "description": "answer is clear", "weight": "high"}],
            "examples": [],
        }


def test_reverse_spec_extracts_from_prompt():
    eng = SpecEngine()
    spec = reverse_spec(PROMPT, "answer research questions", TaskType.ASSISTANT, eng)
    assert isinstance(spec, BehaviorSpec)
    assert "Be clear." in spec.standards and "Never speculate." in spec.do_not
    assert spec.eval_criteria[0].id == "clarity" and spec.persona.voice == "concise"
    # it used the reverse-engineering system prompt, not the compile one
    assert "reverse-engineer" in (eng.calls[0][0] or "").lower()


def test_reverse_project_creates_evaluable_project(tmp_path):
    proj = reverse_project("imported", "answer research questions", PROMPT, SpecEngine(),
                           task_type=TaskType.ASSISTANT, engine_spec="gemma4:e4b@ollama", project_dir=tmp_path)
    assert proj.name == "imported" and proj.spec is not None
    assert proj.tests and proj.tests[0].id == "t1"
    # the created project runs on the specified engine
    assert proj.engines.subject == "gemma4:e4b@ollama" and proj.engines.judge == "gemma4:e4b@ollama"
    # persisted with build bundle + the original prompt for provenance
    assert (tmp_path / "build" / "system_prompt.txt").exists()
    assert (tmp_path / "imported_prompt.txt").read_text() == PROMPT
    # round-trips: the imported project loads back coherently
    assert load_project(tmp_path).spec.standards == proj.spec.standards


def test_reverse_project_uses_default_binding_without_engine_spec(tmp_path):
    proj = reverse_project("imp2", "g", PROMPT, SpecEngine(), project_dir=tmp_path)
    assert proj.engines.subject == "claude-sonnet-4-6@anthropic"  # default cloud binding
