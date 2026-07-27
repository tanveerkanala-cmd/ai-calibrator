"""Reverse-calibrate — extract a tested spec from an existing system prompt."""

from ai_calibrator.models import BehaviorSpec, TaskType
from ai_calibrator.reverse import reverse_project, reverse_spec
from ai_calibrator.store import load_project

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
    assert (tmp_path / "imported_prompt.txt").read_text(encoding="utf-8") == PROMPT
    # round-trips: the imported project loads back coherently
    assert load_project(tmp_path).spec.standards == proj.spec.standards


def test_reverse_project_uses_default_binding_without_engine_spec(tmp_path):
    proj = reverse_project("imp2", "g", PROMPT, SpecEngine(), project_dir=tmp_path)
    assert proj.engines.subject == "claude-sonnet-4-6@anthropic"  # default cloud binding




def test_imported_project_gets_the_gitignore_init_writes(tmp_path):
    """An imported project is the one most likely to be committed, so logs/,
    evals/ and any .env must be ignored from the first write, as `init` does."""
    reverse_project("imp3", "g", PROMPT, SpecEngine(), project_dir=tmp_path)
    body = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "logs/" in body and "evals/" in body and ".env" in body


def test_import_never_clobbers_an_existing_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("mine\n", encoding="utf-8")
    reverse_project("imp4", "g", PROMPT, SpecEngine(), project_dir=tmp_path)
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == "mine\n"
