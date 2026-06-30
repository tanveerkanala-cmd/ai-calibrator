"""Input/output validation — regression tests for the stress-found crashes.

Covers the defensive guards added after stress testing: chunk size, refine-loop
controls (max_rounds / threshold incl. NaN), and the "engine returned a non-JSON
object" path that previously surfaced as a cryptic AttributeError.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrator.engines.base import call_json, require_object
from calibrator.ingest import extract_gaps
from calibrator.models import BehaviorSpec, EvalCriterion, Project, TaskType, Weight
from calibrator.models import TestCase as CaseModel
from calibrator.parsing import chunk_text
from calibrator.pipeline import calibrate_loop


# --- chunk_text size ---------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_project_rejects_empty_name(bad):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Project(name=bad, goal="g")


def test_project_accepts_normal_name():
    assert Project(name="my-project", goal="g").name == "my-project"


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_chunk_text_rejects_nonpositive_size(bad):
    with pytest.raises(ValueError):
        chunk_text("some text here", size=bad)


def test_chunk_text_minimum_valid_size_works():
    assert chunk_text("a\n\nb", size=1) == ["a", "b"]


# --- calibrate_loop controls -------------------------------------------------

class _StubEngine:
    name = "stub@test"

    def complete(self, prompt, *, system=None, schema=None):  # never reached
        raise AssertionError("validation should reject before any engine call")


def _project_with_spec() -> Project:
    p = Project(name="p", goal="g", task_type=TaskType.ASSISTANT)
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="hi", expects=["c1"])]
    return p


@pytest.mark.parametrize("bad_rounds", [0, -1, -5, 1000])
def test_calibrate_loop_rejects_bad_rounds(bad_rounds):
    e = _StubEngine()
    with pytest.raises(ValueError):
        calibrate_loop(_project_with_spec(), e, e, e, max_rounds=bad_rounds)


@pytest.mark.parametrize("bad_threshold", [float("nan"), float("inf"), float("-inf"), -0.1, 1.5, 2.0])
def test_calibrate_loop_rejects_bad_threshold(bad_threshold):
    e = _StubEngine()
    with pytest.raises(ValueError):
        calibrate_loop(_project_with_spec(), e, e, e, threshold=bad_threshold, max_rounds=1)


# --- engine returned a non-object -------------------------------------------

def test_require_object_accepts_dict():
    assert require_object({"a": 1}, "x") == {"a": 1}


@pytest.mark.parametrize("bad", ["a string", 123, 4.5, [1, 2, 3], None, True])
def test_require_object_rejects_non_dict(bad):
    with pytest.raises(RuntimeError):
        require_object(bad, "engine")


@pytest.mark.parametrize("payload", ['"just a string"', "[1, 2, 3]", "42", "true", "null"])
def test_call_json_rejects_valid_json_that_is_not_an_object(payload):
    # Valid JSON, wrong shape → treated as a parse failure → RuntimeError after retry.
    with pytest.raises(RuntimeError):
        call_json(lambda: payload)


def test_call_json_accepts_object():
    assert call_json(lambda: '{"x": 1}') == {"x": 1}


def test_synthesize_spec_tolerates_non_string_fields():
    """Engine emits truthy NON-string values where strings are expected — the
    compiler must coerce/drop them, never raise a Pydantic ValidationError.
    (Regression: the re-stress found persona.voice=123 crashing compile.)"""
    from calibrator.compile import synthesize_spec
    from calibrator.models import InterviewItem

    class Bad:
        name = "bad@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {
                "persona": {"voice": 123, "reading_level": ["nope"]},
                "standards": ["ok", 7, None],
                "do_not": [{"x": 1}],
                "edge_cases": [{"situation": 5, "ruling": "r"}, {"situation": "s", "ruling": "r"}],
                "format": 99,
                "refusal_policy": None,
                "eval_criteria": [
                    {"id": 1, "description": "d", "weight": "high"},
                    {"id": "c1", "description": 2, "weight": "bogus"},
                ],
                "examples": [{"input": 0}, {"input": "real", "good_output": 1}],
            }

    proj = Project(name="p", goal="g",
                   interview=[InterviewItem(id="q1", dimension="d", question="?", answer="a")])
    spec = synthesize_spec(proj, Bad())  # must NOT raise
    assert spec.persona.voice is None and spec.persona.reading_level is None
    assert spec.standards == ["ok"]                       # non-string junk dropped
    assert spec.format is None                            # non-string optional -> None
    assert [e.situation for e in spec.edge_cases] == ["s"]   # non-string situation dropped
    assert [c.id for c in spec.eval_criteria] == ["c1"]      # non-string id dropped
    assert spec.eval_criteria[0].weight.value == "medium"    # invalid weight defaulted
    assert [ex.input for ex in spec.examples] == ["real"]    # non-string input dropped
    assert spec.examples[0].good_output is None              # non-string optional -> None


def test_generate_questions_tolerates_non_string_fields():
    from calibrator.interview import generate_questions
    from calibrator.models import Gap

    class Bad:
        name = "bad@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"questions": [
                {"question": 123, "dimension": "d"},
                {"question": "real?", "dimension": 5, "draft_answer": 9, "rationale": None},
            ]}

    proj = Project(name="p", goal="g", gaps=[Gap(dimension="d")])
    items = generate_questions(proj, Bad())  # must NOT raise
    assert [it.question for it in items] == ["real?"]   # non-string question dropped
    assert items[0].dimension == ""                     # non-string dimension -> ""
    assert items[0].draft_answer is None


def test_extract_gaps_tolerates_non_string_fields():
    from calibrator.ingest import extract_gaps

    class Bad:
        name = "bad@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"facts": ["f", 1], "gaps": [
                {"dimension": 7, "why_it_matters": "x"},
                {"dimension": "tone", "why_it_matters": 9},
            ]}

    facts, gaps = extract_gaps("g", "assistant", [(Path("x.md"), "text")], Bad())  # must NOT raise
    assert [g.dimension for g in gaps] == ["tone"]   # non-string dimension dropped
    assert gaps[0].why_it_matters is None            # non-string optional -> None


def test_stage_raises_clear_error_when_engine_ignores_schema():
    """A pipeline stage given a non-object engine response raises a clear
    RuntimeError, not an AttributeError from calling .get() on a str."""

    class IgnoresSchema:
        name = "bad@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "I am not JSON"  # bypasses call_json entirely

    with pytest.raises(RuntimeError):
        extract_gaps("goal", "assistant", [(Path("x.md"), "some text")], IgnoresSchema())


# --- API EvalBody boundary validation (needs the `api` extra) ---------------

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from calibrator.api import create_app  # noqa: E402


@pytest.mark.parametrize(
    "body",
    [
        {"refine": True, "rounds": 0},
        {"rounds": -1},
        {"rounds": 99999},
        {"threshold": 2.0},
        {"threshold": -1.0},
    ],
)
def test_eval_endpoint_rejects_bad_controls_with_422(tmp_path, body):
    client = TestClient(create_app(tmp_path))
    client.post("/api/projects", json={"name": "p", "goal": "g"})
    # Body validation happens before the handler runs → clean 422, not a 500.
    assert client.post("/api/projects/p/eval", json=body).status_code == 422
