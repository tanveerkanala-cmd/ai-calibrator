"""M2 interview generation, verified with a mocked engine (no network)."""

from calibrator.interview import QUESTION_SCHEMA, generate_questions
from calibrator.models import Gap, Project, TaskType


class FakeEngine:
    name = "fake@test"

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, prompt, *, system=None, schema=None):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        return self.payload


def test_generate_questions_maps_fields():
    engine = FakeEngine(
        {
            "questions": [
                {"dimension": "tone", "question": "What voice?",
                 "draft_answer": "warm, concise", "rationale": "sets the style"},
                {"dimension": "refusal", "question": "Handle medical claims?",
                 "draft_answer": "decline + cite policy", "rationale": "compliance"},
            ]
        }
    )
    project = Project(name="t", goal="answer questions", task_type=TaskType.SUPPORT_ASSISTANT,
                     gaps=[Gap(dimension="tone"), Gap(dimension="refusal")])

    items = generate_questions(project, engine)

    assert [it.id for it in items] == ["q1", "q2"]
    assert items[0].dimension == "tone"
    assert items[0].draft_answer == "warm, concise"
    assert items[0].rationale == "sets the style"
    assert items[0].answer is None  # propose-and-ratify: not answered yet
    assert engine.calls[0]["schema"] is QUESTION_SCHEMA
    # the goal + gaps are passed to the engine
    assert "answer questions" in engine.calls[0]["prompt"]
    assert "refusal" in engine.calls[0]["prompt"]


def test_generate_skips_blank_questions():
    engine = FakeEngine({"questions": [
        {"dimension": "x", "question": "", "draft_answer": "", "rationale": ""},
    ]})
    project = Project(name="t", goal="g", gaps=[Gap(dimension="x")])
    assert generate_questions(project, engine) == []


def test_question_schema_is_strict_compatible():
    assert QUESTION_SCHEMA["additionalProperties"] is False
    item = QUESTION_SCHEMA["properties"]["questions"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
