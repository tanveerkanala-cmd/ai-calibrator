"""M2 interview generation, verified with a mocked engine (no network)."""

from ai_calibrator.interview import QUESTION_SCHEMA, generate_questions, uncovered_gaps
from ai_calibrator.models import Gap, Project, TaskType


class FakeEngine:
    """Returns one queued payload per call (one call per gap)."""

    name = "fake@test"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def complete(self, prompt, *, system=None, schema=None):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        return self.payloads.pop(0) if self.payloads else {}


def test_generate_questions_one_per_gap():
    engine = FakeEngine([
        {"question": "What voice?", "draft_answer": "warm, concise", "rationale": "sets the style"},
        {"question": "Handle medical claims?", "draft_answer": "decline + cite policy", "rationale": "compliance"},
    ])
    project = Project(name="t", goal="answer questions", task_type=TaskType.SUPPORT_ASSISTANT,
                      gaps=[Gap(dimension="tone"), Gap(dimension="refusal")])

    items = generate_questions(project, engine)

    assert [it.id for it in items] == ["q1", "q2"]
    # one engine call per gap — no merge-everything-into-one call
    assert len(engine.calls) == 2
    # dimension comes from the gap, guaranteeing 1:1 coverage mapping
    assert items[0].dimension == "tone"
    assert items[0].draft_answer == "warm, concise"
    assert items[0].rationale == "sets the style"
    assert items[0].answer is None  # propose-and-ratify: not answered yet
    assert engine.calls[0]["schema"] is QUESTION_SCHEMA
    # the goal + the specific gap are passed to the engine
    assert "answer questions" in engine.calls[0]["prompt"]
    assert "refusal" in engine.calls[1]["prompt"]
    assert uncovered_gaps(project, items) == []


def test_generate_reports_uncovered_gap_instead_of_dropping_it():
    # second gap comes back malformed (no question) → skipped, and surfaced as
    # uncovered rather than silently merged away
    engine = FakeEngine([
        {"question": "What voice?", "draft_answer": "warm", "rationale": "style"},
        {"draft_answer": "x", "rationale": "y"},  # no "question"
    ])
    project = Project(name="t", goal="g", gaps=[Gap(dimension="tone"), Gap(dimension="refusal")])
    items = generate_questions(project, engine)
    assert [it.id for it in items] == ["q1"]
    assert uncovered_gaps(project, items) == ["refusal"]


def test_generate_skips_blank_questions():
    engine = FakeEngine([{"question": "", "draft_answer": "", "rationale": ""}])
    project = Project(name="t", goal="g", gaps=[Gap(dimension="x")])
    assert generate_questions(project, engine) == []


def test_on_progress_fires_per_gap():
    engine = FakeEngine([
        {"question": "q1?", "draft_answer": "a", "rationale": "r"},
        {"question": "q2?", "draft_answer": "a", "rationale": "r"},
    ])
    project = Project(name="t", goal="g", gaps=[Gap(dimension="a"), Gap(dimension="b")])
    seen = []
    generate_questions(project, engine, on_progress=lambda items, done, total: seen.append((done, total, len(items))))
    assert seen == [(1, 2, 1), (2, 2, 2)]


def test_question_schema_is_strict_compatible():
    assert QUESTION_SCHEMA["additionalProperties"] is False
    assert set(QUESTION_SCHEMA["required"]) == set(QUESTION_SCHEMA["properties"])
