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


def test_regenerate_preserves_ratified_answers():
    """Regeneration must never destroy the human's answers — the one artifact in a
    project that cannot be recomputed. A gap already answered is not re-asked."""
    from ai_calibrator.models import InterviewItem

    project = Project(name="p", goal="g", task_type=TaskType.ASSISTANT)
    project.gaps = [Gap(dimension="tone"), Gap(dimension="format")]
    project.interview = [
        InterviewItem(id="q1", dimension="tone", question="What tone?",
                      draft_answer="warm", answer="warm and direct"),
    ]
    engine = FakeEngine([{"question": "What format?", "draft_answer": "bullets"}])

    items = generate_questions(project, engine)

    by_dim = {it.dimension: it for it in items}
    assert by_dim["tone"].answer == "warm and direct", "a ratified answer was destroyed"
    assert by_dim["format"].answer is None
    assert len(engine.calls) == 1, "an already-answered gap must not be re-asked"
    assert [it.id for it in items] == ["q1", "q2"]


def test_regenerate_keeps_answers_whose_gap_disappeared():
    """Re-ingesting different materials changes the gap list; answers already given
    are still the user's work and must survive, appended after the gap-driven ones."""
    from ai_calibrator.models import InterviewItem

    project = Project(name="p", goal="g", task_type=TaskType.ASSISTANT)
    project.gaps = [Gap(dimension="format")]
    project.interview = [
        InterviewItem(id="q1", dimension="retired", question="Old question?", answer="a real answer"),
    ]
    engine = FakeEngine([{"question": "What format?"}])

    items = generate_questions(project, engine)

    assert [it.dimension for it in items] == ["format", "retired"]
    assert items[1].answer == "a real answer"


def test_unanswered_items_are_regenerated_not_kept():
    """Only ANSWERED items are protected: a drafted-but-unanswered question is
    regenerable output, so --regenerate must actually regenerate it."""
    from ai_calibrator.models import InterviewItem

    project = Project(name="p", goal="g", task_type=TaskType.ASSISTANT)
    project.gaps = [Gap(dimension="tone")]
    project.interview = [InterviewItem(id="q1", dimension="tone", question="Stale question?")]
    engine = FakeEngine([{"question": "Fresh question?"}])

    items = generate_questions(project, engine)

    assert items[0].question == "Fresh question?"




def test_answered_items_sharing_a_dimension_all_survive():
    """The carry-forward map was keyed on dimension alone, so the second answered
    item for a dimension was dropped and replaced by a freshly drafted, unanswered
    question — the exact loss the docstring promises never happens."""
    from ai_calibrator.models import InterviewItem

    project = Project(name="p", goal="g", task_type=TaskType.ASSISTANT)
    project.gaps = [Gap(dimension="tone"), Gap(dimension="tone")]
    project.interview = [
        InterviewItem(id="q1", dimension="tone", question="Tone with VIPs?", answer="warmer"),
        InterviewItem(id="q2", dimension="tone", question="Tone on refunds?", answer="firm"),
    ]
    engine = FakeEngine([])          # any engine call here means an answer was discarded

    items = generate_questions(project, engine)

    assert engine.calls == [], "an already-answered item was re-asked"
    assert [it.answer for it in items] == ["warmer", "firm"]
    assert [it.id for it in items] == ["q1", "q2"]


def test_surplus_answered_items_are_appended_not_dropped():
    """More answered items than gaps on that dimension: the extras are still the
    user's work, so they are appended like any other answer whose gap is gone."""
    from ai_calibrator.models import InterviewItem

    project = Project(name="p", goal="g", task_type=TaskType.ASSISTANT)
    project.gaps = [Gap(dimension="tone")]
    project.interview = [
        InterviewItem(id="q1", dimension="tone", question="Tone with VIPs?", answer="warmer"),
        InterviewItem(id="q2", dimension="tone", question="Tone on refunds?", answer="firm"),
    ]
    engine = FakeEngine([])

    items = generate_questions(project, engine)

    assert [it.answer for it in items] == ["warmer", "firm"]
    assert [it.id for it in items] == ["q1", "q2"]
