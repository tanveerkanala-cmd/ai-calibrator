"""M2 — Interview: turn the gap list into adaptive, propose-and-ratify questions.

For each gap the interviewer engine writes one question, a DRAFT answer (the
tool's best guess the user accepts or corrects), and a short rationale (why it
matters — teach-while-scaffolding). The user ratifies; their answers are the
signal that later compiles into the behavior spec (M3).
"""

from __future__ import annotations

from typing import Callable, Optional

from .coerce import as_opt_str, is_str
from .engines.base import Engine, require_object
from .models import Gap, InterviewItem, Project

# One question per gap: a strict-compatible single-object schema (works across
# Anthropic / OpenAI / Ollama). Generating per gap — rather than one call that
# merges every gap into a short list — keeps coverage one-to-one (no gap is
# silently dropped) and means a timeout costs a single gap, not the whole run.
QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string"},
        "draft_answer": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["question", "draft_answer", "rationale"],
}

_INTERVIEW_SYSTEM = (
    "You are interviewing a domain expert to capture the judgment needed to "
    "build their AI. For the ONE gap given, write a single clear, specific "
    "question; a DRAFT answer (your best guess they can accept or correct — "
    "propose-and-ratify); and a short rationale explaining why it matters, so "
    "they learn the moving parts. Keep it high-signal; don't ask about anything "
    "the known facts already settle. Respond with JSON only, matching the schema."
)

# Called after each gap's question is drafted: (items_so_far, done, total). Lets
# the caller persist incrementally so an engine timeout mid-interview keeps the
# questions already drafted instead of discarding the whole run. Every payload
# carries every ratified answer, so writing one over the stored interview is safe.
ProgressFn = Callable[[list[InterviewItem], int, int], None]


def _one_question(project: Project, gap: Gap, engine: Engine) -> Optional[InterviewItem]:
    facts = "\n".join(f"- {f}" for f in project.facts) or "(none captured)"
    why = f" — {gap.why_it_matters}" if gap.why_it_matters else ""
    prompt = (
        f"GOAL: {project.goal}\n"
        f"TASK TYPE: {project.task_type.value}\n\n"
        f"FACTS ALREADY KNOWN (do not re-ask these):\n{facts}\n\n"
        f"THE GAP TO RESOLVE:\n- {gap.dimension}{why}\n\n"
        "Write one question that resolves this gap."
    )
    q = require_object(engine.complete(prompt, system=_INTERVIEW_SYSTEM, schema=QUESTION_SCHEMA), "interviewer")
    if not is_str(q.get("question")):
        return None
    # dimension comes from the GAP, not the model — so every produced item maps
    # back to exactly one gap and coverage can be checked precisely.
    return InterviewItem(
        id="",  # assigned by the caller (contiguous q1..qN over produced items)
        dimension=gap.dimension,
        question=q["question"],
        draft_answer=as_opt_str(q.get("draft_answer")),
        rationale=as_opt_str(q.get("rationale")),
    )


def _with_carried_answers(
    items: list[InterviewItem], answered: dict[str, list[InterviewItem]]
) -> list[InterviewItem]:
    """``items`` plus every answer not folded into it yet, renumbered contiguously.

    Answered items are folded in lazily, as each one's own gap comes up, so a
    partial list is a strict prefix of the interview — and a caller that persists
    it (the CLI and the API both do, after every gap) would write that prefix over
    the stored interview, deleting the answers whose gaps hadn't been reached. A
    payload is only safe to persist if it holds every answer.
    """
    out = list(items)
    for leftover in [it for bucket in answered.values() for it in bucket]:
        clone = leftover.model_copy(deep=True)
        clone.id = f"q{len(out) + 1}"
        out.append(clone)
    return out


def generate_questions(
    project: Project, engine: Engine, *, on_progress: Optional[ProgressFn] = None
) -> list[InterviewItem]:
    """Draft one interview question per gap, in order.

    Each gap is a separate engine call, so partial progress survives a mid-run
    failure: ``on_progress(items_so_far, done, total)`` fires after each gap for
    the caller to persist. A gap whose draft comes back malformed is skipped (it
    shows up as uncovered via :func:`uncovered_gaps`, never silently merged away).

    **Ratified answers are never destroyed.** A gap whose dimension already has an
    answered item is carried through untouched instead of re-asked — cheaper, and
    it protects the one artifact in a project that cannot be recomputed. Answered
    items whose dimension is no longer among the gaps (the materials changed since)
    are kept too, appended after the gap-driven ones: a stale question is a small
    cost, a discarded answer is the user's work. Answered items that share a
    dimension are matched to that dimension's gaps in order, and any surplus is
    appended the same way — every answer survives, exactly once.
    """
    # Several items can share a dimension (a gap list can name one twice, and a
    # hand-edited project.yaml can too), so bucket them: keying on the dimension
    # alone kept only the first and quietly re-asked the rest.
    answered: dict[str, list[InterviewItem]] = {}
    for it in project.interview:
        if it.answer:
            answered.setdefault(it.dimension, []).append(it)

    items: list[InterviewItem] = []
    total = len(project.gaps)
    for i, gap in enumerate(project.gaps, start=1):
        bucket = answered.get(gap.dimension)
        kept = bucket.pop(0) if bucket else None
        item = kept.model_copy(deep=True) if kept is not None else _one_question(project, gap, engine)
        if item is not None:
            item.id = f"q{len(items) + 1}"
            items.append(item)
        if on_progress is not None:
            on_progress(_with_carried_answers(items, answered), i, total)

    return _with_carried_answers(items, answered)  # the tail: answers whose gap is gone


def uncovered_gaps(project: Project, items: list[InterviewItem]) -> list[str]:
    """Gap dimensions with no drafted question — the coverage a caller must warn
    about instead of reporting a merged-down count as success."""
    covered = {it.dimension for it in items}
    return [g.dimension for g in project.gaps if g.dimension not in covered]
