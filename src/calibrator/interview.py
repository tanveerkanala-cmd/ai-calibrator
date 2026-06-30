"""M2 — Interview: turn the gap list into adaptive, propose-and-ratify questions.

For each gap the interviewer engine writes one question, a DRAFT answer (the
tool's best guess the user accepts or corrects), and a short rationale (why it
matters — teach-while-scaffolding). The user ratifies; their answers are the
signal that later compiles into the behavior spec (M3).
"""

from __future__ import annotations

from .coerce import as_list, as_opt_str, as_str, is_str
from .engines.base import Engine, require_object
from .models import InterviewItem, Project

# Strict-compatible schema (works across Anthropic / OpenAI / Ollama).
QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dimension": {"type": "string"},
                    "question": {"type": "string"},
                    "draft_answer": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["dimension", "question", "draft_answer", "rationale"],
            },
        }
    },
    "required": ["questions"],
}

_INTERVIEW_SYSTEM = (
    "You are interviewing a domain expert to capture the judgment needed to "
    "build their AI. For each gap, write exactly ONE clear, specific question; "
    "a DRAFT answer (your best guess they can accept or correct — "
    "propose-and-ratify); and a short rationale explaining why it matters, so "
    "they learn the moving parts. Keep questions high-signal and non-redundant; "
    "don't ask about anything the known facts already settle. Respond with JSON "
    "only, matching the provided schema."
)


def _gaps_block(project: Project) -> str:
    lines = []
    for g in project.gaps:
        why = f" — {g.why_it_matters}" if g.why_it_matters else ""
        lines.append(f"- {g.dimension}{why}")
    return "\n".join(lines)


def generate_questions(project: Project, engine: Engine) -> list[InterviewItem]:
    """Run the interviewer engine over the gaps → a list of (unanswered) items."""
    facts = "\n".join(f"- {f}" for f in project.facts) or "(none captured)"
    prompt = (
        f"GOAL: {project.goal}\n"
        f"TASK TYPE: {project.task_type.value}\n\n"
        f"FACTS ALREADY KNOWN (do not re-ask these):\n{facts}\n\n"
        f"GAPS TO RESOLVE:\n{_gaps_block(project)}\n\n"
        "Write one question per gap (merge closely-related gaps where natural)."
    )
    result = require_object(engine.complete(prompt, system=_INTERVIEW_SYSTEM, schema=QUESTION_SCHEMA), "interviewer")

    items: list[InterviewItem] = []
    for i, q in enumerate(as_list(result.get("questions")), start=1):
        if not isinstance(q, dict) or not is_str(q.get("question")):
            continue
        items.append(
            InterviewItem(
                id=f"q{i}",
                dimension=as_str(q.get("dimension")),
                question=q["question"],
                draft_answer=as_opt_str(q.get("draft_answer")),
                rationale=as_opt_str(q.get("rationale")),
            )
        )
    return items
