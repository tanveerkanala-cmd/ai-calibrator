"""Calibrate-by-example — infer the spec from judgments, not articulation.

The deepest friction in calibration: domain experts often can't *write* their
standards, but they instantly *recognize* good vs. bad output. This mode plays to
that. It shows candidate outputs on real inputs; the human approves or rejects
each (with an optional one-line reason); then a compiler engine **reverse-engineers
the implicit standards** — what good answers do, what rejected ones got wrong —
and folds them into the spec, recording each judgment as a golden example.

Teach by reacting, not by writing. It complements the interview (it can refine an
existing spec) and can also bootstrap one from scratch (judge the raw model's
outputs first, build the spec from your verdicts).
"""

from __future__ import annotations

from dataclasses import dataclass

from .coerce import as_list, as_opt_str, as_str, is_str
from .compile import render_system_prompt
from .engines.base import Engine, require_object
from .models import BehaviorSpec, Example, Project

INPUTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"inputs": {"type": "array", "items": {"type": "string"}}},
    "required": ["inputs"],
}

INFER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "standards": {"type": "array", "items": {"type": "string"}},
        "do_not": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["standards", "do_not"],
}

_INPUTS_SYSTEM = (
    "Generate realistic, varied user inputs to probe an AI built for the stated "
    "goal — a mix of everyday cases and tricky edge cases. Respond with JSON "
    "only, matching the schema."
)

_INFER_SYSTEM = (
    "You reverse-engineer an AI's behavior standards from a human's approve/reject "
    "judgments of its outputs. Infer the GENERAL, reusable standards the human is "
    "applying (what approved outputs do right) and the do_not rules (what rejected "
    "outputs did wrong). State each as a concrete, generalizable instruction — not "
    "a comment about one specific answer. Omit anything the judgments don't support. "
    "Respond with JSON only, matching the schema."
)


@dataclass
class Candidate:
    id: str
    input: str
    output: str


@dataclass
class Judged:
    input: str
    output: str
    approved: bool
    reason: str | None = None


@dataclass
class LearnResult:
    standards: list[str]
    do_not: list[str]
    examples_recorded: int
    standards_added: int
    do_not_added: int


def _generate_inputs(project: Project, engine: Engine, n: int) -> list[str]:
    prompt = (
        f"GOAL: {project.goal}\nTASK TYPE: {project.task_type.value}\n\n"
        f"Generate {n} user inputs to test this AI."
    )
    out = require_object(engine.complete(prompt, system=_INPUTS_SYSTEM, schema=INPUTS_SCHEMA), "input generator")
    return [s for s in as_list(out.get("inputs")) if is_str(s)][:n]


def propose_candidates(
    project: Project,
    generator: Engine,
    subject: Engine,
    *,
    n: int = 5,
) -> list[Candidate]:
    """Produce ``n`` (input, output) candidates for the human to judge.

    Reuses the project's existing test inputs when present (real scenarios);
    generates fresh inputs from the goal to top up. Outputs come from the subject
    running under the current spec's system prompt (or raw, if no spec yet)."""
    system = render_system_prompt(project.spec) if project.spec else None
    inputs = [t.input for t in project.tests if is_str(t.input)][:n]
    if len(inputs) < n:
        inputs = inputs + _generate_inputs(project, generator, n - len(inputs))
    candidates: list[Candidate] = []
    for i, inp in enumerate(inputs[:n], start=1):
        output = as_str(subject.complete(inp, system=system))  # tolerate non-string output
        candidates.append(Candidate(id=f"ex{i}", input=inp, output=output))
    return candidates


def infer_standards(goal: str, judged: list[Judged], engine: Engine) -> dict:
    """Compiler engine: human judgments → inferred {standards, do_not}."""
    blocks = []
    for j in judged:
        verdict = "APPROVED" if j.approved else "REJECTED"
        reason = f"\nREASON: {j.reason}" if j.reason else ""
        blocks.append(f"INPUT: {j.input}\nOUTPUT: {j.output}\nVERDICT: {verdict}{reason}")
    prompt = (
        f"GOAL: {goal}\n\n"
        "Here are human judgments of candidate AI outputs. Infer the standards.\n\n"
        + "\n\n".join(blocks)
    )
    out = require_object(engine.complete(prompt, system=_INFER_SYSTEM, schema=INFER_SCHEMA), "compiler")
    return {
        "standards": [s for s in as_list(out.get("standards")) if is_str(s)],
        "do_not": [s for s in as_list(out.get("do_not")) if is_str(s)],
    }


def apply_learned(project: Project, judged: list[Judged], learned: dict) -> LearnResult:
    """Fold inferred standards into the spec and record judgments as examples.

    Creates a minimal spec if the project has none (judgment-first bootstrap).
    De-duplicates against existing standards/never-rules."""
    if project.spec is None:
        project.spec = BehaviorSpec(goal=project.goal, task_type=project.task_type)
    spec = project.spec

    new_standards = [s for s in as_list(learned.get("standards")) if is_str(s) and s not in spec.standards]
    new_do_not = [s for s in as_list(learned.get("do_not")) if is_str(s) and s not in spec.do_not]
    spec.standards.extend(new_standards)
    spec.do_not.extend(new_do_not)

    recorded = 0
    for j in judged:
        # Coerce defensively: Judged is a plain dataclass (no validation), so a
        # direct caller can pass non-string input/output. Example requires str.
        out = as_str(j.output)
        spec.examples.append(Example(
            input=as_str(j.input),
            good_output=out if j.approved else None,
            bad_output=None if j.approved else out,
            why=as_opt_str(j.reason),
        ))
        recorded += 1

    return LearnResult(
        standards=new_standards,
        do_not=new_do_not,
        examples_recorded=recorded,
        standards_added=len(new_standards),
        do_not_added=len(new_do_not),
    )
