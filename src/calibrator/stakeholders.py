"""Multi-stakeholder calibration — merge several stakeholders' specs, reconcile.

In an org, one AI must satisfy several voices: legal, sales, support, brand.
Each calibrates their own project; their standards then *contradict* ("always add
a disclaimer" vs "keep it punchy"; "offer a refund" vs "never promise refunds").
A single-voice spec silently averages or loses those. This module makes the
conflict explicit and reconcilable:

1. ``gather`` collects every standard / never-rule across stakeholders, tagged by
   author and given a stable index.
2. ``detect_conflicts`` asks an engine to find pairs that directly contradict.
3. A decider rules on each (keep A, keep B, or write a merged rule); the rulings
   become ``drops`` + ``additions``.
4. ``build_merged_spec`` produces the unified spec (everyone's non-conflicting
   rules + the rulings), with the conflicts and decisions kept as an audit.

The engine references statements by INDEX (gather is deterministic), so detect
and apply stay consistent without server-side session state.
"""

from __future__ import annotations

from dataclasses import dataclass

from .coerce import as_list, as_str, is_str
from .engines.base import Engine, require_object
from .models import BehaviorSpec, EdgeCase, EvalCriterion, Example, Persona, Project, TaskType

CONFLICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                    "explanation": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["a", "b", "explanation", "severity"],
            },
        }
    },
    "required": ["conflicts"],
}

_CONFLICT_SYSTEM = (
    "You find genuine contradictions between an AI's behavior rules contributed by "
    "different stakeholders. Two rules conflict when satisfying one forces breaking "
    "the other (e.g. 'always add a legal disclaimer' vs 'never add disclaimers', or "
    "'offer a refund' vs 'never promise refunds'). Reference rules by their number. "
    "Report ONLY real conflicts — not mere differences in topic or emphasis. Respond "
    "with JSON only, matching the schema."
)


@dataclass
class Statement:
    idx: int
    text: str
    kind: str           # "standard" | "do_not"
    stakeholder: str


@dataclass
class Conflict:
    id: str
    a: Statement
    b: Statement
    explanation: str
    severity: str


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def gather(named_specs: dict[str, BehaviorSpec]) -> list[Statement]:
    """Flatten every standard / never-rule across specs, tagged + indexed (1-based).

    Deterministic in the dict's order, so indices are stable between detect and
    apply for the same stakeholder ordering."""
    statements: list[Statement] = []
    i = 1
    for name, spec in named_specs.items():
        for s in spec.standards:
            if is_str(s):
                statements.append(Statement(i, s, "standard", name))
                i += 1
        for d in spec.do_not:
            if is_str(d):
                statements.append(Statement(i, d, "do_not", name))
                i += 1
    return statements


def detect_conflicts(statements: list[Statement], engine: Engine) -> list[Conflict]:
    """Engine pass: find pairs of statements that directly contradict."""
    if len(statements) < 2:
        return []
    block = "\n".join(
        f"{s.idx}. [{s.stakeholder}] ({'NEVER' if s.kind == 'do_not' else 'STANDARD'}) {s.text}"
        for s in statements
    )
    prompt = (
        "Behavior rules contributed by different stakeholders:\n\n"
        f"{block}\n\n"
        "Identify the pairs that directly conflict."
    )
    out = require_object(engine.complete(prompt, system=_CONFLICT_SYSTEM, schema=CONFLICT_SCHEMA), "conflict detector")
    by_idx = {s.idx: s for s in statements}
    conflicts: list[Conflict] = []
    seen: set[tuple[int, int]] = set()
    k = 1
    for c in as_list(out.get("conflicts")):
        if not isinstance(c, dict):
            continue
        a, b = c.get("a"), c.get("b")
        if not isinstance(a, int) or not isinstance(b, int) or a == b:
            continue
        if a not in by_idx or b not in by_idx:
            continue
        pair = (min(a, b), max(a, b))
        if pair in seen:
            continue
        seen.add(pair)
        sev = c.get("severity")
        conflicts.append(Conflict(
            id=f"k{k}", a=by_idx[a], b=by_idx[b],
            explanation=as_str(c.get("explanation")),
            severity=sev if sev in ("low", "medium", "high") else "medium",
        ))
        k += 1
    return conflicts


def conflict_dict(c: Conflict) -> dict:
    return {
        "id": c.id, "severity": c.severity, "explanation": c.explanation,
        "a": {"idx": c.a.idx, "text": c.a.text, "kind": c.a.kind, "stakeholder": c.a.stakeholder},
        "b": {"idx": c.b.idx, "text": c.b.text, "kind": c.b.kind, "stakeholder": c.b.stakeholder},
    }


def build_merged_spec(
    named_specs: dict[str, BehaviorSpec],
    *,
    goal: str,
    task_type: TaskType,
    drops: set[int] | None = None,
    additions: list[str] | None = None,
) -> BehaviorSpec:
    """The unified spec: everyone's rules minus the dropped (losing) statements,
    plus any merged additions, plus an additive union of the non-conflicting
    dimensions (edge cases, criteria, examples, knowledge, persona)."""
    drops = drops or set()
    additions = [a for a in (additions or []) if is_str(a)]
    statements = gather(named_specs)

    standards = [s.text for s in statements if s.kind == "standard" and s.idx not in drops]
    do_not = [s.text for s in statements if s.kind == "do_not" and s.idx not in drops]
    standards = _dedup(standards + additions)
    do_not = _dedup(do_not)

    specs = list(named_specs.values())
    # Additive unions across stakeholders (dedup by a natural key).
    edge_cases: list[EdgeCase] = []
    seen_edges: set[tuple[str, str]] = set()
    for sp in specs:
        for e in sp.edge_cases:
            key = (e.situation, e.ruling)
            if key not in seen_edges:
                seen_edges.add(key)
                edge_cases.append(e)
    criteria: list[EvalCriterion] = []
    seen_crit: set[str] = set()
    for sp in specs:
        for c in sp.eval_criteria:
            if c.id not in seen_crit:
                seen_crit.add(c.id)
                criteria.append(c)
    examples: list[Example] = []
    seen_ex: set[str] = set()
    for sp in specs:
        for ex in sp.examples:
            if ex.input not in seen_ex:
                seen_ex.add(ex.input)
                examples.append(ex)
    knowledge = _dedup([k for sp in specs for k in sp.knowledge_sources])

    persona = next((sp.persona for sp in specs if sp.persona and (sp.persona.voice or sp.persona.reading_level)), Persona())
    fmt = next((sp.format for sp in specs if sp.format), None)
    refusal = next((sp.refusal_policy for sp in specs if sp.refusal_policy), None)

    return BehaviorSpec(
        goal=goal, task_type=task_type, persona=persona,
        standards=standards, do_not=do_not, edge_cases=edge_cases,
        format=fmt, refusal_policy=refusal, knowledge_sources=knowledge,
        eval_criteria=criteria, examples=examples,
    )


def merged_project(
    out_name: str,
    named_specs: dict[str, BehaviorSpec],
    *,
    goal: str,
    task_type: TaskType,
    drops: set[int] | None = None,
    additions: list[str] | None = None,
) -> Project:
    spec = build_merged_spec(named_specs, goal=goal, task_type=task_type, drops=drops, additions=additions)
    return Project(name=out_name, goal=goal, task_type=task_type, spec=spec)
