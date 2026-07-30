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

    Ordering is by STAKEHOLDER NAME (not the dict's insertion order), so the
    indices are reproducible from the same set of specs regardless of how the
    sources were ordered. This is load-bearing: `detect` and `apply` are separate
    API requests, and a client that reordered `sources` between them would
    otherwise get misaligned indices and silently drop the wrong statement."""
    statements: list[Statement] = []
    i = 1
    for name, spec in sorted(named_specs.items()):
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


def scalar_conflicts(named_specs: dict[str, BehaviorSpec]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Scalar behavior fields where stakeholders disagree.

    Returns ``[(field, [(stakeholder, value), ...]), ...]`` for every field where two
    or more stakeholders supplied DIFFERENT non-empty values. These never reach
    ``detect_conflicts`` (which only ever sees standards and never-rules), so
    without this a merge silently keeps one team's refusal policy and drops
    another's while reporting "no conflicts". Needs no engine — string inequality
    is the whole test."""
    fields = (
        ("persona.voice", lambda sp: sp.persona.voice if sp.persona else None),
        ("persona.reading_level", lambda sp: sp.persona.reading_level if sp.persona else None),
        ("format", lambda sp: sp.format),
        ("refusal_policy", lambda sp: sp.refusal_policy),
    )
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for label, get in fields:
        raw = [(n, get(sp)) for n, sp in sorted(named_specs.items())]
        vals = [(n, v) for n, v in raw if is_str(v) and v.strip()]
        if len({v for _, v in vals}) > 1:
            out.append((label, vals))
    return out


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
    # Criteria: collapse only TRUE duplicates. Criterion ids are engine-generated
    # snake_case labels, so two independently compiled specs routinely both have
    # `accuracy` or `tone` meaning different things — deduping on the bare id would
    # silently discard one stakeholder's criterion along with its deterministic
    # check. Namespace a genuine collision instead of losing it.
    criteria: list[EvalCriterion] = []
    seen_crit: dict[str, EvalCriterion] = {}
    for name, sp in sorted(named_specs.items()):
        for c in sp.eval_criteria:
            prior = seen_crit.get(c.id)
            if prior is None:
                seen_crit[c.id] = c
                criteria.append(c)
            elif (prior.description, prior.weight, prior.check) == (c.description, c.weight, c.check):
                continue  # the same criterion contributed twice — genuinely a dup
            else:
                alt = c.model_copy(deep=True)
                alt.id = f"{name}_{c.id}"
                n = 2
                while alt.id in seen_crit:
                    alt.id = f"{name}_{c.id}_{n}"
                    n += 1
                seen_crit[alt.id] = alt
                criteria.append(alt)
    examples: list[Example] = []
    seen_ex: set[str] = set()
    for sp in specs:
        for ex in sp.examples:
            if ex.input not in seen_ex:
                seen_ex.add(ex.input)
                examples.append(ex)
    knowledge = _dedup([k for sp in specs for k in sp.knowledge_sources])

    # Scalar behavior fields. These render straight into the system prompt, so
    # picking one by argument order silently ships a different AI depending on the
    # order of --from flags — and the conflict never appears in the audit file.
    # Resolve deterministically by stakeholder name (never insertion order) and let
    # the caller report the values that lost. See scalar_conflicts().
    by_name = sorted(named_specs.items())
    # voice and reading_level are two independently reported fields, so resolve
    # them independently: taking one stakeholder's whole persona object ships a
    # reading level the audit file says lost, and drops an uncontested one. The
    # base copy keeps any extra persona keys a hand-edited spec carried.
    base = next((sp.persona for _, sp in by_name
                 if sp.persona and (sp.persona.voice or sp.persona.reading_level)), None)
    persona = base.model_copy(deep=True) if base is not None else Persona()
    persona.voice = next((sp.persona.voice for _, sp in by_name
                          if sp.persona and sp.persona.voice), None)
    persona.reading_level = next((sp.persona.reading_level for _, sp in by_name
                                  if sp.persona and sp.persona.reading_level), None)
    fmt = next((sp.format for _, sp in by_name if sp.format), None)
    refusal = next((sp.refusal_policy for _, sp in by_name if sp.refusal_policy), None)

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
