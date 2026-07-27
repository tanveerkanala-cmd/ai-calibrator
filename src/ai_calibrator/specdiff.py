"""Behavior diff — what changed between two specs.

`drift` compares *scorecards* (did behavior regress?); this compares the *specs*
themselves (what rules changed). Useful for reviewing the effect of a refine,
teach, merge, or hand-edit — or PR-reviewing a behavior change before shipping.
Deterministic, no engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import BehaviorSpec


@dataclass
class SpecDiff:
    standards_added: list[str] = field(default_factory=list)
    standards_removed: list[str] = field(default_factory=list)
    do_not_added: list[str] = field(default_factory=list)
    do_not_removed: list[str] = field(default_factory=list)
    edge_cases_added: list[str] = field(default_factory=list)
    edge_cases_removed: list[str] = field(default_factory=list)
    criteria_added: list[str] = field(default_factory=list)
    criteria_removed: list[str] = field(default_factory=list)
    criteria_changed: list[str] = field(default_factory=list)  # id in both, description/weight/check differs
    knowledge_added: list[str] = field(default_factory=list)
    knowledge_removed: list[str] = field(default_factory=list)
    # Scalar behavior fields: (field, before, after). These render straight into the
    # system prompt, so a change here changes the deployed AI as surely as a new
    # standard does — reporting "no behavior change" for a reversed refusal policy
    # is worse than reporting nothing.
    fields_changed: list[tuple[str, str | None, str | None]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any((self.standards_added, self.standards_removed, self.do_not_added, self.do_not_removed,
                    self.edge_cases_added, self.edge_cases_removed,
                    self.criteria_added, self.criteria_removed, self.criteria_changed,
                    self.fields_changed, self.knowledge_added, self.knowledge_removed))


def _added_removed(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    bset, aset = set(before), set(after)
    return [x for x in after if x not in bset], [x for x in before if x not in aset]


# Scalar spec fields that render into the system prompt, as (label, accessor).
_SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("goal", "goal"),
    ("persona.voice", "persona.voice"),
    ("persona.reading_level", "persona.reading_level"),
    ("format", "format"),
    ("refusal_policy", "refusal_policy"),
)


def _get(spec: BehaviorSpec, path: str) -> str | None:
    obj: object = spec
    for part in path.split("."):
        obj = getattr(obj, part, None)
    return obj if isinstance(obj, str) else None


def diff_specs(before: BehaviorSpec, after: BehaviorSpec) -> SpecDiff:
    """Diff ``before`` → ``after`` across the goal, persona, format, refusal policy,
    standards, never-rules, edge cases, and criteria (description, weight, check)."""
    d = SpecDiff()
    d.fields_changed = [
        (label, _get(before, path), _get(after, path))
        for label, path in _SCALAR_FIELDS
        if _get(before, path) != _get(after, path)
    ]
    d.standards_added, d.standards_removed = _added_removed(before.standards, after.standards)
    d.do_not_added, d.do_not_removed = _added_removed(before.do_not, after.do_not)
    be = [f"When {e.situation}: {e.ruling}" for e in before.edge_cases]
    ae = [f"When {e.situation}: {e.ruling}" for e in after.edge_cases]
    d.edge_cases_added, d.edge_cases_removed = _added_removed(be, ae)
    # render_system_prompt appends a grounding paragraph iff knowledge_sources is
    # non-empty, so this genuinely changes the deployed prompt.
    d.knowledge_added, d.knowledge_removed = _added_removed(
        list(before.knowledge_sources), list(after.knowledge_sources))

    bc = {c.id: c for c in before.eval_criteria}
    ac = {c.id: c for c in after.eval_criteria}
    d.criteria_added = [i for i in ac if i not in bc]
    d.criteria_removed = [i for i in bc if i not in ac]
    # `check` counts: retargeting a deterministic check (contains "30-day" →
    # "60-day") changes the grading contract without touching the description.
    d.criteria_changed = [i for i in ac if i in bc
                          and (ac[i].description != bc[i].description
                               or ac[i].weight != bc[i].weight
                               or ac[i].check != bc[i].check)]
    return d


def diff_dict(d: SpecDiff) -> dict:
    return {
        "changed": d.changed,
        "fields": [{"field": f, "before": b, "after": a} for f, b, a in d.fields_changed],
        "standards": {"added": d.standards_added, "removed": d.standards_removed},
        "do_not": {"added": d.do_not_added, "removed": d.do_not_removed},
        "edge_cases": {"added": d.edge_cases_added, "removed": d.edge_cases_removed},
        "criteria": {"added": d.criteria_added, "removed": d.criteria_removed, "changed": d.criteria_changed},
        "knowledge_sources": {"added": d.knowledge_added, "removed": d.knowledge_removed},
    }
