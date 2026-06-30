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
    criteria_changed: list[str] = field(default_factory=list)  # id in both, description/weight differs

    @property
    def changed(self) -> bool:
        return any((self.standards_added, self.standards_removed, self.do_not_added, self.do_not_removed,
                    self.edge_cases_added, self.edge_cases_removed,
                    self.criteria_added, self.criteria_removed, self.criteria_changed))


def _added_removed(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    bset, aset = set(before), set(after)
    return [x for x in after if x not in bset], [x for x in before if x not in aset]


def diff_specs(before: BehaviorSpec, after: BehaviorSpec) -> SpecDiff:
    """Diff ``before`` → ``after`` across standards, never-rules, edge cases, criteria."""
    d = SpecDiff()
    d.standards_added, d.standards_removed = _added_removed(before.standards, after.standards)
    d.do_not_added, d.do_not_removed = _added_removed(before.do_not, after.do_not)
    be = [f"When {e.situation}: {e.ruling}" for e in before.edge_cases]
    ae = [f"When {e.situation}: {e.ruling}" for e in after.edge_cases]
    d.edge_cases_added, d.edge_cases_removed = _added_removed(be, ae)

    bc = {c.id: c for c in before.eval_criteria}
    ac = {c.id: c for c in after.eval_criteria}
    d.criteria_added = [i for i in ac if i not in bc]
    d.criteria_removed = [i for i in bc if i not in ac]
    d.criteria_changed = [i for i in ac if i in bc
                          and (ac[i].description != bc[i].description or ac[i].weight != bc[i].weight)]
    return d


def diff_dict(d: SpecDiff) -> dict:
    return {
        "changed": d.changed,
        "standards": {"added": d.standards_added, "removed": d.standards_removed},
        "do_not": {"added": d.do_not_added, "removed": d.do_not_removed},
        "edge_cases": {"added": d.edge_cases_added, "removed": d.edge_cases_removed},
        "criteria": {"added": d.criteria_added, "removed": d.criteria_removed, "changed": d.criteria_changed},
    }
