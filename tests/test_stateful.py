"""Stateful (model-based) property testing — Hypothesis generates random SEQUENCES
of project operations and shrinks any failing sequence to a minimal reproducer.
This targets bugs in operation INTERACTIONS and state consistency across a
workflow, which fixed-sequence example tests miss.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from hypothesis import strategies as st

from calibrator.ci import config_hash
from calibrator.models import BehaviorSpec, Check, EvalCriterion, Example, Project, Weight
from calibrator.models import TestCase as CaseModel  # aliased: pytest would try to collect `TestCase`
from calibrator.store import load_project, save_project

_TXT = st.text(st.characters(min_codepoint=32, codec="utf-8"), min_size=1, max_size=40)


class ProjectLifecycle(RuleBasedStateMachine):
    """Drive a real project dir through random mutation sequences; after EVERY
    operation the persisted state must stay consistent."""

    @initialize()
    def setup(self):
        self.dir = Path(tempfile.mkdtemp())
        p = Project(name="p", goal="a goal")
        p.spec = BehaviorSpec(goal="a goal")
        save_project(p, self.dir)
        self._crit_ids: set[str] = set()

    def _load(self) -> Project:
        return load_project(self.dir)

    def _save(self, p: Project) -> None:
        save_project(p, self.dir)

    @rule(text=_TXT)
    def add_standard(self, text):
        p = self._load()
        p.spec.standards.append(text)
        self._save(p)

    @rule(text=_TXT)
    def add_do_not(self, text):
        p = self._load()
        p.spec.do_not.append(text)
        self._save(p)

    @rule(desc=_TXT, weight=st.sampled_from(list(Weight)))
    def add_criterion(self, desc, weight):
        p = self._load()
        cid = f"c{len(p.spec.eval_criteria) + 1}"
        # skip if this id somehow already exists (keep ids unique)
        if any(c.id == cid for c in p.spec.eval_criteria):
            return
        p.spec.eval_criteria.append(EvalCriterion(id=cid, description=desc, weight=weight))
        self._crit_ids.add(cid)
        self._save(p)

    @rule(kind=st.sampled_from(["contains", "not_contains", "non_empty"]), value=_TXT)
    def attach_check(self, kind, value):
        p = self._load()
        if not p.spec.eval_criteria:
            return
        p.spec.eval_criteria[0].check = Check(kind=kind, value=value)
        self._save(p)

    @rule(inp=_TXT, out=_TXT)
    def add_example(self, inp, out):
        p = self._load()
        p.spec.examples.append(Example(input=inp, good_output=out))
        self._save(p)

    @rule(inp=_TXT)
    def add_test(self, inp):
        p = self._load()
        tid = f"t{len(p.tests) + 1}"
        if any(t.id == tid for t in p.tests):
            return
        p.tests.append(CaseModel(id=tid, input=inp))
        self._save(p)

    # --- invariants checked after EVERY rule ---------------------------------
    @invariant()
    def project_yaml_always_loads(self):
        self._load()   # must never raise — a mutation that corrupts project.yaml fails here

    @invariant()
    def save_reload_is_a_fixed_point(self):
        # loading, re-saving, and reloading must yield identical serialized state
        import yaml
        p = self._load()
        first = yaml.safe_dump(p.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        self._save(p)
        second = yaml.safe_dump(self._load().model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        assert first == second

    @invariant()
    def config_hash_is_deterministic(self):
        p = self._load()
        assert config_hash(p, self.dir) == config_hash(p, self.dir)

    @invariant()
    def criterion_ids_are_unique(self):
        ids = [c.id for c in self._load().spec.eval_criteria]
        assert len(ids) == len(set(ids))

    @invariant()
    def test_ids_are_unique(self):
        ids = [t.id for t in self._load().tests]
        assert len(ids) == len(set(ids))


TestProjectLifecycle = ProjectLifecycle.TestCase
TestProjectLifecycle.settings = settings(max_examples=60, stateful_step_count=25, deadline=None)
