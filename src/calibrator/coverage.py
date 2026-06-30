"""Behavioral coverage — "test coverage, but for AI behavior".

The spec (§4) declares what the AI must do (standards, never-rules, edge cases)
and how that is measured (eval_criteria). The test suite exercises criteria via
``TestCase.expects``. This module answers the questions a careful calibrator
actually has:

- Which criteria have a **targeted** test, and which don't?
- Are any criteria graded only by "broad" tests (empty ``expects`` → graded
  against *all* criteria in :func:`calibrator.eval.run_eval`)? Broad grading is
  weak coverage: it never isolates the behavior.
- Is the spec under-measured — many standards/never-rules/edge cases but few
  criteria to check them?

It is fully deterministic (no engine), so it is fast, free, and exact. The
engine-assisted red-team (:mod:`calibrator.redteam`) and the calibration report
build on the structure defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import BehaviorSpec, TestCase


@dataclass
class CriterionCoverage:
    id: str
    description: str
    weight: str
    targeted_by: list[str] = field(default_factory=list)  # test ids that name this criterion

    @property
    def covered(self) -> bool:
        return bool(self.targeted_by)


@dataclass
class CoverageReport:
    criteria: list[CriterionCoverage]
    broad_tests: list[str]          # tests with empty `expects` → grade against ALL criteria
    orphan_expectations: list[str]  # criterion ids referenced by a test but absent from the spec
    standards: int
    do_not: int
    edge_cases: int
    total_tests: int

    @property
    def total_criteria(self) -> int:
        return len(self.criteria)

    @property
    def covered_criteria(self) -> list[CriterionCoverage]:
        return [c for c in self.criteria if c.covered]

    @property
    def uncovered_criteria(self) -> list[CriterionCoverage]:
        """Criteria with no *targeted* test (broad grade-all tests don't count —
        they never isolate the behavior)."""
        return [c for c in self.criteria if not c.covered]

    @property
    def coverage_rate(self) -> float:
        """Fraction of criteria with at least one targeted test."""
        if not self.criteria:
            return 0.0
        return len(self.covered_criteria) / len(self.criteria)

    @property
    def warnings(self) -> list[str]:
        w: list[str] = []
        if not self.criteria:
            w.append("No eval criteria — run `calibrate compile` first.")
            return w
        if not self.total_tests:
            w.append("No tests — run `calibrate compile` first.")
        # High-weight behavior that nothing targets is the most dangerous gap.
        high_uncovered = [c.id for c in self.uncovered_criteria if c.weight == "high"]
        if high_uncovered:
            w.append(f"HIGH-weight criteria with no targeted test: {', '.join(high_uncovered)}")
        if self.broad_tests and self.uncovered_criteria:
            w.append(
                f"{len(self.uncovered_criteria)} criterion(s) are only covered by "
                f"{len(self.broad_tests)} broad grade-all test(s) — add targeted tests."
            )
        if self.orphan_expectations:
            w.append(
                "Tests reference criterion ids not in the spec: "
                + ", ".join(sorted(set(self.orphan_expectations)))
            )
        # Under-measurement heuristic: many stated rules, few ways to check them
        # (criteria fewer than half the rules, with enough rules to matter).
        behavioral = self.standards + self.do_not + self.edge_cases
        if behavioral >= 4 and self.total_criteria * 2 < behavioral:
            w.append(
                f"{behavioral} behavioral rules (standards/never/edge) but only "
                f"{self.total_criteria} criteria — some behavior may be unmeasured."
            )
        return w


def analyze_coverage(spec: BehaviorSpec, tests: list[TestCase]) -> CoverageReport:
    """Build the deterministic behavioral-coverage report for a compiled spec."""
    valid_ids = {c.id for c in spec.eval_criteria}
    criteria = [
        CriterionCoverage(
            id=c.id,
            description=c.description,
            weight=c.weight.value,
            targeted_by=[t.id for t in tests if c.id in (t.expects or [])],
        )
        for c in spec.eval_criteria
    ]
    broad_tests = [t.id for t in tests if not t.expects]
    orphan = [e for t in tests for e in (t.expects or []) if e not in valid_ids]
    return CoverageReport(
        criteria=criteria,
        broad_tests=broad_tests,
        orphan_expectations=orphan,
        standards=len(spec.standards),
        do_not=len(spec.do_not),
        edge_cases=len(spec.edge_cases),
        total_tests=len(tests),
    )


def coverage_dict(report: CoverageReport) -> dict:
    """JSON-serializable view (for the API and the calibration report)."""
    return {
        "coverage_rate": report.coverage_rate,
        "total_criteria": report.total_criteria,
        "covered": len(report.covered_criteria),
        "uncovered": [{"id": c.id, "description": c.description, "weight": c.weight}
                      for c in report.uncovered_criteria],
        "criteria": [{"id": c.id, "weight": c.weight, "covered": c.covered, "targeted_by": c.targeted_by}
                     for c in report.criteria],
        "broad_tests": report.broad_tests,
        "orphan_expectations": sorted(set(report.orphan_expectations)),
        "counts": {"standards": report.standards, "do_not": report.do_not,
                   "edge_cases": report.edge_cases, "tests": report.total_tests},
        "warnings": report.warnings,
    }
