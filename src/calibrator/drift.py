"""Behavior drift detection — has the AI's behavior changed vs a baseline?

Providers silently update models; a spec edit can fix one thing and break
another. Drift detection re-runs the test suite and compares the fresh scorecard
to a baseline, reporting the pass-rate delta and exactly which tests flipped
pass→fail (regressions) or fail→pass (improvements).

It is CI-friendly: ``calibrate drift`` exits non-zero when behavior regresses
beyond tolerance, so it can gate a deploy or run on a schedule after a model
bump. Reuses :func:`calibrator.eval.run_eval`; the comparison is deterministic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .engines.base import Engine
from .eval import next_run_id, run_eval, save_scorecard
from .models import Scorecard


@dataclass
class DriftReport:
    baseline_run: str
    candidate_run: str
    baseline_rate: float
    candidate_rate: float
    regressed_tests: list[str]  # passed in baseline, failed in candidate
    fixed_tests: list[str]      # failed in baseline, passed in candidate
    tolerance: float

    @property
    def delta(self) -> float:
        return self.candidate_rate - self.baseline_rate

    @property
    def regressed(self) -> bool:
        """Drift worth alerting on: a pass-rate drop beyond tolerance, OR any
        individual test that went from passing to failing."""
        return self.delta < -self.tolerance or bool(self.regressed_tests)


def compare_scorecards(baseline: Scorecard, candidate: Scorecard, *, tolerance: float = 0.0) -> DriftReport:
    if not isinstance(baseline, Scorecard) or not isinstance(candidate, Scorecard):
        raise TypeError("baseline and candidate must be Scorecard instances")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be a finite number >= 0 (got {tolerance!r})")
    before = {r.test_id: r.passed for r in baseline.results}
    after = {r.test_id: r.passed for r in candidate.results}
    shared = before.keys() & after.keys()
    regressed = sorted(t for t in shared if before[t] and not after[t])
    fixed = sorted(t for t in shared if not before[t] and after[t])
    return DriftReport(
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        baseline_rate=baseline.pass_rate,
        candidate_rate=candidate.pass_rate,
        regressed_tests=regressed,
        fixed_tests=fixed,
        tolerance=tolerance,
    )


def load_scorecard(project_dir: str | Path, run_id: str) -> Scorecard:
    # run_id is used as a path component (and may be user-supplied via
    # --baseline / --candidate) — reject empty or traversal-bearing ids.
    if not isinstance(run_id, str) or not run_id.strip() \
            or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError(f"invalid run id: {run_id!r}")
    f = Path(project_dir) / "evals" / run_id / "scorecard.json"
    if not f.exists():
        raise FileNotFoundError(f"no scorecard at evals/{run_id}/")
    return Scorecard.model_validate(json.loads(f.read_text()))


def run_drift(
    project,
    subject: Engine,
    judge: Engine,
    *,
    baseline: Scorecard,
    project_dir: str | Path,
    tolerance: float = 0.0,
) -> tuple[DriftReport, Scorecard]:
    """Run a fresh eval (the candidate), persist it, and compare to ``baseline``."""
    candidate = run_eval(project, subject, judge, run_id=next_run_id(project_dir), project_dir=project_dir)
    save_scorecard(project_dir, candidate)
    return compare_scorecards(baseline, candidate, tolerance=tolerance), candidate


def drift_dict(report: DriftReport) -> dict:
    return {
        "baseline_run": report.baseline_run,
        "candidate_run": report.candidate_run,
        "baseline_rate": report.baseline_rate,
        "candidate_rate": report.candidate_rate,
        "delta": report.delta,
        "regressed": report.regressed,
        "regressed_tests": report.regressed_tests,
        "fixed_tests": report.fixed_tests,
        "tolerance": report.tolerance,
    }
