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
from dataclasses import dataclass, field
from pathlib import Path

from .engines.base import Engine
from .eval import next_run_id, run_eval, save_scorecard
from .models import Scorecard, same_question


@dataclass
class DriftReport:
    baseline_run: str
    candidate_run: str
    baseline_rate: float        # each run's own headline rate, over everything it graded
    candidate_rate: float
    regressed_tests: list[str]  # passed in baseline, failed in candidate
    fixed_tests: list[str]      # failed in baseline, passed in candidate
    tolerance: float
    # Ids both runs graded whose recorded question CHANGED between them (a
    # recompile rewrites t1..tN under the same ids). Their verdicts are not
    # comparable and are excluded from the counts above rather than being
    # silently treated as the same test.
    incomparable_tests: list[str] = field(default_factory=list)
    # How many ids the two runs actually compared: shared, and asking the same
    # question in both. This is the population the flip lists AND the delta are
    # computed over — everything else describes only one of the two runs.
    compared: int = 0
    # Pass rate over those `compared` ids alone, per run. The whole-run rates
    # above are each true of their own run and are NOT subtractable: they
    # denominate over every graded test, including the ones this comparison
    # refused to make. None when nothing was comparable.
    baseline_shared_rate: float | None = None
    candidate_shared_rate: float | None = None

    @property
    def delta(self) -> float | None:
        """Pass-rate change over the tests both runs graded and both still ask.

        None when nothing was comparable: there is no measurement, and every
        stand-in lies — 0.0 reads as "behavior held", and the whole-run
        difference reports a collapse the model never caused."""
        if self.baseline_shared_rate is None or self.candidate_shared_rate is None:
            return None
        return self.candidate_shared_rate - self.baseline_shared_rate

    @property
    def comparable(self) -> bool:
        """Whether the two runs still share any question worth comparing.

        False when `compile` replaced every shared probe, and false when the two
        runs graded no id in common. The rates are still computed and still
        real, but they describe two different exams, so their difference is not
        a result — which is the same fact `delta is None` states."""
        return self.compared > 0

    @property
    def regressed(self) -> bool:
        """Drift worth alerting on: a larger SHARE of the comparable suite went
        from passing to failing than tolerance allows.

        Reads the comparable population only. A recompile that swapped in harder
        questions, or a test the baseline never graded, must not read as a
        regression the model never caused.

        One rule, not two. This was `delta < -tolerance OR any flip`, and once
        the delta is confined to the shared population the first clause can only
        fire when the second already has — which made the tolerance the user
        asked for unreachable. Gating the flipped share instead leaves the
        default (0.0 — a single flip is drift) answering identically on every
        possible suite, so nothing changes for anyone who does not pass the flag.

        The share is GROSS, not the net delta: an improvement elsewhere is not
        evidence that a test which started failing did not. `--tolerance 0.05`
        therefore means "up to 5% of the compared tests may flip down", which is
        what makes it usable against judge nondeterminism, where a 100-test suite
        with a 2% per-test flip rate alarms on ~87% of clean runs."""
        if not self.compared:
            return False
        return len(self.regressed_tests) / self.compared > self.tolerance


def compare_scorecards(baseline: Scorecard, candidate: Scorecard, *, tolerance: float = 0.0) -> DriftReport:
    if not isinstance(baseline, Scorecard) or not isinstance(candidate, Scorecard):
        raise TypeError("baseline and candidate must be Scorecard instances")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be a finite number >= 0 (got {tolerance!r})")
    # Match on the QUESTION, not just the slot — see models.same_question. Keying
    # on the id alone compared an old run's verdicts to tests that now ask
    # something else, and reported the difference as a regression or a fix.
    by_id_before = {r.test_id: r for r in baseline.results}
    regressed, fixed, incomparable = [], [], []
    compared = base_passes = cand_passes = 0
    for cand in candidate.results:
        base = by_id_before.get(cand.test_id)
        if base is None:
            continue
        if not same_question(base, cand):
            incomparable.append(cand.test_id)
            continue
        # Counted whether or not the verdict moved: "we compared these and
        # nothing flipped" is a real result, and the caller has to be able to
        # tell it apart from "there was nothing left to compare".
        compared += 1
        # The rates the delta subtracts are tallied HERE, over this exact set —
        # the whole-run rates include tests the loop above just refused to
        # compare, so their difference gates on results this comparison
        # deliberately threw away.
        base_passes += base.passed
        cand_passes += cand.passed
        if base.passed and not cand.passed:
            regressed.append(cand.test_id)
        elif not base.passed and cand.passed:
            fixed.append(cand.test_id)
    regressed, fixed, incomparable = sorted(regressed), sorted(fixed), sorted(set(incomparable))
    return DriftReport(
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        baseline_rate=baseline.pass_rate,
        candidate_rate=candidate.pass_rate,
        regressed_tests=regressed,
        fixed_tests=fixed,
        tolerance=tolerance,
        incomparable_tests=incomparable,
        compared=compared,
        baseline_shared_rate=(base_passes / compared) if compared else None,
        candidate_shared_rate=(cand_passes / compared) if compared else None,
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
    return Scorecard.model_validate(json.loads(f.read_text(encoding="utf-8")))


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
        # The pair `delta` is the difference of — a surface that prints a Δ must
        # print these beside it, not the whole-run rates above.
        "baseline_shared_rate": report.baseline_shared_rate,
        "candidate_shared_rate": report.candidate_shared_rate,
        "delta": report.delta,
        "regressed": report.regressed,
        "regressed_tests": report.regressed_tests,
        "fixed_tests": report.fixed_tests,
        "incomparable_tests": report.incomparable_tests,
        "tolerance": report.tolerance,
        "compared": report.compared,
        "comparable": report.comparable,
    }
