"""Model rightsizing — the cheapest model that still meets your bar.

Calibration usually defaults to the strongest (most expensive) model. But the
only question that matters is: *does it pass the user's own tests?* This runs the
existing eval suite (subject = each candidate model, judge held fixed) and reports
pass rate against price, then recommends the cheapest candidate that clears the
threshold. The economics are concrete: "Haiku passes 94% at ~1/20th Opus's cost."

It reuses :func:`calibrator.eval.run_eval` unchanged and writes a single summary
artifact; it never mutates the project, so it is safe to run anytime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .engines.base import Engine, parse_engine_spec
from .eval import run_eval
from .models import Project
from .store import atomic_write_text

# USD per 1M tokens (input, output) for models we can price confidently. Unknown
# models still get ranked by pass rate; their cost just shows as unknown.
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Default candidate ladder when the caller doesn't specify one: the Claude tiers,
# strongest → cheapest. The user can pass any `model@provider` set instead.
DEFAULT_LADDER = [
    "claude-opus-4-8@anthropic",
    "claude-sonnet-4-6@anthropic",
    "claude-haiku-4-5@anthropic",
]


@dataclass
class ModelResult:
    spec: str
    model: str
    pass_rate: float
    passed: int
    graded: int
    in_price: float | None
    out_price: float | None
    error: str | None = None

    @property
    def cost_score(self) -> float | None:
        """A simple cheaper-is-lower proxy: blended input+output $/Mtok."""
        if self.in_price is None or self.out_price is None:
            return None
        return self.in_price + self.out_price


@dataclass
class RightsizeReport:
    results: list[ModelResult]
    threshold: float

    @property
    def passing(self) -> list[ModelResult]:
        return [r for r in self.results if r.error is None and r.pass_rate >= self.threshold]

    @property
    def recommended(self) -> ModelResult | None:
        """Cheapest candidate that meets the bar.

        Prefer the lowest known cost (ties broken by higher pass rate). If no
        passing candidate has known pricing, fall back to the best pass rate. If
        nothing meets the bar, there is no recommendation.
        """
        passing = self.passing
        if not passing:
            return None
        priced = [r for r in passing if r.cost_score is not None]
        if priced:
            return min(priced, key=lambda r: (r.cost_score, -r.pass_rate))
        return max(passing, key=lambda r: r.pass_rate)


def rightsize(
    project: Project,
    model_specs: list[str],
    judge: Engine,
    make_engine,
    *,
    threshold: float = 0.8,
    project_dir: str | Path | None = None,
) -> RightsizeReport:
    """Evaluate the test suite with each candidate as the subject; rank by cost.

    ``make_engine(spec) -> Engine`` builds a subject for each candidate. A model
    that can't be built or evaluated (missing creds, bad spec, API error) is
    recorded as an error rather than aborting the whole sweep.
    """
    spec = project.spec
    if spec is None or not project.tests:
        raise ValueError("Nothing to rightsize — run `calibrate compile` first.")

    results: list[ModelResult] = []
    for i, ms in enumerate(model_specs, start=1):
        model, _provider = parse_engine_spec(ms)
        in_price, out_price = PRICING.get(model, (None, None))
        try:
            subject = make_engine(ms)
            # Pass project_dir so each candidate is graded WITH RAG retrieval when
            # an index exists — rightsize recommends a model for production, and
            # production serves with RAG, so the comparison must match it.
            card = run_eval(project, subject, judge, run_id=f"rightsize-{i:02d}", project_dir=project_dir)
        except Exception as exc:
            results.append(ModelResult(spec=ms, model=model, pass_rate=0.0, passed=0, graded=0,
                                       in_price=in_price, out_price=out_price, error=str(exc)))
            continue
        graded = [r for r in card.results if r.criteria]
        passed = sum(1 for r in graded if r.passed)
        results.append(ModelResult(spec=ms, model=model, pass_rate=card.pass_rate,
                                    passed=passed, graded=len(graded),
                                    in_price=in_price, out_price=out_price))

    report = RightsizeReport(results=results, threshold=threshold)
    if project_dir is not None:
        save_rightsize(project_dir, report)
    return report


def rightsize_dict(report: RightsizeReport) -> dict:
    rec = report.recommended
    return {
        "threshold": report.threshold,
        "recommended": rec.spec if rec else None,
        "results": [
            {"spec": r.spec, "model": r.model, "pass_rate": r.pass_rate,
             "passed": r.passed, "graded": r.graded,
             "in_price": r.in_price, "out_price": r.out_price,
             "cost_score": r.cost_score, "error": r.error}
            for r in report.results
        ],
    }


def save_rightsize(project_dir: str | Path, report: RightsizeReport) -> Path:
    target = Path(project_dir) / "evals" / "rightsize.json"
    return atomic_write_text(target, json.dumps(rightsize_dict(report), indent=2))
