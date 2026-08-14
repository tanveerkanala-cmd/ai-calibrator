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
import re
from dataclasses import dataclass
from pathlib import Path

from .engines.base import Engine, parse_engine_spec
from .eval import run_eval
from .models import Project
from .store import atomic_write_text

# USD per 1M tokens (input, output). This table IS the recommendation: a model
# absent from it has no cost_score, so it drops out of the "cheapest that meets
# the bar" ranking without a word — and the survivor is then announced as the
# cheapest while a candidate that also passed may cost a fraction of it. It must
# therefore cover every provider the tool documents, not just Anthropic's ladder.
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI is a first-class provider (README, docs/USAGE.md, engines/openai.py),
    # and these two ids are the ones both put in front of the user.
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}

# `-latest` and a dated snapshot are the same model at the same price.
_ID_ALIAS = re.compile(r"-(?:latest|\d{8})$")


def model_price(model: str) -> tuple[float | None, float | None]:
    """Published $/Mtok (input, output) for a model id, or ``(None, None)``.

    Providers ship one model under a bare id, a ``-latest`` alias and a dated
    snapshot (``claude-haiku-4-5-20260401``). An exact-match lookup priced one
    spelling of the three, so the same model fell out of the cost ranking
    depending on how the user typed it — silently, the way any unpriced
    candidate does."""
    key = _ID_ALIAS.sub("", (model or "").strip().casefold())
    return PRICING.get(key, (None, None))


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
    local: bool = False  # runs on the owner's own hardware — no per-token bill

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

        A local candidate wins outright — it has no per-token bill, so nothing on
        a price list can undercut it (best pass rate breaks ties among locals).
        Otherwise prefer the lowest known cost, ties broken by higher pass rate;
        if no passing candidate has known pricing, fall back to the best pass
        rate. If nothing meets the bar, there is no recommendation.
        """
        passing = self.passing
        if not passing:
            return None
        # Without this a paid model gets called "the cheapest that meets the bar"
        # while a free local candidate on the same ladder also met it — sometimes
        # with a higher pass rate.
        local = [r for r in passing if r.local]
        if local:
            return max(local, key=lambda r: r.pass_rate)
        # A candidate with no published price cannot be placed on a cost ordering
        # at all, so it cannot be the answer to "which is cheapest" — keeping
        # PRICING current for every documented provider is what keeps that rare.
        # The report still carries the candidate, its pass rate and a null
        # cost_score, so an excluded passer stays visible in rightsize.json.
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
        model, provider = parse_engine_spec(ms)
        in_price, out_price = model_price(model)
        local = provider == "ollama"  # served from the owner's machine: no per-token cost
        try:
            subject = make_engine(ms)
            # Pass project_dir so each candidate is graded WITH RAG retrieval when
            # an index exists — rightsize recommends a model for production, and
            # production serves with RAG, so the comparison must match it.
            card = run_eval(project, subject, judge, run_id=f"rightsize-{i:02d}", project_dir=project_dir)
        except Exception as exc:
            results.append(ModelResult(spec=ms, model=model, pass_rate=0.0, passed=0, graded=0,
                                       in_price=in_price, out_price=out_price, error=str(exc),
                                       local=local))
            continue
        graded = [r for r in card.results if r.criteria]
        passed = sum(1 for r in graded if r.passed)
        results.append(ModelResult(spec=ms, model=model, pass_rate=card.pass_rate,
                                    passed=passed, graded=len(graded),
                                    in_price=in_price, out_price=out_price, local=local))

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
             "cost_score": r.cost_score, "local": r.local, "error": r.error}
            for r in report.results
        ],
    }


def save_rightsize(project_dir: str | Path, report: RightsizeReport) -> Path:
    target = Path(project_dir) / "evals" / "rightsize.json"
    return atomic_write_text(target, json.dumps(rightsize_dict(report), indent=2))
