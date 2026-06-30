"""M4 — Evaluate: run the configured AI on the test cases and grade it.

Two layers: a **deterministic guard** (an empty output fails immediately, no
judge call) and **LLM-as-judge** — the judge engine scores each output against
the eval criteria the test targets. Produces a `Scorecard`; results persist
under `<project>/evals/<run-id>/`. The refine loop lives in `pipeline.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .coerce import as_list, as_opt_str, as_str
from .compile import render_system_prompt
from .engines.base import Engine, require_object
from .models import CriterionResult, Project, Scorecard, TestResult
from .store import atomic_write_text

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "criterion_id": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "score": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["criterion_id", "passed", "score", "rationale"],
            },
        }
    },
    "required": ["results"],
}

_JUDGE_SYSTEM = (
    "You are a strict grader. Judge the AI output against each listed criterion "
    "independently. Pass a criterion only if the output clearly satisfies it. "
    "Give a 0-1 score and a one-line rationale per criterion. Respond with JSON "
    "only, matching the schema."
)


def _as_float(value: object) -> float:
    """Coerce a model-supplied score to a finite float, defaulting to 0.0.

    A non-compliant judge could return a non-numeric or non-finite score; that
    must not raise or poison the scorecard arithmetic."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _judge(
    judge: Engine,
    test_input: str,
    output: str,
    criteria: list[tuple[str, str]],
) -> list[CriterionResult]:
    block = "\n".join(f"- {cid}: {desc}" for cid, desc in criteria)
    prompt = (
        f"INPUT:\n{test_input}\n\nAI OUTPUT:\n{output}\n\nCRITERIA:\n{block}\n\n"
        "Grade each criterion."
    )
    out = require_object(judge.complete(prompt, system=_JUDGE_SYSTEM, schema=JUDGE_SCHEMA), "judge")
    by_id = {
        r.get("criterion_id"): r
        for r in as_list(out.get("results"))
        if isinstance(r, dict)
    }
    results = []
    for cid, _ in criteria:
        r = by_id.get(cid, {})
        results.append(
            CriterionResult(
                criterion_id=cid,
                passed=bool(r.get("passed", False)),
                score=_as_float(r.get("score", 0.0)),
                rationale=as_opt_str(r.get("rationale")),
            )
        )
    return results


def run_eval(
    project: Project,
    subject: Engine,
    judge: Engine,
    *,
    run_id: str = "run-0001",
) -> Scorecard:
    """Run each test on the subject and grade the output against its criteria."""
    spec = project.spec
    if spec is None:
        raise ValueError("No behavior spec — run `calibrate compile` first.")
    system = render_system_prompt(spec)
    crit_by_id = {c.id: c for c in spec.eval_criteria}

    results: list[TestResult] = []
    for test in project.tests:
        # Coerce defensively: a misbehaving subject can return a non-string
        # (despite the str contract); as_str makes that an empty output (caught
        # by the guard below) instead of an AttributeError on .strip().
        output = as_str(subject.complete(test.input, system=system))
        expected = [cid for cid in (test.expects or list(crit_by_id)) if cid in crit_by_id]
        criteria = [(cid, crit_by_id[cid].description) for cid in expected]

        if not output.strip():
            # Deterministic guard — don't spend a judge call on empty output.
            crs = [
                CriterionResult(criterion_id=cid, passed=False, score=0.0, rationale="empty output")
                for cid, _ in criteria
            ]
        elif not criteria:
            crs = []
        else:
            crs = _judge(judge, test.input, output, criteria)

        results.append(TestResult(test_id=test.id, output=output, criteria=crs))

    return Scorecard(run_id=run_id, results=results)


def next_run_id(project_dir: str | Path) -> str:
    """Next sequential run id (run-0001, run-0002, …) under <project>/evals/.

    Not atomic on its own: concurrent callers on the same project can read the
    same state and collide on an id. The CLI and API serialize every eval /
    drift / calibrate run with ``store.project_lock``, so the product paths are
    safe; a direct library caller running these concurrently on one project must
    hold that lock too (it is intentionally not taken here — these helpers run
    *inside* the caller's locked section, and the lock is not re-entrant)."""
    evals = Path(project_dir) / "evals"
    n = 0
    if evals.exists():
        for d in evals.iterdir():
            if d.is_dir() and d.name.startswith("run-"):
                try:
                    n = max(n, int(d.name.split("-", 1)[1]))
                except ValueError:
                    pass
    return f"run-{n + 1:04d}"


def latest_run_id(project_dir: str | Path) -> str | None:
    """The most recent ``run-NNNN`` that has a saved scorecard, or None."""
    evals = Path(project_dir) / "evals"
    best: str | None = None
    n = 0
    if evals.exists():
        for d in evals.iterdir():
            if d.is_dir() and d.name.startswith("run-") and (d / "scorecard.json").exists():
                try:
                    k = int(d.name.split("-", 1)[1])
                except ValueError:
                    continue
                if k > n:
                    n, best = k, d.name
    return best


def save_scorecard(project_dir: str | Path, card: Scorecard) -> Path:
    """Write scorecard.json + failures.jsonl under <project>/evals/<run-id>/."""
    d = Path(project_dir) / "evals" / card.run_id
    atomic_write_text(d / "scorecard.json", json.dumps(card.model_dump(mode="json"), indent=2))
    fails = [r for r in card.results if not r.passed]
    atomic_write_text(d / "failures.jsonl",
                      "".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in fails))
    return d
