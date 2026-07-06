"""`calibrate ci` — the whole verification surface as ONE gate.

Each verify feature answers its own question (lint: is the spec sound? eval: does
the AI meet it? drift: did it regress vs the baseline? snapshot: did any answer's
*text* change?). CI wants one command and one exit code. This composes them, in
the cheap-to-expensive order, into a single pass/fail:

    lint  →  eval  →  drift (vs previous / --baseline run)  →  snapshot (vs golden)

Lint errors stop the gate before any engine call is spent (a spec with no
criteria can't be meaningfully evaluated). Skips are honest: a stage that can't
run (no baseline yet, no golden pinned) reports *skip*, never a silent pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Union

from .compile import render_system_prompt
from .drift import compare_scorecards, load_scorecard
from .engines.base import Engine
from .eval import latest_run_id, next_run_id, run_eval, save_scorecard
from .fmt import pct, pct_delta
from .lint import lint_spec, lint_unknown_fields
from .models import Project, Scorecard
from .snapshot import compare, load_golden, outputs_of
from .store import atomic_write_text

GATE_FILE = "last-gate.json"  # under <project>/evals/

# An engine, or a zero-arg factory for one. Factories are resolved only AFTER
# lint passes — a lint-broken spec must not demand credentials/engines it will
# never use (and an engine problem must not mask the lint verdict).
EngineOrFactory = Union[Engine, Callable[[], Engine]]


def _resolve(engine: EngineOrFactory) -> Engine:
    return engine() if callable(engine) and not hasattr(engine, "complete") else engine


@dataclass
class CiStage:
    name: str        # "lint" | "eval" | "drift" | "snapshot"
    status: str      # "pass" | "fail" | "skip"
    detail: str


@dataclass
class CiResult:
    stages: list[CiStage] = field(default_factory=list)
    run_id: str | None = None            # the eval run this gate produced
    pass_rate: float | None = None
    weighted_score: float | None = None

    @property
    def ok(self) -> bool:
        return all(s.status != "fail" for s in self.stages)


def run_ci(
    project: Project,
    subject: EngineOrFactory,
    judge: EngineOrFactory,
    *,
    project_dir: str | Path,
    threshold: float = 0.8,
    tolerance: float = 0.0,
    judge_passes: int = 1,
    baseline: str | None = None,
) -> CiResult:
    """Run the full gate; persists the eval scorecard like `calibrate eval` does.

    ``baseline``: run id to drift against (default: the latest run before this
    one). ``subject``/``judge`` may be engines or zero-arg factories (factories
    are called only after lint passes). The caller validates numeric inputs."""
    result = CiResult()

    # 1. lint — free; errors mean the gate can't measure anything meaningful.
    report = lint_spec(project.spec, project.tests)
    report.issues.extend(lint_unknown_fields(project))
    n_err, n_warn = len(report.errors), len(report.warnings)
    detail = f"{n_err} error(s), {n_warn} warning(s)"
    if n_err:
        first = report.errors[0].message
        result.stages.append(CiStage("lint", "fail", f"{detail} — {first}"))
        result.stages.append(CiStage("eval", "skip", "skipped — fix lint errors first"))
        result.stages.append(CiStage("drift", "skip", "skipped — fix lint errors first"))
        result.stages.append(CiStage("snapshot", "skip", "skipped — fix lint errors first"))
        save_gate(project, result, project_dir)
        return result
    result.stages.append(CiStage("lint", "pass", detail))

    # 2. eval — the fresh run under test. Baseline resolves BEFORE it exists.
    subject, judge = _resolve(subject), _resolve(judge)
    baseline_id = baseline or latest_run_id(project_dir)
    card = run_eval(project, subject, judge, run_id=next_run_id(project_dir), judge_passes=judge_passes)
    save_scorecard(project_dir, card)
    result.run_id = card.run_id
    result.pass_rate = card.pass_rate
    result.weighted_score = card.weighted_score
    graded = [r for r in card.results if r.criteria]
    passed = sum(1 for r in graded if r.passed)
    ok = card.pass_rate >= threshold
    eval_detail = (f"{card.run_id}: {pct(card.pass_rate)} pass ({passed}/{len(graded)}), "
                   f"weighted {pct(card.weighted_score)}, threshold {threshold:.0%}")
    if not ok:  # disambiguate the boundary: "80% pass, threshold 80%, fail" reads wrong
        eval_detail += f" — below threshold ({card.pass_rate:.1%} < {threshold:.0%})"
    result.stages.append(CiStage("eval", "pass" if ok else "fail", eval_detail))

    # 3. drift — vs the previous (or pinned) run.
    result.stages.append(_drift_stage(project_dir, baseline_id, card, tolerance))

    # 4. snapshot — vs the pinned golden outputs.
    result.stages.append(_snapshot_stage(project_dir, card))

    save_gate(project, result, project_dir)  # the record `calibrate run` boots against
    return result


def _drift_stage(project_dir: str | Path, baseline_id: str | None, card: Scorecard, tolerance: float) -> CiStage:
    if not baseline_id:
        return CiStage("drift", "skip", "no baseline run yet — the next `ci` will drift against this one")
    try:
        base = load_scorecard(project_dir, baseline_id)
    except (FileNotFoundError, ValueError) as exc:
        return CiStage("drift", "fail", f"baseline {baseline_id!r} unusable: {exc}")
    d = compare_scorecards(base, card, tolerance=tolerance)
    detail = f"vs {baseline_id}: Δ {pct_delta(d.delta)}"
    if d.regressed:
        what = f", regressed: {', '.join(d.regressed_tests)}" if d.regressed_tests else ""
        return CiStage("drift", "fail", detail + what)
    fixed = f", fixed: {', '.join(d.fixed_tests)}" if d.fixed_tests else ""
    return CiStage("drift", "pass", detail + ", no regressions" + fixed)


def _snapshot_stage(project_dir: str | Path, card: Scorecard) -> CiStage:
    golden = load_golden(project_dir)
    if golden is None:
        return CiStage("snapshot", "skip", "no golden pinned — `calibrate snapshot` to pin outputs")
    diff = compare(golden, outputs_of(card))
    if diff.drifted:
        parts = []
        if diff.changed:
            parts.append(f"changed: {', '.join(diff.changed)}")
        if diff.removed:
            parts.append(f"removed: {', '.join(diff.removed)}")
        return CiStage("snapshot", "fail", "; ".join(parts))
    extra = f" ({len(diff.added)} new test(s) not in golden)" if diff.added else ""
    return CiStage("snapshot", "pass", f"{len(golden)} output(s) match the golden{extra}")


def ci_dict(result: CiResult) -> dict:
    return {
        "ok": result.ok,
        "run_id": result.run_id,
        "pass_rate": result.pass_rate,
        "weighted_score": result.weighted_score,
        "stages": [{"name": s.name, "status": s.status, "detail": s.detail} for s in result.stages],
    }


# --- the persisted gate record (what `calibrate run` boots against) ----------

def config_hash(project: Project) -> str:
    """Fingerprint of what a gate certifies: the compiled system prompt, the
    subject binding, the grading contract (criteria + checks + weights), and the
    test suite. If ANY of it changes — including new tests absorbed from live
    feedback — a previous certification is STALE and must be re-earned."""
    if project.spec is None:
        return ""
    # Reordering list entries in the YAML is not a behavior change and must not
    # spuriously stale a certification. Criteria and tests are already sorted
    # below; render the prompt from a COPY whose prompt-affecting list fields are
    # sorted too, so reordering standards / never-rules / edge-cases is inert to
    # the hash — matching how criteria/tests are treated. (audit: round 18)
    spec = project.spec.model_copy(deep=True)
    spec.standards = sorted(spec.standards)
    spec.do_not = sorted(spec.do_not)
    spec.edge_cases = sorted(spec.edge_cases, key=lambda e: (e.situation, e.ruling))
    criteria = [{"id": c.id, "description": c.description, "weight": c.weight.value,
                 "check": {"kind": c.check.kind, "value": c.check.value} if c.check else None}
                for c in sorted(project.spec.eval_criteria, key=lambda c: c.id)]
    tests = [{"id": t.id, "input": t.input, "expects": t.expects, "follow_ups": t.follow_ups}
             for t in sorted(project.tests, key=lambda t: t.id)]
    material = (render_system_prompt(spec)
                + "\n@@subject=" + project.engines.subject
                # The judge is part of the grading contract — a different judge can
                # score the same outputs differently, so switching it must re-earn
                # the certification just as a criterion change does.
                + "\n@@judge=" + project.engines.judge
                + "\n@@criteria=" + json.dumps(criteria, sort_keys=True, ensure_ascii=False)
                + "\n@@tests=" + json.dumps(tests, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def save_gate(project: Project, result: CiResult, project_dir: str | Path) -> Path:
    """Persist the gate verdict (pass AND fail — a red gate is a fact, not a secret)."""
    record = ci_dict(result)
    record["config_hash"] = config_hash(project)
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return atomic_write_text(Path(project_dir) / "evals" / GATE_FILE,
                             json.dumps(record, indent=2, ensure_ascii=False))


def latest_gate(project_dir: str | Path) -> dict | None:
    """The most recent persisted gate verdict, or None if `ci` has never run."""
    f = Path(project_dir) / "evals" / GATE_FILE
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def certification_status(project: Project, project_dir: str | Path) -> tuple[str, str]:
    """(status, detail) for serving: pass | fail | stale | none.

    - pass:  the latest gate passed AND certifies the CURRENT config
    - stale: a gate exists but the spec/subject changed since it ran
    - fail:  the latest gate failed (and still matches the current config)
    - none:  `calibrate ci` has never been run"""
    gate = latest_gate(project_dir)
    if gate is None:
        return "none", "no gate on record — run `calibrate ci` to certify this AI"
    run = gate.get("run_id") or "?"
    when = gate.get("finished_at") or "unknown time"
    if gate.get("config_hash") != config_hash(project):
        return "stale", (f"the last gate ({run}, {when}) certified a DIFFERENT spec/subject — "
                         "re-run `calibrate ci`")
    if gate.get("ok"):
        rate = gate.get("pass_rate")
        detail = f"gate {run} passed at {when}" + (f" ({pct(rate)} pass rate)" if isinstance(rate, (int, float)) else "")
        return "pass", detail
    failed = [s["name"] for s in gate.get("stages", []) if isinstance(s, dict) and s.get("status") == "fail"]
    return "fail", (f"gate {run} FAILED at {when}" + (f" ({', '.join(failed)})" if failed else "")
                    + " — fix and re-run `calibrate ci`")
