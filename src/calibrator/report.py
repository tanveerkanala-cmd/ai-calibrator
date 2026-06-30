"""Calibration report — a shareable "nutrition label" for the configured AI.

Quality is invisible: a stakeholder can't tell a carefully-calibrated AI from a
hand-waved prompt. This renders one honest artifact that makes it visible — a
**Calibration Confidence** score (how much behavior is tested × how much passes),
the specification at a glance, coverage gaps, the latest evaluation's weak spots,
and provenance (the ratified answers and materials the spec was built from).

Deterministic and engine-free, so it's instant and stable across runs.
"""

from __future__ import annotations

from pathlib import Path

from .coverage import CoverageReport
from .models import BehaviorSpec, Project, Scorecard
from .store import atomic_write_text


def calibration_confidence(coverage_rate: float, pass_rate: float, has_eval: bool) -> float:
    """How calibrated the AI is: tested-coverage × pass-rate.

    Zero until an eval exists — untested behavior is, honestly, uncalibrated. A
    product is only as trustworthy as the share of its behavior that is both
    *checked* and *passing*."""
    if not has_eval:
        return 0.0
    return round(coverage_rate * pass_rate, 4)


def report_dict(project: Project, coverage: CoverageReport, latest: Scorecard | None) -> dict:
    pass_rate = latest.pass_rate if latest else 0.0
    spec = project.spec
    return {
        "confidence": calibration_confidence(coverage.coverage_rate, pass_rate, latest is not None),
        "coverage_rate": coverage.coverage_rate,
        "pass_rate": pass_rate if latest else None,
        "latest_run": latest.run_id if latest else None,
        "standards": len(spec.standards) if spec else 0,
        "do_not": len(spec.do_not) if spec else 0,
        "edge_cases": len(spec.edge_cases) if spec else 0,
        "criteria": len(spec.eval_criteria) if spec else 0,
        "tests": len(project.tests),
        "uncovered_criteria": [c.id for c in coverage.uncovered_criteria],
        "warnings": coverage.warnings,
    }


def render_report(project: Project, coverage: CoverageReport, latest: Scorecard | None) -> str:
    """Render the calibration report as Markdown."""
    spec = project.spec or BehaviorSpec(goal=project.goal, task_type=project.task_type)
    pass_rate = latest.pass_rate if latest else 0.0
    conf = calibration_confidence(coverage.coverage_rate, pass_rate, latest is not None)
    L: list[str] = []

    L += [f"# Calibration Report — {project.name}", ""]
    L += [f"**Goal:** {project.goal}  ", f"**Task type:** {project.task_type.value}", ""]

    L += [f"## Calibration Confidence: {conf:.0%}", ""]
    L += [f"- Behavioral coverage: **{coverage.coverage_rate:.0%}** "
          f"({len(coverage.covered_criteria)}/{coverage.total_criteria} criteria targeted by a test)"]
    if latest:
        L += [f"- Latest pass rate: **{pass_rate:.0%}** (run `{latest.run_id}`)"]
        L += ["- Confidence = coverage × pass rate."]
    else:
        L += ["- Latest pass rate: — (no eval yet — run `calibrate eval`)"]
        L += ["- Confidence is 0 until the AI is evaluated: untested behavior is uncalibrated."]
    L += [""]

    L += ["## Specification", ""]
    if spec.persona and (spec.persona.voice or spec.persona.reading_level):
        rl = f" (reading level: {spec.persona.reading_level})" if spec.persona.reading_level else ""
        L += [f"- **Voice:** {spec.persona.voice or '—'}{rl}"]
    L += [f"- **Standards:** {len(spec.standards)} · **Never-rules:** {len(spec.do_not)} · **Edge cases:** {len(spec.edge_cases)}"]
    L += [f"- **Eval criteria:** {len(spec.eval_criteria)} · **Tests:** {len(project.tests)} · **Examples:** {len(spec.examples)}"]
    if spec.refusal_policy:
        L += [f"- **Refusal policy:** {spec.refusal_policy}"]
    L += [""]

    L += ["## Coverage", ""]
    if coverage.uncovered_criteria:
        L += ["Criteria with **no targeted test**:"]
        L += [f"- ⚠ `{c.id}` ({c.weight}): {c.description}" for c in coverage.uncovered_criteria]
    else:
        L += ["✓ Every criterion has a targeted test."]
    L += [f"- ⚠ {w}" for w in coverage.warnings]
    L += [""]

    if latest:
        L += ["## Latest evaluation", ""]
        L += [f"- Run `{latest.run_id}` — pass rate **{pass_rate:.0%}**"]
        fails = [r for r in latest.results if not r.passed and r.criteria]
        if fails:
            L += ["- Weak spots:"]
            for r in fails[:10]:
                why = "; ".join(c.rationale or c.criterion_id for c in r.criteria if not c.passed) or "—"
                L += [f"  - `{r.test_id}`: {why}"]
        else:
            L += ["- ✓ No failing tests."]
        L += [""]

    L += ["## Knowledge sources", ""]
    L += ([f"- {k}" for k in spec.knowledge_sources] or ["- (none)"])
    L += [""]

    L += ["## Provenance", ""]
    answered = [it for it in project.interview if it.answer]
    L += [f"The behavior spec was synthesized from **{len(answered)}** ratified interview "
          f"answer(s) and **{len(spec.knowledge_sources)}** material(s):"]
    for it in answered[:20]:
        L += [f"- **{it.dimension or it.id}** — {it.question} → _{it.answer}_"]
    L += [""]

    return "\n".join(L).rstrip() + "\n"


def save_report(project_dir: str | Path, markdown: str) -> Path:
    return atomic_write_text(Path(project_dir) / "calibration-report.md", markdown)
