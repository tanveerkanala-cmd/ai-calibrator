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
from .fmt import pct
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
        "weighted_score": latest.weighted_score if latest else None,
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

    L += [f"## Calibration Confidence: {pct(conf)}", ""]
    L += [f"- Behavioral coverage: **{pct(coverage.coverage_rate)}** "
          f"({len(coverage.covered_criteria)}/{coverage.total_criteria} criteria targeted by a test)"]
    if latest:
        L += [f"- Latest pass rate: **{pct(pass_rate)}** (run `{latest.run_id}`)"]
        L += [f"- Weighted score: **{pct(latest.weighted_score)}** "
              "(criteria weighted high=3 / medium=2 / low=1 — how much of what *matters* passed)"]
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
        L += [f"- Run `{latest.run_id}` — pass rate **{pct(pass_rate)}**"]
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


# --- badge + HTML certificate (the shareable, git-native face) ---------------

def badge_dict(project: Project, project_dir: str | Path) -> dict:
    """shields.io *endpoint* JSON — a build badge for AI behavior.

    Point `https://img.shields.io/endpoint?url=<your-served-badge-url>` (or
    commit badge.json and use a raw URL) and the repo wears its calibration the
    way it wears CI. Colors are honest: green only for a PASSING gate that
    certifies the CURRENT spec/subject."""
    from .ci import certification_status
    from .drift import load_scorecard
    from .eval import latest_run_id

    from .ci import latest_gate

    status, _ = certification_status(project, project_dir)
    latest = None
    # A PASSING badge must carry the number the GATE certified, not whatever ran
    # last: a `--max-tests 1` smoke run after a green gate would otherwise publish
    # its own 100% in the gate's green. Elsewhere (ungated), the latest full run is
    # the honest headline — a partial one is not a summary of anything.
    rid = None
    if status == "pass":
        gate = latest_gate(project_dir)
        rid = (gate or {}).get("run_id")
    if not rid:
        rid = latest_run_id(project_dir, full_only=True)
    if rid:
        try:
            latest = load_scorecard(project_dir, rid)
        except (FileNotFoundError, ValueError):
            latest = None
    n = len([r for r in latest.results if r.criteria]) if latest else 0

    if status == "pass" and latest:
        message, color = f"{pct(latest.pass_rate)} · {n} tests", "brightgreen"
    elif status == "fail":
        message, color = "gate failing", "red"
    elif status == "stale":
        message, color = "stale — re-run ci", "orange"
    elif latest:
        message, color = f"{pct(latest.pass_rate)} · ungated", "orange"
    else:
        message, color = "uncalibrated", "lightgrey"
    return {"schemaVersion": 1, "label": "calibrated", "message": message, "color": color}


def save_badge(project_dir: str | Path, badge: dict) -> Path:
    import json
    return atomic_write_text(Path(project_dir) / "badge.json", json.dumps(badge, indent=2))


_HTML_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration Certificate — {name}</title>
<style>
 body {{ font: 16px/1.55 -apple-system, "Segoe UI", sans-serif; color: #1a202c;
        max-width: 780px; margin: 3rem auto; padding: 0 1.25rem; }}
 h1 {{ font-size: 1.6rem; margin-bottom: .25rem; }}
 .goal {{ color: #4a5568; margin-top: 0; }}
 .conf {{ font-size: 3.2rem; font-weight: 700; margin: 1.2rem 0 0; }}
 .conf small {{ font-size: 1rem; font-weight: 400; color: #4a5568; display: block; }}
 .pass {{ color: #2f855a; }} .warn {{ color: #c05621; }} .fail {{ color: #c53030; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1.2rem 0; }}
 th, td {{ text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #e2e8f0; }}
 th {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; color: #718096; }}
 .tag {{ font-size: .75rem; padding: .1rem .45rem; border-radius: 999px; background: #edf2f7; }}
 footer {{ margin-top: 2.5rem; font-size: .85rem; color: #718096; }}
</style></head><body>
<h1>Calibration Certificate — {name}</h1>
<p class="goal">{goal}</p>
<p class="conf {conf_class}">{confidence_pct}<small>calibration confidence = behavioral coverage × pass rate</small></p>
<table>
<tr><th>Measure</th><th>Value</th></tr>
{measure_rows}
</table>
<h2>Certification gate</h2>
<table><tr><th>Stage</th><th>Status</th><th>Detail</th></tr>
{gate_rows}
</table>
<h2>What this AI is graded on</h2>
<table><tr><th>Criterion</th><th>Weight</th><th>Grading</th></tr>
{criteria_rows}
</table>
<footer>Generated by <a href="https://github.com/">AI Calibrator</a> on {date} —
run <code>calibrate ci</code> to re-certify. Numbers reflect the latest eval run{run_note}.</footer>
</body></html>
"""


def _esc(text: object) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_html_report(project: Project, coverage: CoverageReport, latest: Scorecard | None,
                       project_dir: str | Path) -> str:
    """A single-file, shareable HTML certificate — every number traces to a
    computed value (same sources as the markdown report)."""
    from .ci import certification_status, latest_gate

    pass_rate = latest.pass_rate if latest else 0.0
    conf = calibration_confidence(coverage.coverage_rate, pass_rate, latest is not None)
    status, detail = certification_status(project, project_dir)
    conf_class = "pass" if status == "pass" and conf >= 0.8 else ("fail" if status == "fail" else "warn")

    measures = [("Behavioral coverage",
                 f"{pct(coverage.coverage_rate)} ({len(coverage.covered_criteria)}/{coverage.total_criteria} criteria tested)")]
    if latest:
        graded = [r for r in latest.results if r.criteria]
        measures += [("Pass rate", f"{pct(pass_rate)} ({sum(1 for r in graded if r.passed)}/{len(graded)} tests)"),
                     ("Weighted score", f"{pct(latest.weighted_score)} (high=3 · medium=2 · low=1)")]
    else:
        measures += [("Pass rate", "— (no eval yet)")]
    measures += [("Certification", f"{status} — {detail}")]
    measure_rows = "\n".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in measures)

    gate = latest_gate(project_dir)
    if gate and isinstance(gate.get("stages"), list):
        gate_rows = "\n".join(
            f"<tr><td>{_esc(s.get('name'))}</td>"
            f"<td class=\"{ {'pass': 'pass', 'fail': 'fail'}.get(s.get('status'), 'warn') }\">{_esc(s.get('status'))}</td>"
            f"<td>{_esc(s.get('detail'))}</td></tr>"
            for s in gate["stages"] if isinstance(s, dict))
    else:
        gate_rows = "<tr><td colspan=\"3\">no gate on record — run <code>calibrate ci</code></td></tr>"

    spec = project.spec
    criteria_rows = "\n".join(
        f"<tr><td>{_esc(c.description)}</td><td><span class=\"tag\">{_esc(c.weight.value)}</span></td>"
        f"<td>{'deterministic check (' + _esc(c.check.kind) + ')' if c.check else 'LLM judge'}</td></tr>"
        for c in (spec.eval_criteria if spec else []))
    if not criteria_rows:
        criteria_rows = "<tr><td colspan=\"3\">no criteria yet</td></tr>"

    from datetime import datetime, timezone
    return _HTML_PAGE.format(
        name=_esc(project.name), goal=_esc(project.goal), confidence_pct=pct(conf), conf_class=conf_class,
        measure_rows=measure_rows, gate_rows=gate_rows, criteria_rows=criteria_rows,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        run_note=f" (`{_esc(latest.run_id)}`)" if latest else "",
    )


def save_html_report(project_dir: str | Path, html: str) -> Path:
    return atomic_write_text(Path(project_dir) / "calibration-report.html", html)
