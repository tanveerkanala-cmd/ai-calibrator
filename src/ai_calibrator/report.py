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

from collections import Counter

from .coerce import as_str
from .coverage import CoverageReport
from .fmt import pct
from .models import BehaviorSpec, Project, Scorecard, test_input_hash
from .store import atomic_write_text


def calibration_confidence(coverage_rate: float, pass_rate: float, has_eval: bool) -> float:
    """How calibrated the AI is: tested-coverage × pass-rate.

    Zero until an eval exists — untested behavior is, honestly, uncalibrated. A
    product is only as trustworthy as the share of its behavior that is both
    *checked* and *passing*."""
    if not has_eval:
        return 0.0
    return round(coverage_rate * pass_rate, 4)


def _unratified_answers(project: Project) -> list[str]:
    """Dimensions whose interview answer the tool wrote and nobody reviewed.

    The spec is synthesized from these answers, so they are upstream of every
    standard, criterion and test — and the tool cannot know what the materials
    leave unstated, so a draft can assert policy nobody wrote. This report is the
    artifact that gets shared as evidence the AI is calibrated; a spec built on
    unreviewed guesses is a different claim from one built on ratified answers,
    and the difference has to be visible here."""
    return [it.dimension for it in project.interview if it.unratified]


def _matches(test, result) -> bool:
    """Does ``result`` record a run of ``test``?

    The id alone is not identity: `compile` mints t1..tN positionally and
    regenerates the whole range every time it runs, so the ordinary workflow
    (compile -> eval -> answer more questions -> compile -> report) replaces every
    probe with different text under the same id. Matching on the id alone hands
    an old run's verdicts to tests that have never been executed — exactly the
    unearned credit the ungraded/dropped receipts exist to prevent.

    A result with no recorded hash predates the field. It is matched by id, so
    existing scorecards keep reporting exactly as they did; every run from here
    on records the content and gets the stricter check.
    """
    if result.test_id != test.id:
        return False
    return result.input_hash is None or result.input_hash == test_input_hash(test)


# NOTE: drift.py answers the same question for two RESULTS (models.same_question).
# Both are statements of one rule — "an id names a slot, not a question" — and when
# only this one existed, `calibrate drift` compared recompiled suites as if the
# tests had not changed. Keep them in step.


def _graded(latest: Scorecard | None) -> list:
    return [r for r in latest.results if r.criteria] if latest else []


def _ungraded_tests(project: Project, latest: Scorecard | None) -> list[str]:
    """Ids of tests in the CURRENT suite that ``latest`` never graded.

    A scorecard is a claim about the suite as it stood when the run happened, and
    a run that was full when it ran keeps saying so forever. Every command that
    teaches the AI something new — `absorb`, `redteam --promote`,
    `examples-to-tests` — grows the suite past the newest scorecard, and `compile`
    rewrites the probes under their existing ids."""
    if latest is None:
        return []
    graded = _graded(latest)
    return sorted(t.id for t in project.tests
                  if not any(_matches(t, r) for r in graded))


def _dropped_tests(project: Project, latest: Scorecard | None) -> list[str]:
    """Ids ``latest`` graded that the CURRENT suite no longer asks — failures first.

    The mirror image of :func:`_ungraded_tests`, and the reason it matters: the
    confidence divides by the CURRENT suite, so removing a test the run failed
    raises the headline number without the AI having changed. That can be an
    honest edit (a test that encoded the wrong rule), but it must never be
    invisible — a score that goes up is a claim, and this is the receipt for it."""
    if latest is None:
        return []
    gone = [r for r in _graded(latest)
            if not any(_matches(t, r) for t in project.tests)]
    return [r.test_id for r in gone if not r.passed] + [r.test_id for r in gone if r.passed]


def _suite_pass_rate(project: Project, latest: Scorecard | None) -> float:
    """Pass rate recomputed over the CURRENT suite — the confidence's second factor.

    ``Scorecard.pass_rate`` divides by what that run graded, which is the honest
    number *for that run*. Confidence is a claim about the AI as configured now,
    so it divides by the suite as it stands now: a test no run has executed is
    unproven, and unproven behavior cannot be credited as passing."""
    if latest is None or not project.tests:
        return 0.0
    # Consume each passing result at most once. Two suite rows sharing an id
    # otherwise both claim the same verdict, which turns a duplicated failing
    # test into a pass.
    available = Counter(id(r) for r in _graded(latest) if r.passed)
    passing = [r for r in _graded(latest) if r.passed]
    credited = 0
    for test in project.tests:
        for r in passing:
            if available[id(r)] and _matches(test, r):
                available[id(r)] -= 1
                credited += 1
                break
    return credited / len(project.tests)


def report_dict(project: Project, coverage: CoverageReport, latest: Scorecard | None) -> dict:
    pass_rate = latest.pass_rate if latest else 0.0
    spec = project.spec
    return {
        "confidence": calibration_confidence(coverage.coverage_rate,
                                             _suite_pass_rate(project, latest), latest is not None),
        "coverage_rate": coverage.coverage_rate,
        "pass_rate": pass_rate if latest else None,
        # The confidence's second factor, published so every surface shows the
        # number the headline was actually built from rather than recomputing one.
        "suite_pass_rate": _suite_pass_rate(project, latest) if latest else None,
        "weighted_score": latest.weighted_score if latest else None,
        "latest_run": latest.run_id if latest else None,
        "standards": len(spec.standards) if spec else 0,
        "do_not": len(spec.do_not) if spec else 0,
        "edge_cases": len(spec.edge_cases) if spec else 0,
        "criteria": len(spec.eval_criteria) if spec else 0,
        "tests": len(project.tests),
        "uncovered_criteria": [c.id for c in coverage.uncovered_criteria],
        "unratified_answers": _unratified_answers(project),
        "ungraded_tests": _ungraded_tests(project, latest),
        "dropped_tests": _dropped_tests(project, latest),
        "warnings": coverage.warnings,
    }


def render_report(project: Project, coverage: CoverageReport, latest: Scorecard | None) -> str:
    """Render the calibration report as Markdown."""
    spec = project.spec or BehaviorSpec(goal=project.goal, task_type=project.task_type)
    pass_rate = latest.pass_rate if latest else 0.0
    suite_rate = _suite_pass_rate(project, latest)
    ungraded = _ungraded_tests(project, latest)
    dropped = _dropped_tests(project, latest)
    # _dropped_tests puts the failures first; keep that split so the note can name
    # them — a removed failure is the case that moves the number.
    failed_dropped = ({r.test_id for r in latest.results if r.criteria and not r.passed}
                      if latest else set())
    conf = calibration_confidence(coverage.coverage_rate, suite_rate, latest is not None)
    L: list[str] = []

    L += [f"# Calibration Report — {project.name}", ""]
    L += [f"**Goal:** {project.goal}  ", f"**Task type:** {project.task_type.value}", ""]

    L += [f"## Calibration Confidence: {pct(conf)}", ""]
    unratified = _unratified_answers(project)
    if unratified:
        L += [f"> ⚠ **This spec was built from {len(unratified)} interview answer(s) the tool "
              "wrote and nobody reviewed** "
              f"({', '.join(unratified[:5])}{', …' if len(unratified) > 5 else ''}). It cannot "
              "know what your materials leave unstated, so anything those answers assert is "
              "policy that may never have been yours — and it is now a standard the AI follows "
              "and a criterion it is graded against. A high score below measures agreement with "
              "those answers, not with your materials. Re-run `calibrate interview` to review "
              "them.", ""]
    L += [f"- Behavioral coverage: **{pct(coverage.coverage_rate)}** "
          f"({len(coverage.covered_criteria)}/{coverage.total_criteria} criteria targeted by a test)"]
    if latest:
        L += [f"- Latest pass rate: **{pct(pass_rate)}** (run `{latest.run_id}`)"]
        L += [f"- Weighted score: **{pct(latest.weighted_score)}** "
              "(criteria weighted high=3 / medium=2 / low=1 — how much of what *matters* passed)"]
        if ungraded:
            L += [f"- ⚠ **{len(ungraded)} of {len(project.tests)}** current test(s) have never been "
                  f"graded — `{latest.run_id}` predates them; re-run `calibrate eval`."]
        if dropped:
            failed = [t for t in dropped if t in failed_dropped]
            L += [f"- ⚠ **{len(dropped)}** test(s) `{latest.run_id}` graded are no longer in the "
                  + (f"suite, **{len(failed)}** of which it FAILED "
                     f"({', '.join(f'`{t}`' for t in failed[:5])}"
                     + (", …" if len(failed) > 5 else "") + "). Removing a failing test raises the "
                     "confidence above without the AI having changed."
                     if failed else "suite. The rate above is over the tests that remain.")]
        if suite_rate == pass_rate:
            L += ["- Confidence = coverage × pass rate."]
        else:
            L += [f"- Confidence = coverage × **{pct(suite_rate)}**, the pass rate recomputed over "
                  "the tests in the CURRENT suite."]
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
        elif ungraded:
            # "No failing tests" would be a flat falsehood about the tests it skipped.
            n_graded = len([r for r in latest.results if r.criteria])
            L += [f"- ✓ No failures among the {n_graded} test(s) this run graded."]
        else:
            L += ["- ✓ No failing tests."]
        if ungraded:
            shown = ", ".join(f"`{t}`" for t in ungraded[:10])
            more = f" (+{len(ungraded) - 10} more)" if len(ungraded) > 10 else ""
            L += [f"- ⚠ Never graded by this run: {shown}{more}"]
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
<p class="conf {conf_class}">{confidence_pct}<small>calibration confidence = {conf_formula}</small></p>
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
    suite_rate = _suite_pass_rate(project, latest)
    conf = calibration_confidence(coverage.coverage_rate, suite_rate, latest is not None)
    # The subtitle is the reader's only way to reconcile the headline with the rows
    # below it, so it has to name the rate the headline was actually built from.
    conf_formula = ("behavioral coverage × pass rate" if suite_rate == pass_rate else
                    "behavioral coverage × pass rate over the CURRENT test suite")
    status, detail = certification_status(project, project_dir)
    conf_class = "pass" if status == "pass" and conf >= 0.8 else ("fail" if status == "fail" else "warn")

    measures = [("Behavioral coverage",
                 f"{pct(coverage.coverage_rate)} ({len(coverage.covered_criteria)}/{coverage.total_criteria} criteria tested)")]
    if latest:
        graded = [r for r in latest.results if r.criteria]
        n_tests = len(project.tests)
        measures += [("Pass rate", f"{pct(pass_rate)} ({sum(1 for r in graded if r.passed)}/{len(graded)} tests)"),
                     ("Weighted score", f"{pct(latest.weighted_score)} (high=3 · medium=2 · low=1)"),
                     # Without this the coverage and pass-rate rows sit side by side
                     # counting two different test sets, and nothing says so.
                     ("Suite coverage of this run",
                      f"{n_tests - len(_ungraded_tests(project, latest))}/{n_tests} current test(s) graded")]
        # The certificate is the artifact that gets shared, so it must carry the
        # same receipt the markdown does: removing a test this run FAILED raises
        # the headline without the AI having changed, and a reader multiplying the
        # rows above would otherwise never see it.
        dropped = _dropped_tests(project, latest)
        if dropped:
            failed = {r.test_id for r in latest.results if r.criteria and not r.passed}
            n_failed = sum(1 for t in dropped if t in failed)
            measures += [("Graded but no longer in the suite",
                          f"{len(dropped)} test(s)"
                          + (f", {n_failed} of which this run FAILED" if n_failed else ""))]
    else:
        measures += [("Pass rate", "— (no eval yet)")]
    # The certificate is what gets shared as evidence. A spec synthesized from
    # answers nobody reviewed is a materially different claim, so it belongs
    # beside the score rather than in a footnote.
    unratified = _unratified_answers(project)
    if unratified:
        measures += [("Spec provenance",
                      f"⚠ built from {len(unratified)} unreviewed drafted answer(s) — "
                      "may assert policy your materials never stated")]
    measures += [("Certification", f"{status} — {detail}")]
    measure_rows = "\n".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in measures)

    gate = latest_gate(project_dir)
    if gate and isinstance(gate.get("stages"), list):
        gate_rows = "\n".join(
            f"<tr><td>{_esc(as_str(s.get('name')))}</td>"
            f"<td class=\"{ {'pass': 'pass', 'fail': 'fail'}.get(as_str(s.get('status')), 'warn') }\">"
            f"{_esc(as_str(s.get('status')))}</td>"
            f"<td>{_esc(as_str(s.get('detail')))}</td></tr>"
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
        conf_formula=conf_formula,
        measure_rows=measure_rows, gate_rows=gate_rows, criteria_rows=criteria_rows,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        run_note=f" (`{_esc(latest.run_id)}`)" if latest else "",
    )


def save_html_report(project_dir: str | Path, html: str) -> Path:
    return atomic_write_text(Path(project_dir) / "calibration-report.html", html)
