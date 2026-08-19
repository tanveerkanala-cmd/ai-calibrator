"""The thesis experiment — the same suite, calibrated AI vs baseline, one delta.

Everything else in this tool measures the calibrated AI against the user's
standards; nothing measures what those standards *bought*. This runs the
identical test suite twice on the same model — once as deployed (compiled
prompt, RAG when indexed), once as a baseline — and reports the difference.

Fairness rules, all load-bearing:

- The baseline is the **goal line only** by default (`vs="goal"`): what a person
  gets by pasting their one-sentence ask into a chat window. Beating a model
  that was never told the job (`vs="bare"`) proves nothing; that floor exists
  but is not the default claim.
- The judge grades both sides under the **same context** — the compiled spec.
  The user's standards are the measuring stick regardless of what the subject
  was told; two runs graded under different judge context are not comparable.
- Deterministic checks are reported apart from judged criteria: they are graded
  by code, so that part of the delta owes nothing to any judge's opinion.
- A losing or tied specialist is reported in those words. This is an
  instrument, not a gate — `calibrate ci` is the gate.

Like `rightsize`, this never writes into the project's run history: `drift` and
`ci` baseline against saved scorecards, and a baseline-configuration run
appearing there would poison every later comparison. One summary artifact
(`evals/compare.json`) is the only thing saved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .engines.base import Engine
from .eval import ProgressFn, run_eval
from .models import Project, Scorecard
from .store import atomic_write_text

BASELINES = ("goal", "bare")


def baseline_system(project: Project, vs: str) -> str | None:
    """The baseline's system prompt: the goal line, or nothing at all."""
    if vs == "goal":
        return project.goal
    if vs == "bare":
        return None
    raise ValueError(f"Unknown baseline {vs!r} — expected one of: goal, bare.")


def compare_call_estimate(n_tests: int, judge_passes: int = 1) -> int:
    """Billed engine calls a compare will make: two full suite runs."""
    return 2 * n_tests * (1 + judge_passes)


@dataclass
class SideResult:
    pass_rate: float
    passed: int
    graded: int


@dataclass
class CriterionCompare:
    id: str
    kind: str  # "check" (graded by code) | "judged" (graded by the LLM judge)
    specialist_passed: int
    specialist_graded: int
    baseline_passed: int
    baseline_graded: int


@dataclass
class CompareReport:
    vs: str
    subject: str
    judge: str
    judge_passes: int
    n_tests: int
    retrieval: bool  # the calibrated side ran with RAG enabled
    partial: bool
    specialist: SideResult
    baseline: SideResult
    per_criterion: list[CriterionCompare] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.specialist.pass_rate - self.baseline.pass_rate


def _retrieval_live(project_dir: str | Path | None) -> bool:
    """Whether the calibrated side will ACTUALLY retrieve — not merely may.

    Passing ``project_dir`` enables retrieval; it does not make it happen. A
    project with no usable index retrieves nothing, and a report that says
    "retrieval ON" for a prompt-only bot misstates the experiment's conditions.
    """
    if project_dir is None:
        return False
    from . import rag
    return rag.index_available() and not rag.probe(project_dir)


def _side(card: Scorecard) -> SideResult:
    graded = [r for r in card.results if r.criteria]
    passed = sum(1 for r in graded if r.passed)
    return SideResult(pass_rate=card.pass_rate, passed=passed, graded=len(graded))


def _per_criterion(project: Project, spec_card: Scorecard, base_card: Scorecard) -> list[CriterionCompare]:
    rows = []
    for c in project.spec.eval_criteria if project.spec else []:
        counts = []
        for card in (spec_card, base_card):
            crs = [cr for r in card.results for cr in r.criteria if cr.criterion_id == c.id]
            counts += [sum(1 for cr in crs if cr.passed), len(crs)]
        rows.append(CriterionCompare(id=c.id, kind="check" if c.check else "judged",
                                     specialist_passed=counts[0], specialist_graded=counts[1],
                                     baseline_passed=counts[2], baseline_graded=counts[3]))
    return rows


def compare(
    project: Project,
    subject: Engine,
    judge: Engine,
    *,
    vs: str = "goal",
    judge_passes: int = 1,
    project_dir: str | Path | None = None,
    max_tests: int | None = None,
    on_progress: ProgressFn | None = None,
) -> CompareReport:
    """Run the suite as deployed, then as the baseline, and measure the gap.

    Same model, same tests, same judge, same judge context. The baseline run
    passes ``project_dir=None`` so it never retrieves — stripping the
    specialization means stripping ALL of it, not just the prompt.
    """
    override = baseline_system(project, vs)  # validates `vs` before anything is spent
    if project.spec is None or not project.tests:
        raise ValueError("Nothing to compare — run `calibrate compile` first.")
    spec_card = run_eval(project, subject, judge, run_id="compare-specialist",
                         judge_passes=judge_passes, project_dir=project_dir,
                         max_tests=max_tests, on_progress=on_progress)
    base_card = run_eval(project, subject, judge, run_id="compare-baseline",
                         judge_passes=judge_passes, project_dir=None,
                         max_tests=max_tests, on_progress=on_progress,
                         system_override=override)
    return CompareReport(
        vs=vs, subject=subject.name, judge=judge.name, judge_passes=judge_passes,
        n_tests=len(spec_card.results), retrieval=_retrieval_live(project_dir),
        partial=spec_card.partial or base_card.partial,
        specialist=_side(spec_card), baseline=_side(base_card),
        per_criterion=_per_criterion(project, spec_card, base_card),
    )


def summary_lines(report: CompareReport) -> list[str]:
    """The report as terminal lines — headline, verdict, receipts, conditions."""
    pp = round(report.delta * 100)
    lines = []
    if report.partial:
        lines.append("PARTIAL — a smoke run, not the full suite; these numbers are not a certification.")
    lines.append(
        f"baseline ({report.vs}): {report.baseline.pass_rate:.0%}  →  "
        f"calibrated: {report.specialist.pass_rate:.0%}   "
        f"(Δ {pp:+d}pp, N={report.n_tests} test(s))"
    )
    if pp > 0:
        lines.append(f"The calibrated AI outperformed the {report.vs} baseline by {pp} point(s) on this suite.")
    elif pp < 0:
        lines.append(f"The baseline outperformed the calibrated AI by {-pp} point(s) — "
                     "the specialization is not paying for itself on this suite.")
    else:
        lines.append("No measurable difference on this suite — the specialization is not paying for itself yet.")
    for c in report.per_criterion:
        lines.append(f"  {c.id} [{c.kind}]: baseline {c.baseline_passed}/{c.baseline_graded}"
                     f" → calibrated {c.specialist_passed}/{c.specialist_graded}")
    baseline_got = ("the one-line goal as its whole prompt" if report.vs == "goal"
                    else "no system prompt at all")
    lines.append(f"Conditions: subject={report.subject}, judge={report.judge}"
                 + (f", judged {report.judge_passes}× each" if report.judge_passes > 1 else "")
                 + f"; retrieval {'ON' if report.retrieval else 'OFF'} on the calibrated side; "
                 f"the baseline ran on the same model with {baseline_got} and no retrieval.")
    return lines


def save_compare(project_dir: str | Path, report: CompareReport) -> Path:
    """Write the summary artifact. Never a scorecard — see the module docstring."""
    path = Path(project_dir) / "evals" / "compare.json"
    atomic_write_text(path, json.dumps(asdict(report) | {"delta": report.delta}, indent=2))
    return path
