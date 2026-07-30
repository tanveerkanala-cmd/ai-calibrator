"""M4 — the calibrate loop: evaluate → diagnose → refine → re-evaluate.

When a round's pass rate is below threshold, the refiner engine proposes
additional standards to add to the spec (the source of truth). The system prompt
is re-rendered from the updated spec on the next round, so fixes flow through
automatically. Loops until the threshold is met or rounds run out.
"""

from __future__ import annotations

import math

from .coerce import as_list
from .engines.base import Engine, require_object
from .eval import next_run_id, run_eval, save_scorecard
from .models import Project, Scorecard

# A generous upper bound on refine rounds — refinement converges in a handful of
# rounds; a very large value almost always signals a mistaken/abusive argument.
MAX_REFINE_ROUNDS = 100

REFINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "new_standards": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["new_standards"],
}

_REFINE_SYSTEM = (
    "You improve an AI's behavior spec to fix evaluation failures. Propose 1-4 "
    "additional, specific standards (instructions) that would fix the failures "
    "without breaking other behavior. Be concrete and minimal. Respond with JSON "
    "only, matching the schema."
)


def refine_spec(project: Project, scorecard: Scorecard, engine: Engine) -> list[str]:
    """Diagnose failures → propose new standards to add to the spec."""
    fails = []
    for r in scorecard.results:
        for c in r.criteria:
            if not c.passed:
                fails.append(f"- test {r.test_id} / {c.criterion_id}: {c.rationale or 'failed'}")
    if not fails:
        return []
    prompt = (
        f"GOAL: {project.goal}\n\n"
        f"The AI failed these checks:\n" + "\n".join(fails) + "\n\n"
        "Propose additional standards to fix them."
    )
    out = require_object(engine.complete(prompt, system=_REFINE_SYSTEM, schema=REFINE_SCHEMA), "refiner")
    # Dedup within the batch AND against the spec (both lists): a refiner that
    # proposes the same standard every round otherwise bloats the spec without
    # bound, and a "standard" that already exists as a never-rule would put the
    # same sentence on both sides of the contract.
    existing = set(project.spec.standards) | set(project.spec.do_not) if project.spec else set()
    fresh: list[str] = []
    for s in as_list(out.get("new_standards")):
        if isinstance(s, str):
            st = s.strip()  # dedup + store by trimmed content, so " foo " ≠ "foo" doesn't slip a dup in
            if st and st not in existing:
                existing.add(st)
                fresh.append(st)
    return fresh


def calibrate_loop(
    project: Project,
    subject: Engine,
    judge: Engine,
    refiner: Engine,
    *,
    threshold: float = 0.8,
    max_rounds: int = 3,
    judge_passes: int = 1,
    project_dir=None,
    on_spec_change=None,
) -> list[Scorecard]:
    """Eval → (if below threshold) refine the spec → re-eval, up to max_rounds.

    Mutates ``project.spec.standards`` as it refines. Returns one scorecard per
    round so callers can show the pass-rate trajectory.

    ``on_spec_change(project)`` fires immediately after each round's standards are
    added, BEFORE the next round runs. Persist there: each round's scorecard is
    saved as soon as it is graded, so a caller that only saves after the loop
    would — on any interruption, a Ctrl-C or a 529 — leave scorecards on disk that
    were earned under a spec the project never recorded. The newest full run and
    the pinned golden would then describe a spec that never existed.

    Raises ``ValueError`` for nonsensical controls — ``max_rounds < 1`` (the
    loop would silently run zero rounds and return no scorecards) or a
    non-finite / out-of-range ``threshold`` (e.g. ``NaN``, against which the
    ``pass_rate >= threshold`` check is always False, so the loop would never
    stop early). Failing fast beats surprising the caller with empty or
    never-converging results.
    """
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError(f"max_rounds must be an integer >= 1 (got {max_rounds!r})")
    if max_rounds > MAX_REFINE_ROUNDS:
        raise ValueError(f"max_rounds must be <= {MAX_REFINE_ROUNDS} (got {max_rounds})")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be a finite number in [0, 1] (got {threshold!r})")
    if project.spec is None:
        # The loop refines the spec's standards, so an uncompiled project has
        # nothing to refine — fail here rather than partway through round one,
        # after the subject and judge have already been billed for an eval.
        raise ValueError("No spec to refine — run `calibrate compile` first.")
    spec = project.spec

    cards: list[Scorecard] = []
    for rnd in range(1, max_rounds + 1):
        run_id = next_run_id(project_dir) if project_dir is not None else f"run-{rnd:04d}"
        card = run_eval(project, subject, judge, run_id=run_id, judge_passes=judge_passes,
                        project_dir=project_dir)
        if project_dir is not None:
            save_scorecard(project_dir, card)
        cards.append(card)

        if card.pass_rate >= threshold or rnd == max_rounds:
            break
        new_standards = refine_spec(project, card, refiner)  # deduped vs the spec
        if not new_standards:
            break  # nothing NEW to add — more rounds would just repeat themselves
        # Update the source of truth; the next round re-renders the system prompt.
        spec.standards.extend(new_standards)
        if on_spec_change is not None:
            on_spec_change(project)  # persist before the next round earns a scorecard

    return cards
