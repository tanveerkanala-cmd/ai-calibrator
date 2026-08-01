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
from typing import Callable, Optional

from .checks import run_check_turns
from .coerce import as_bool, as_list, as_opt_str, as_str, is_str
from .compile import render_system_prompt
from .engines.base import Engine, require_object
from .models import CriterionResult, Project, Scorecard, TestResult, test_input_hash
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

JUDGE_SYSTEM = (
    "You are a strict grader. Judge the AI output against each listed criterion "
    "independently. Pass a criterion only if the output clearly satisfies it. "
    "Give a 0-1 score and a one-line rationale per criterion. Respond with JSON "
    "only, matching the schema."
)
_JUDGE_SYSTEM = JUDGE_SYSTEM  # back-compat alias


def judge_system(instructions: str | None = None) -> str:
    """The judge's system message: how to grade, plus what the AI was told.

    Without the instructions a judge cannot grade any criterion that refers to
    them — "cites only policies stated in the spec" is the natural way to phrase
    "don't invent", and the compiler writes criteria like it. Asked to check
    output against rules it was never shown, a judge does not abstain: in a live
    run it failed answers for stating policy the materials really did contain,
    with confident, specific rationales and a 0% headline. Grading an answer
    against criteria without the instructions is marking an exam without the
    syllabus.

    It goes in the SYSTEM message, not the per-test prompt, because it is
    identical for every test in a run — so Anthropic's prompt cache (applied to
    `system` in the adapter) charges for it roughly once rather than once per
    test, which matters for the highest-volume role in the tool.

    Public for the same reason ``judge_prompt`` is: the Engine-Trainer rebuilds
    ground-truth rows and must reproduce byte-for-byte what the judge saw."""
    rules = (instructions or "").strip()
    if not rules:
        return JUDGE_SYSTEM
    return (
        f"{JUDGE_SYSTEM}\n\n"
        "The AI under test was given the instructions below. Treat them as the "
        "authority for what it was told to do: a fact or figure stated there is "
        "NOT invented, and a rule stated there is the rule to grade against. "
        "They are context, not criteria — grade only the criteria you are given.\n"
        "--- INSTRUCTIONS GIVEN TO THE AI ---\n"
        f"{rules}\n"
        "--- END INSTRUCTIONS ---"
    )


def judge_prompt(test_input: str, output: str, criteria: list[tuple[str, str]]) -> str:
    """The exact prompt the judge grades with.

    Public because the Engine-Trainer builds *ground-truth* training rows from
    human judge-check labels — those must use the identical format the judge role
    sees at inference time, or the fine-tune learns the wrong distribution."""
    block = "\n".join(f"- {cid}: {desc}" for cid, desc in criteria)
    return (
        f"INPUT:\n{test_input}\n\nAI OUTPUT:\n{output}\n\nCRITERIA:\n{block}\n\n"
        "Grade each criterion."
    )


def _as_float(value: object) -> float:
    """Coerce a model-supplied score to a float in [0, 1], defaulting to 0.0.

    A non-compliant judge could return a non-numeric, non-finite, or out-of-range
    score; none of those may raise or poison the scorecard arithmetic. Clamping
    matters as much as the finite check: a judge answering on a 0-100 or 1-5 scale
    (routine for the small local models this tool supports) would otherwise push
    the weighted mean above 1.0, and `pct` renders anything >= 1 as the reassuring
    ">99%" — an anomaly hidden instead of surfaced, right next to a 0% pass rate."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return min(1.0, max(0.0, f))


def _judge(
    judge: Engine,
    test_input: str,
    output: str,
    criteria: list[tuple[str, str]],
    instructions: str | None = None,
) -> list[CriterionResult]:
    prompt = judge_prompt(test_input, output, criteria)
    out = require_object(
        judge.complete(prompt, system=judge_system(instructions), schema=JUDGE_SCHEMA), "judge")
    # Key only on string ids. A non-compliant judge can return criterion_id as a
    # list or dict, and an unhashable key raises TypeError here — destroying a
    # whole graded run over one bad row. Dropping the row instead leaves that
    # criterion ungraded, which the loop below already records as a fail.
    by_id = {
        r["criterion_id"]: r
        for r in as_list(out.get("results"))
        if isinstance(r, dict) and is_str(r.get("criterion_id"))
    }
    results = []
    for cid, _ in criteria:
        r = by_id.get(cid, {})
        results.append(
            CriterionResult(
                criterion_id=cid,
                passed=as_bool(r.get("passed", False)),
                score=_as_float(r.get("score", 0.0)),
                # Never leave a failing verdict with a null rationale — the failure
                # list must read consistently, so substitute a clear placeholder.
                rationale=as_opt_str(r.get("rationale")) or "judge gave no rationale",
            )
        )
    return results


def _judge_consensus(judge: Engine, test_input: str, output: str,
                     criteria: list[tuple[str, str]], passes: int,
                     instructions: str | None = None) -> list[CriterionResult]:
    """Grade with ``passes`` independent judge calls, majority-vote each criterion,
    and record agreement as ``confidence`` — self-consistency over a noisy judge."""
    runs = [_judge(judge, test_input, output, criteria, instructions) for _ in range(passes)]
    results: list[CriterionResult] = []
    for idx, (cid, _) in enumerate(criteria):
        verdicts = [run[idx].passed for run in runs]
        yes = sum(verdicts)
        passed = yes * 2 > passes  # strict majority
        agreement = max(yes, passes - yes) / passes
        rationale = next((run[idx].rationale for run in runs if run[idx].passed == passed), None)
        results.append(CriterionResult(
            criterion_id=cid, passed=passed,
            score=round(sum(r[idx].score for r in runs) / passes, 3),
            rationale=rationale or "judge gave no rationale", confidence=round(agreement, 3),
        ))
    return results


def conversation_prompt(history_lines: list[str], user_turn: str) -> str:
    """The exact transcript-encoded prompt for the next turn of a conversation.

    Public because the runtime (`calibrate run`) must encode live chats the SAME
    way the eval harness encodes multi-turn tests — what you tested is what you
    serve. ``history_lines`` alternate ``User: …`` / ``Assistant: …``."""
    history = "\n".join(history_lines)
    return (history + "\n" if history else "") + f"User: {user_turn}\nAssistant:"


def _conversation_output(subject: Engine, system: str | None, user_turns: list[str],
                         project_dir: str | Path | None = None) -> tuple[str, list[str]]:
    """Run a multi-turn conversation; return ``(transcript, replies)``.

    Two different views, for two different consumers. The **transcript** (the
    alternating ``User:`` / ``Assistant:`` lines) is what gets recorded on the
    scorecard and shown to the judge, which needs the questions to grade the
    replies in context. The **replies** (the assistant turns, kept apart) are
    what the deterministic checks grade: a check asks what the AI *said*, so it
    must never see the user's words — otherwise ``not_contains "cure"`` fails
    because the user asked about a cure, and ``max_chars`` bills the AI for the
    question — and each reply is a whole answer, so a per-answer check must not
    be billed for the other turns either.

    History is encoded into each prompt (works across every engine without a
    messages-based interface). The base system prompt is constant; when a
    knowledge index exists, each turn's system is augmented with chunks retrieved
    for THAT turn (identical to how the runtime serves — what you test is what
    you serve)."""
    from . import rag
    lines: list[str] = []
    replies: list[str] = []
    for turn in user_turns:
        eff_system = rag.augment_system(system, project_dir, turn)
        prompt = conversation_prompt(lines, turn)
        reply = as_str(subject.complete(prompt, system=eff_system)).strip()
        replies.append(reply)
        lines.append(f"User: {turn}")
        lines.append(f"Assistant: {reply}")
    return "\n".join(lines), replies


# Called before each test runs: (done, total, test_id). Lets the CLI show live
# progress so a multi-minute run isn't indistinguishable from a hang.
ProgressFn = Callable[[int, int, str], None]


class EvalInterrupted(KeyboardInterrupt):
    """Raised when an eval is Ctrl-C'd mid-run. ``partial`` holds the scorecard
    for the tests that finished, so the caller can save progress rather than
    discard every graded result."""

    def __init__(self, partial: "Scorecard") -> None:
        super().__init__()
        self.partial = partial


def run_eval(
    project: Project,
    subject: Engine,
    judge: Engine,
    *,
    run_id: str = "run-0001",
    judge_passes: int = 1,
    project_dir: str | Path | None = None,
    max_tests: int | None = None,
    on_progress: Optional[ProgressFn] = None,
) -> Scorecard:
    """Run each test on the subject and grade the output against its criteria.

    ``judge_passes > 1`` grades each criterion with that many independent judge
    calls and majority-votes (self-consistency), recording per-criterion
    confidence so split verdicts can be surfaced for human review.

    ``project_dir`` enables **RAG retrieval**: when the project has a knowledge
    index, each test's subject prompt is augmented with chunks retrieved for that
    input — identical to what `calibrate run` serves — so the scorecard reflects
    the AI you actually deploy. Omit it (default) to grade the prompt-only AI.

    ``max_tests`` runs only the first N tests (a smoke check on a slow model).
    ``on_progress(done, total, test_id)`` fires before each test so a caller can
    show live progress. A ``KeyboardInterrupt`` mid-run propagates with the
    results gathered so far attached as ``.partial_results`` so the caller can
    still save what completed."""
    if not isinstance(judge_passes, int) or judge_passes < 1:
        raise ValueError(f"judge_passes must be an integer >= 1 (got {judge_passes!r})")
    spec = project.spec
    if spec is None:
        raise ValueError("No behavior spec — run `calibrate compile` first.")
    from . import rag
    system = render_system_prompt(spec)
    crit_by_id = {c.id: c for c in spec.eval_criteria}

    tests = project.tests if max_tests is None else project.tests[:max_tests]
    total = len(tests)
    results: list[TestResult] = []

    def _card(partial: bool) -> Scorecard:
        from datetime import datetime, timezone

        from . import __version__
        return Scorecard(
            run_id=run_id, results=results, subject=subject.name, judge=judge.name,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tool_version=__version__,
            partial=partial or (max_tests is not None and max_tests < len(project.tests)),
        )

    try:
        for _i, test in enumerate(tests, start=1):
            if on_progress is not None:
                try:
                    on_progress(_i, total, test.id)
                except KeyboardInterrupt:
                    raise
                except Exception:  # a progress printer must never break the eval
                    pass
            # Coerce defensively: a misbehaving subject can return a non-string
            # (despite the str contract); as_str makes that an empty output (caught
            # by the guard below) instead of an AttributeError on .strip().
            turns = [test.input] + [f for f in test.follow_ups if is_str(f)]
            if len(turns) > 1:  # multi-turn conversation test
                # `output` is the transcript (recorded + judged); `replies` are the
                # assistant's words alone, one per turn, which is what the checks grade.
                output, replies = _conversation_output(subject, system, turns, project_dir)
            else:
                eff_system = rag.augment_system(system, project_dir, test.input)  # RAG when indexed
                # Encode the single turn exactly as the runtime and the API's /try
                # do (`conversation_prompt`), so the certified pass rate is earned
                # on the prompt the deployed endpoint actually sends.
                output = as_str(subject.complete(conversation_prompt([], test.input), system=eff_system))
                replies = [output]
            # De-dup while preserving order: a duplicated id in `expects` (hand-edited
            # YAML, or an engine that repeats one) would otherwise append the same
            # CriterionResult multiple times, multiplying that criterion's weight in
            # the weighted score. Each criterion counts once.
            _seen: set[str] = set()
            expected: list[str] = []
            for cid in (test.expects or list(crit_by_id)):
                if cid in crit_by_id and cid not in _seen:
                    _seen.add(cid)
                    expected.append(cid)
            graded: dict[str, CriterionResult] = {}

            # An answer that says NOTHING fails everything, before any grading layer
            # runs. This has to come first: the negative-form checks (not_contains,
            # max_chars) are all trivially satisfied by "", so a subject that returned
            # nothing would otherwise score 1.0 on a test whose criteria are entirely
            # deterministic — a certified, green gate in front of a silent AI.
            if not any(reply.strip() for reply in replies):
                for cid in expected:
                    graded[cid] = CriterionResult(criterion_id=cid, passed=False, score=0.0,
                                                  rationale="empty output")
            else:
                # First grading layer — criteria with a deterministic check are graded
                # exactly by code (no judge), against the AI's words only.
                for cid in expected:
                    chk = crit_by_id[cid].check
                    if chk is not None:
                        passed, why = run_check_turns(chk, replies)
                        graded[cid] = CriterionResult(criterion_id=cid, passed=passed,
                                                      score=1.0 if passed else 0.0, rationale=why)

                # Remaining criteria go to the LLM judge, which sees the full transcript.
                judged = [(cid, crit_by_id[cid].description) for cid in expected
                          if crit_by_id[cid].check is None]
                if judged and judge_passes > 1:
                    for cr in _judge_consensus(judge, test.input, output, judged,
                                              judge_passes, system):
                        graded[cr.criterion_id] = cr
                elif judged:
                    for cr in _judge(judge, test.input, output, judged, system):
                        graded[cr.criterion_id] = cr

            # Reassemble in requested order — test.expects is an ordered list, so the
            # results must follow it; the checked/judged split is an internal detail.
            # Stamp each verdict with the weight it was graded under, so the scorecard
            # stays honest even if the spec's weights change later.
            crs: list[CriterionResult] = []
            for cid in expected:
                if cid in graded:
                    graded[cid].weight = crit_by_id[cid].weight
                    crs.append(graded[cid])
            results.append(TestResult(test_id=test.id, output=output, criteria=crs,
                                      input_hash=test_input_hash(test)))
    except KeyboardInterrupt:
        # Ctrl-C mid-run: surface what completed so the caller can still
        # save a partial scorecard instead of losing every graded test.
        raise EvalInterrupted(_card(partial=True))

    return _card(partial=False)


def low_confidence_results(card: Scorecard, *, threshold: float = 0.67) -> list[tuple[str, CriterionResult]]:
    """(test_id, criterion) pairs where the judge was split below ``threshold`` —
    the verdicts worth a human spot-check. Empty unless graded with judge_passes>1."""
    return [(r.test_id, c) for r in card.results for c in r.criteria
            if c.confidence is not None and c.confidence < threshold]


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


def latest_run_id(project_dir: str | Path, *, full_only: bool = False) -> str | None:
    """The most recent ``run-NNNN`` that has a saved scorecard, or None.

    Returns the newest run whose scorecard file EXISTS — it does not validate the
    contents. A corrupt/truncated scorecard is surfaced honestly by the caller
    ("Could not read scorecard <id>" on the CLI, a 409 on the API) rather than
    silently skipped, so the user learns their file is broken instead of seeing a
    misleading "no scorecard yet".

    ``full_only`` skips PARTIAL scorecards (an interrupted run, or ``--max-tests``).
    Use it wherever the run is a *reference point* rather than the latest news —
    a regression baseline compares pass rates over two test sets, so a smoke run
    silently becoming the baseline hides every regression it never ran."""
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
                if k <= n:
                    continue
                if full_only:
                    # Read the flag directly (drift.load_scorecard would be a cycle).
                    # A scorecard we can't read isn't provably full, so it can't serve
                    # as a reference point either — skip it and let the caller's own
                    # "could not read" path report a broken file.
                    try:
                        raw = json.loads((d / "scorecard.json").read_text(encoding="utf-8"))
                        if not isinstance(raw, dict) or raw.get("partial"):
                            continue
                    except (OSError, ValueError):
                        continue
                n, best = k, d.name
    return best


def save_scorecard(project_dir: str | Path, card: Scorecard) -> Path:
    """Write scorecard.json + failures.jsonl under <project>/evals/<run-id>/."""
    d = Path(project_dir) / "evals" / card.run_id
    atomic_write_text(d / "scorecard.json", json.dumps(card.model_dump(mode="json"), indent=2))
    # Only a GRADED result can be a failure. `TestResult.passed` is also False for
    # a test nothing graded, so an ungraded result would land here indistinguishable
    # from a real failure — and contradict the pass rate the same run reports, which
    # leaves ungraded tests out of its denominator for exactly this reason.
    fails = [r for r in card.results if r.criteria and not r.passed]
    atomic_write_text(d / "failures.jsonl",
                      "".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in fails))
    return d
