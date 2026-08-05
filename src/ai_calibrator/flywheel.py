"""The flywheel — live feedback becomes calibration, permanently.

`calibrate run` records thumbs-up / thumbs-down from real traffic
(``POST /v1/feedback`` → ``logs/feedback.jsonl``). `calibrate absorb` folds each
one into the project:

- the exchange becomes a spec **example** (up → ``good_output``; down →
  ``bad_output``, with the human ``correction`` as ``good_output`` when given) —
  the same asset `teach` produces, so it also feeds fine-tuning datasets;
- the conversation becomes a **pinned regression test** (multi-turn feedback
  keeps its follow-ups), so the exact exchange someone flagged can never
  silently regress;
- absorbing a new TEST changes the certification fingerprint (see
  :func:`calibrator.ci.config_hash`), so the gate goes **stale** until
  `calibrate ci` re-proves the AI against the suite that now includes what it
  just learned. An examples-only absorb does NOT: the fingerprint covers the
  rendered system prompt, bindings, criteria, tests and the RAG index, not
  ``spec.examples``, so the gate keeps meaning what it meant.

Use → flag → absorb → re-certify: the AI gets measurably more reliable the more
it's used, with receipts. Deterministic; no engine.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .coerce import as_opt_str, as_str, is_str
from .models import BehaviorSpec, Example, Project, TestCase
from .store import atomic_write_text, project_lock

FEEDBACK_FILE = "feedback.jsonl"            # under <project>/logs/
ABSORBED_FILE = "feedback-absorbed.jsonl"   # consumed records (audit trail)


@contextmanager
def _feedback_lock(project_dir: str | Path, wait_seconds: float | None):
    """The project lock, held indefinitely (CLI) or up to ``wait_seconds`` (HTTP)."""
    if wait_seconds is None:
        with project_lock(project_dir):
            yield
        return

    import time

    from .locking import LockBusy
    lock = project_lock(project_dir, blocking=False)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            lock.acquire()
            break
        except LockBusy:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.release()


def append_feedback(project_dir: str | Path, record: dict, *,
                    wait_seconds: float | None = None) -> None:
    """Durably append one live-feedback record (called by the runtime).

    Takes the project lock: `absorb_feedback` empties the inbox after reading
    it (under the same lock, via the CLI/API), so an unserialized append landing
    in that read→truncate window would be silently DESTROYED. The lock closes
    the window.

    ``wait_seconds`` bounds that wait, and every HTTP caller passes it. The claim
    that an append "waits out an in-flight absorb (milliseconds)" was wrong: the
    same lock is held across whole engine runs by `calibrate eval` / `calibrate
    ci`, so the wait is minutes — and since the routes run in a threadpool, every
    waiter holds one of the worker slots the whole process shares. Park enough of
    them and the server stops answering anything, including the served AI itself.
    A bounded wait raises :class:`~.locking.LockBusy`, which the routes turn into
    the same 423 every other busy-project route already returns."""
    from .store import open_private_append
    with _feedback_lock(project_dir, wait_seconds):
        d = Path(project_dir) / "logs"
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open_private_append(d / FEEDBACK_FILE) as fh:  # 0600 — holds user queries
            # A lone surrogate anywhere in the record would raise UnicodeEncodeError
            # on the UTF-8 handle, losing a human's verdict to a bare 500. Scrub
            # rather than reject: the feedback matters more than the exact bytes.
            line = json.dumps(record, ensure_ascii=False)
            fh.write(line.encode("utf-8", "replace").decode("utf-8") + "\n")


def read_feedback(project_dir: str | Path) -> list[dict]:
    """Pending feedback records, junk-tolerant (malformed lines are skipped)."""
    return read_feedback_lines(project_dir)[0]


def read_feedback_lines(project_dir: str | Path) -> tuple[list[dict], list[str]]:
    """``(records, unparsed_lines)`` from the inbox.

    The unparsed half matters because the inbox is TRUNCATED after absorb: a line
    that cannot be parsed is not absorbed, so truncating it away destroys it —
    silently, and it was the only copy of something a human took the trouble to
    send. A partial write or an encoding accident is exactly how one arises."""
    f = Path(project_dir) / "logs" / FEEDBACK_FILE
    if not f.exists():
        return [], []
    out: list[dict] = []
    junk: list[str] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            junk.append(line)
            continue
        if isinstance(obj, dict):
            out.append(obj)
        else:
            junk.append(line)
    return out, junk


@dataclass
class AbsorbResult:
    ups: int = 0
    downs: int = 0
    examples_added: int = 0
    tests_added: int = 0
    skipped: int = 0                       # malformed or duplicate records
    superseded: int = 0                    # earlier examples retracted by a later verdict
    unparsed: int = 0                      # inbox lines left in place because nothing could read them
    test_ids: list[str] = field(default_factory=list)


def _turns_of(record: dict) -> list[str]:
    turns = record.get("turns")
    if isinstance(turns, list):
        return [t for t in turns if is_str(t) and t.strip()]
    return []


def _fb_id_allocator(tests: list[TestCase]):
    """Yield fb_N ids not yet taken. The taken-set is built ONCE — a per-record
    rescan is O(records × tests) and measurably slow at scale (~2.7s for 1000
    records × 5000 tests); this stays O(records + tests)."""
    taken = {t.id for t in tests}
    n = 1
    while True:
        while f"fb_{n}" in taken:
            n += 1
        taken.add(f"fb_{n}")
        yield f"fb_{n}"


def absorb_feedback(project: Project, project_dir: str | Path, *,
                    commit: Callable[[], object] | None = None) -> AbsorbResult:
    """Fold all pending feedback into the spec + tests; archive the records.

    Idempotent: consumed records move to ``feedback-absorbed.jsonl`` and
    duplicates (same conversation already pinned / same example already present)
    are skipped, so running twice adds nothing new.

    CONCURRENCY CONTRACT: the caller must hold ``store.project_lock`` (the CLI
    and API do). This function reads the inbox and then EMPTIES it; the lock is
    what stops a concurrent ``append_feedback`` from landing a record in that
    window and having it destroyed by the truncate.

    DURABILITY CONTRACT: the inbox is the only copy of a live verdict a human
    took the trouble to give. Pass ``commit`` — the caller's project save — and it
    runs while the records are still pending, so a save that fails (disk full,
    permissions, a crash) leaves them there to absorb again instead of draining
    them into a project that was never written. Re-absorbing costs nothing;
    losing a flagged exchange the flywheel promised to pin forever does."""
    records, unparsed = read_feedback_lines(project_dir)
    result = AbsorbResult()
    if not records:
        return result
    if project.spec is None:  # judgment-first bootstrap, same as teach
        project.spec = BehaviorSpec(goal=project.goal, task_type=project.task_type)
    spec = project.spec

    existing_examples = {(e.input, e.good_output, e.bad_output) for e in spec.examples}
    existing_tests = {(t.input, tuple(t.follow_ups)) for t in project.tests}
    fb_ids = _fb_id_allocator(project.tests)

    for r in records:
        turns, output = _turns_of(r), as_str(r.get("output"))
        verdict = r.get("verdict")
        if not turns or not output.strip() or verdict not in ("up", "down"):
            result.skipped += 1
            continue
        correction = as_opt_str(r.get("correction"))
        reason = as_opt_str(r.get("reason"))
        if verdict == "up":
            result.ups += 1
            example = Example(input=turns[-1], good_output=output,
                              why=reason or "approved in live use",
                              source="human_ratified")  # a human approved this answer
        else:
            result.downs += 1
            example = Example(input=turns[-1], bad_output=output, good_output=correction,
                              why=reason or "flagged in live use",
                              source="human_ratified")  # a human flagged it and may have corrected it

        # Feedback is time-ordered, so the LATEST verdict on an answer wins. Without
        # this, a `down` on text an earlier `up` stored as good_output just appends a
        # contradiction: the spec asserts the same text is both good and bad, and the
        # fine-tune dataset (which filters on good_output alone) keeps training
        # toward the answer a human rejected.
        retracted = [
            ex for ex in spec.examples
            if ex.input == example.input
            and ((verdict == "down" and ex.good_output is not None and ex.good_output == output)
                 or (verdict == "up" and ex.bad_output is not None and ex.bad_output == output))
        ]
        for ex in retracted:
            spec.examples.remove(ex)
            existing_examples.discard((ex.input, ex.good_output, ex.bad_output))
            result.superseded += 1

        ekey = (example.input, example.good_output, example.bad_output)
        if ekey not in existing_examples:
            existing_examples.add(ekey)
            spec.examples.append(example)
            result.examples_added += 1

        tkey = (turns[0], tuple(turns[1:]))
        if tkey not in existing_tests:
            existing_tests.add(tkey)
            tid = next(fb_ids)
            project.tests.append(TestCase(id=tid, input=turns[0], follow_ups=turns[1:],
                                          expects=[], notes=f"from live feedback ({verdict})"))
            result.tests_added += 1
            result.test_ids.append(tid)

    # Persist what was folded BEFORE the records stop existing: nothing reads
    # feedback-absorbed.jsonl back, so a failure after the truncate is final.
    if commit is not None:
        commit()

    # Archive: consumed records append to the audit trail; the inbox empties.
    logs = Path(project_dir) / "logs"
    absorbed = logs / ABSORBED_FILE
    prior = absorbed.read_text(encoding="utf-8") if absorbed.exists() else ""
    atomic_write_text(absorbed, prior + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    # Keep what could not be parsed. Truncating it away would destroy the only
    # copy of a record nothing consumed — and a later release that can read it
    # (or a human eye) still can.
    atomic_write_text(logs / FEEDBACK_FILE, "".join(line + "\n" for line in unparsed))
    result.unparsed = len(unparsed)
    return result


def absorb_dict(result: AbsorbResult) -> dict:
    return {"ups": result.ups, "downs": result.downs,
            "examples_added": result.examples_added, "tests_added": result.tests_added,
            "skipped": result.skipped, "superseded": result.superseded,
            "unparsed": result.unparsed, "test_ids": result.test_ids}
