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
- absorbing changes the certification fingerprint (see :func:`calibrator.ci.config_hash`),
  so the gate goes **stale** until `calibrate ci` re-proves the AI against the
  suite that now includes what it just learned.

Use → flag → absorb → re-certify: the AI gets measurably more reliable the more
it's used, with receipts. Deterministic; no engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .coerce import as_opt_str, as_str, is_str
from .models import BehaviorSpec, Example, Project, TestCase
from .store import atomic_write_text, project_lock

FEEDBACK_FILE = "feedback.jsonl"            # under <project>/logs/
ABSORBED_FILE = "feedback-absorbed.jsonl"   # consumed records (audit trail)


def append_feedback(project_dir: str | Path, record: dict) -> None:
    """Durably append one live-feedback record (called by the runtime).

    Takes the project lock: `absorb_feedback` empties the inbox after reading
    it (under the same lock, via the CLI/API), so an unserialized append landing
    in that read→truncate window would be silently DESTROYED. The lock closes
    the window; appends simply wait out an in-flight absorb (milliseconds)."""
    with project_lock(project_dir):
        d = Path(project_dir) / "logs"
        d.mkdir(parents=True, exist_ok=True)
        with (d / FEEDBACK_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_feedback(project_dir: str | Path) -> list[dict]:
    """Pending feedback records, junk-tolerant (malformed lines are skipped)."""
    f = Path(project_dir) / "logs" / FEEDBACK_FILE
    if not f.exists():
        return []
    out: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


@dataclass
class AbsorbResult:
    ups: int = 0
    downs: int = 0
    examples_added: int = 0
    tests_added: int = 0
    skipped: int = 0                       # malformed or duplicate records
    test_ids: list[str] = field(default_factory=list)


def _turns_of(record: dict) -> list[str]:
    turns = record.get("turns")
    if isinstance(turns, list):
        return [t for t in turns if is_str(t) and t.strip()]
    return []


def _fb_id_allocator(tests: list[TestCase]):
    """Yield fb_N ids not yet taken. The taken-set is built ONCE — the previous
    per-record rescan was O(records × tests) and measurably cliffed at scale
    (audit: ~2.7s for 1000 records × 5000 tests; now O(records + tests))."""
    taken = {t.id for t in tests}
    n = 1
    while True:
        while f"fb_{n}" in taken:
            n += 1
        taken.add(f"fb_{n}")
        yield f"fb_{n}"


def absorb_feedback(project: Project, project_dir: str | Path) -> AbsorbResult:
    """Fold all pending feedback into the spec + tests; archive the records.

    Idempotent: consumed records move to ``feedback-absorbed.jsonl`` and
    duplicates (same conversation already pinned / same example already present)
    are skipped, so running twice adds nothing new.

    CONCURRENCY CONTRACT: the caller must hold ``store.project_lock`` (the CLI
    and API do). This function reads the inbox and then EMPTIES it; the lock is
    what stops a concurrent ``append_feedback`` from landing a record in that
    window and having it destroyed by the truncate."""
    records = read_feedback(project_dir)
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
                              why=reason or "approved in live use")
        else:
            result.downs += 1
            example = Example(input=turns[-1], bad_output=output, good_output=correction,
                              why=reason or "flagged in live use")

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

    # Archive: consumed records append to the audit trail; the inbox empties.
    logs = Path(project_dir) / "logs"
    absorbed = logs / ABSORBED_FILE
    prior = absorbed.read_text(encoding="utf-8") if absorbed.exists() else ""
    atomic_write_text(absorbed, prior + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    atomic_write_text(logs / FEEDBACK_FILE, "")
    return result


def absorb_dict(result: AbsorbResult) -> dict:
    return {"ups": result.ups, "downs": result.downs,
            "examples_added": result.examples_added, "tests_added": result.tests_added,
            "skipped": result.skipped, "test_ids": result.test_ids}
