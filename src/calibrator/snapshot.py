"""Golden-output snapshots — catch output changes the pass/fail rubric misses.

`eval` reports pass/fail; `drift` reports the pass-rate moved; neither tells you
the actual *text* of an answer changed while still passing. This is snapshot
testing for AI: pin each test's current output as a golden, then flag when a later
run's output differs — tone shifts, semantic drift, regressions too subtle for the
rubric. Deterministic: reads the saved scorecards, writes a golden file. No engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import Scorecard
from .store import atomic_write_text

GOLDEN_FILE = "golden.json"


@dataclass
class SnapshotDiff:
    changed: list[str]   # test ids whose output text differs from the golden
    added: list[str]     # in the latest run, not in the golden (new tests)
    removed: list[str]   # in the golden, missing from the latest run

    @property
    def drifted(self) -> bool:
        # A changed answer or a vanished test is a regression worth flagging;
        # a brand-new test isn't drift.
        return bool(self.changed or self.removed)


def outputs_of(card: Scorecard) -> dict[str, str]:
    return {r.test_id: r.output for r in card.results}


def compare(golden: dict[str, str], latest: dict[str, str]) -> SnapshotDiff:
    return SnapshotDiff(
        changed=sorted(t for t in golden if t in latest and latest[t] != golden[t]),
        added=sorted(t for t in latest if t not in golden),
        removed=sorted(t for t in golden if t not in latest),
    )


def save_golden(project_dir: str | Path, golden: dict[str, str]) -> Path:
    return atomic_write_text(Path(project_dir) / GOLDEN_FILE, json.dumps(golden, indent=2, ensure_ascii=False))


def load_golden(project_dir: str | Path) -> dict | None:
    f = Path(project_dir) / GOLDEN_FILE
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except (ValueError, OSError):  # corrupt/hand-edited golden → treat as unpinned, don't traceback
        return None
    return data if isinstance(data, dict) else {}


def snapshot_dict(diff: SnapshotDiff) -> dict:
    return {"drifted": diff.drifted, "changed": diff.changed, "added": diff.added, "removed": diff.removed}
