"""Golden-output snapshots — catch output changes the pass/fail rubric misses.

`eval` reports pass/fail; `drift` reports the pass-rate moved; neither tells you
the actual *text* of an answer changed while still passing. This is snapshot
testing for AI: pin each test's current output as a golden, then flag when a later
run's output differs — tone shifts, semantic drift, regressions too subtle for the
rubric. Deterministic: reads the saved scorecards, writes a golden file. No engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .identity import hashes_compatible
from .models import Scorecard
from .store import atomic_write_text

GOLDEN_FILE = "golden.json"


@dataclass
class SnapshotDiff:
    changed: list[str]   # test ids whose output text differs from the golden
    added: list[str]     # in the latest run, not in the golden (new tests)
    removed: list[str]   # in the golden, missing from the latest run
    # In both, but no longer the same question: `compile` re-minted the slot, so
    # the pinned answer belongs to text this run never sent. Comparing the two
    # texts reports drift the model never caused — or, for a classifier whose
    # outputs repeat, reports NO drift while the pin sits on a question that no
    # longer exists.
    replaced: list[str] = field(default_factory=list)

    @property
    def drifted(self) -> bool:
        # A changed answer or a vanished test is a regression worth flagging;
        # a brand-new test isn't drift. A REPLACED test is a pin that has
        # stopped checking anything, which must not read as a check that passed.
        return bool(self.changed or self.removed or self.replaced)


def _entry(value: object) -> tuple[str, str | None]:
    """(output, input_hash) from either golden format.

    A golden pinned before the question was recorded is a bare string: its
    output is known and its question is not. Unknown means "unknown", never
    "matches" — the same rule scorecards use — so goldens already on disk keep
    comparing by id exactly as they always did.
    """
    if isinstance(value, dict):
        out, h = value.get("output"), value.get("input_hash")
        return (out if isinstance(out, str) else ""), (h if isinstance(h, str) else None)
    return (value if isinstance(value, str) else ""), None


def outputs_of(card: Scorecard) -> dict[str, dict]:
    """Pinnable outputs, each carrying a fingerprint of the question it answered."""
    return {r.test_id: {"output": r.output, "input_hash": r.input_hash} for r in card.results}


def compare(golden: dict, latest: dict) -> SnapshotDiff:
    g = {k: _entry(v) for k, v in golden.items()}
    now = {k: _entry(v) for k, v in latest.items()}
    shared = g.keys() & now.keys()
    replaced = sorted(t for t in shared if not hashes_compatible(g[t][1], now[t][1]))
    comparable = shared - set(replaced)
    return SnapshotDiff(
        changed=sorted(t for t in comparable if now[t][0] != g[t][0]),
        added=sorted(t for t in now if t not in g),
        removed=sorted(t for t in g if t not in now),
        replaced=replaced,
    )


def save_golden(project_dir: str | Path, golden: dict) -> Path:
    return atomic_write_text(Path(project_dir) / GOLDEN_FILE, json.dumps(golden, indent=2, ensure_ascii=False))


def load_golden(project_dir: str | Path) -> dict | None:
    f = Path(project_dir) / GOLDEN_FILE
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):  # corrupt/hand-edited golden → treat as unpinned, don't traceback
        return None
    return data if isinstance(data, dict) else None  # a non-dict golden is corrupt → unpinned


def snapshot_dict(diff: SnapshotDiff) -> dict:
    return {"drifted": diff.drifted, "changed": diff.changed, "added": diff.added,
            "removed": diff.removed, "replaced": diff.replaced}
