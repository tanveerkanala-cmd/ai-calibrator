"""Judge calibration — measure how much to trust the LLM judge.

The whole promise is a *tested, reliable* AI — but the test is only as good as
the judge doing the grading. `--judge-passes` checks the judge is *consistent*;
this checks it's *correct*: confirm a sample of its verdicts against your own
judgment and measure agreement, overall and per criterion. Low agreement on a
criterion means that criterion is too subjective (reword it, or lean on
deterministic checks) — and tells you how far to trust the scorecard.

This is the §9 mitigation ("calibrate the judge against a small human-labeled
set before trusting it"), made concrete. Deterministic — it reads the judge's
already-saved verdicts and compares them to your labels; no engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import Scorecard
from .store import atomic_write_text

LABELS_FILE = "human-labels.json"


def _labels_path(project_dir: str | Path, run_id: str) -> Path:
    # run_id becomes a path component — same guard as drift.load_scorecard.
    if not isinstance(run_id, str) or not run_id.strip() \
            or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError(f"invalid run id: {run_id!r}")
    return Path(project_dir) / "evals" / run_id / LABELS_FILE


def save_labels(project_dir: str | Path, run_id: str, labels: list[dict]) -> Path:
    """Persist human labels under ``evals/<run>/human-labels.json`` (merged).

    Saving them makes the calibration an *asset*, not a one-off reading: the
    Engine-Trainer uses these as ground-truth rows when localizing the judge
    role, and agreement can be recomputed later. Re-labeling the same
    (test, criterion) replaces the old label."""
    path = _labels_path(project_dir, run_id)
    merged: dict[tuple, dict] = {}
    for label in load_labels(project_dir, run_id) + list(labels):
        # Require an explicit "passed" — otherwise a label missing it would be
        # silently persisted as a FAIL (bool(None)); load_labels enforces the same.
        if isinstance(label, dict) and label.get("test_id") and label.get("criterion_id") and "passed" in label:
            merged[(label["test_id"], label["criterion_id"])] = {
                "test_id": label["test_id"], "criterion_id": label["criterion_id"],
                "passed": bool(label.get("passed")),
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(path, json.dumps({"run_id": run_id, "labels": list(merged.values())},
                                              indent=2, ensure_ascii=False))


def load_labels(project_dir: str | Path, run_id: str) -> list[dict]:
    """Labels saved for one run ([] if none / unreadable — labels are advisory).

    Applies the same required-field filter as ``save_labels`` so a hand-edited
    file can't feed half-formed labels downstream."""
    path = _labels_path(project_dir, run_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    labels = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, list):
        return []
    # Require the verdict too: a label without "passed" carries no judgment, and
    # letting it through would be silently treated as a fail downstream.
    return [x for x in labels
            if isinstance(x, dict) and x.get("test_id") and x.get("criterion_id") and "passed" in x]


def all_labels(project_dir: str | Path) -> list[tuple[str, list[dict]]]:
    """(run_id, labels) for every run that has saved human labels."""
    evals = Path(project_dir) / "evals"
    if not evals.is_dir():
        return []
    out: list[tuple[str, list[dict]]] = []
    for d in sorted(evals.iterdir()):
        if d.is_dir() and (d / LABELS_FILE).exists():
            labels = load_labels(project_dir, d.name)
            if labels:
                out.append((d.name, labels))
    return out


@dataclass
class JudgeAgreement:
    total: int
    agreed: int
    by_criterion: dict[str, tuple[int, int]] = field(default_factory=dict)  # id -> (agreed, total)
    disagreements: list[dict] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    def unreliable_criteria(self, threshold: float = 0.8) -> list[str]:
        return sorted(cid for cid, (a, t) in self.by_criterion.items() if t and a / t < threshold)


def gradings(card: Scorecard) -> list[dict]:
    """Every (test_id, criterion_id, output, judge verdict) in a scorecard — the
    judgments a human can confirm or correct."""
    return [{"test_id": r.test_id, "criterion_id": c.criterion_id, "output": r.output,
             "judge_passed": c.passed, "rationale": c.rationale}
            for r in card.results for c in r.criteria]


def judge_agreement(card: Scorecard, human_labels: list[dict]) -> JudgeAgreement:
    """Compare the judge's saved verdicts to human labels.

    ``human_labels``: ``[{"test_id", "criterion_id", "passed": bool}, ...]``.
    Returns overall + per-criterion agreement and the specific disagreements."""
    judge = {(r.test_id, c.criterion_id): c for r in card.results for c in r.criteria}
    ag = JudgeAgreement(total=0, agreed=0)
    for label in human_labels:
        key = (label.get("test_id"), label.get("criterion_id"))
        cr = judge.get(key)
        if cr is None:
            continue
        human_passed = bool(label.get("passed"))
        match = cr.passed == human_passed
        ag.total += 1
        ag.agreed += int(match)
        a, t = ag.by_criterion.get(key[1], (0, 0))
        ag.by_criterion[key[1]] = (a + int(match), t + 1)
        if not match:
            ag.disagreements.append({"test_id": key[0], "criterion_id": key[1],
                                     "judge": cr.passed, "human": human_passed, "rationale": cr.rationale})
    return ag


def agreement_dict(ag: JudgeAgreement) -> dict:
    return {
        "agreement_rate": round(ag.agreement_rate, 3),
        "agreed": ag.agreed, "total": ag.total,
        "by_criterion": {cid: {"agreed": a, "total": t, "rate": round(a / t, 3)}
                         for cid, (a, t) in ag.by_criterion.items()},
        "unreliable_criteria": ag.unreliable_criteria(),
        "disagreements": ag.disagreements,
    }
