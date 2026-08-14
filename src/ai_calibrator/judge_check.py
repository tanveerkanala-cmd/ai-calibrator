"""Judge calibration — measure how much to trust the LLM judge.

The whole promise is a *tested, reliable* AI — but the test is only as good as
the judge doing the grading. `--judge-passes` checks the judge is *consistent*;
this checks it's *correct*: confirm a sample of its verdicts against your own
judgment and measure agreement, overall and per criterion. Low agreement on a
criterion means that criterion is too subjective (reword it, or lean on
deterministic checks) — and tells you how far to trust the scorecard.

This is the standard mitigation ("calibrate the judge against a small
human-labeled set before trusting it"), made concrete. Deterministic — it
reads the judge's already-saved verdicts and compares them to your labels;
no engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .coerce import as_bool
from .models import BehaviorSpec, Scorecard
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
    (test, criterion) replaces the old label.

    CONCURRENCY CONTRACT: the caller must hold ``store.project_lock``. This
    RELOADS the file and rewrites it whole, so two unserialized writers each
    merge onto a stale read and the loser's labels are gone — silently, since
    the write itself is atomic and every caller still succeeds.

    Raises ValueError if the labels file exists but cannot be read. The merge
    would otherwise start from the empty list ``load_labels`` returns for an
    unreadable file and then rewrite the file whole, destroying every label the
    corrupt bytes still held — the same situation the golden-output paths refuse
    to overwrite, on data that is even more expensive to recreate."""
    path = _labels_path(project_dir, run_id)
    merged: dict[tuple, dict] = {}
    for label in _prior_labels(path) + list(labels):
        # Require an explicit "passed" — otherwise a label missing it would be
        # silently persisted as a FAIL (bool(None)); load_labels enforces the same.
        # Ids must be STRINGS: they become a dict key here, so a list/dict id would
        # raise "unhashable type" and escape the API route as an unhandled 500.
        if (isinstance(label, dict) and isinstance(label.get("test_id"), str)
                and isinstance(label.get("criterion_id"), str)
                and label["test_id"] and label["criterion_id"] and "passed" in label):
            merged[(label["test_id"], label["criterion_id"])] = {
                "test_id": label["test_id"], "criterion_id": label["criterion_id"],
                # as_bool, not bool: an API client or a hand-edited file can carry
                # the verdict as the STRING "false", which bool() reads as PASS.
                "passed": as_bool(label.get("passed")),
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
    return _valid_labels(labels)


def _prior_labels(path: Path) -> list[dict]:
    """The labels already on disk, for a merge that will rewrite the file whole.

    Unlike ``load_labels`` this tells "nothing saved yet" apart from "what is
    saved cannot be read": to a reader the difference is academic (labels are
    advisory), but a writer that treats them the same rewrites the file with only
    the new labels and takes every prior one with it. Half-formed ROWS are not
    corruption — they are filtered, exactly as the reader filters them."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(
            f"{path} is unreadable ({exc}) — refusing to overwrite it. Saving merges onto "
            "the labels already there, so writing now would replace every one of them with "
            "just this session's. Repair the file (a merge-conflict marker or a truncated "
            "write is the usual cause), or move it aside, then re-run."
        ) from exc
    labels = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, list):
        raise ValueError(
            f"{path} is unreadable — refusing to overwrite it. It parses, but carries no "
            "`labels` list, so whatever labels it was meant to hold cannot be merged "
            "forward. Repair the file, or move it aside, then re-run."
        )
    return _valid_labels(labels)


def _valid_labels(labels: list) -> list[dict]:
    # Require the verdict too: a label without "passed" carries no judgment, and
    # letting it through would be silently treated as a fail downstream.
    # isinstance, not truthiness: a list or dict id is truthy and reaches the
    # consumers as an unhashable dictionary key, which aborts a whole export with
    # a TypeError. save_labels writes strings; a hand-edited file may not.
    return [x for x in labels
            if isinstance(x, dict) and isinstance(x.get("test_id"), str) and x["test_id"]
            and isinstance(x.get("criterion_id"), str) and x["criterion_id"] and "passed" in x]


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
    # Labels there was nothing to compare against (see judge_agreement). They are
    # neither agreements nor disagreements, so they stay out of both counts — but
    # they must be reported, or the rate reads as covering every label given.
    unmatched: int = 0

    @property
    def agreement_rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    def unreliable_criteria(self, threshold: float = 0.8) -> list[str]:
        return sorted(cid for cid, (a, t) in self.by_criterion.items() if t and a / t < threshold)


# Rationales minted by run_eval's own short-circuits rather than by the judge.
# A verdict carrying one was decided by code, so it is not the judge's to defend.
_NOT_JUDGE_RATIONALES = frozenset({"empty output"})


def gradings(card: Scorecard, spec: "BehaviorSpec | None" = None) -> list[dict]:
    """The judge's verdicts in a scorecard — the judgments a human can confirm.

    ONLY the judge's. A criterion carrying a deterministic ``check`` is graded by
    ``run_check``, and an empty answer is failed by the harness before any judge
    call, so neither is a judgment the judge made. Presenting them here asked the
    owner to calibrate a grader that never ran and then reported the result as
    "judge agreement" — on a project whose criteria all carry checks, a 100%
    agreement rate could be printed for a judge that was never invoked.

    ``spec`` supplies which criteria are judge-graded. Omitting it keeps the old
    permissive behaviour for callers that have no spec to hand; every shipped
    caller passes one."""
    code_graded = {c.id for c in spec.eval_criteria if c.check is not None} if spec else set()
    return [{"test_id": r.test_id, "criterion_id": c.criterion_id, "output": r.output,
             "judge_passed": c.passed, "rationale": c.rationale}
            for r in card.results for c in r.criteria
            if c.criterion_id not in code_graded
            and (c.rationale or "") not in _NOT_JUDGE_RATIONALES]


def judge_agreement(card: Scorecard, human_labels: list[dict]) -> JudgeAgreement:
    """Compare the judge's saved verdicts to human labels.

    ``human_labels``: ``[{"test_id", "criterion_id", "passed": bool}, ...]``.
    Returns overall + per-criterion agreement and the specific disagreements."""
    judge = {(r.test_id, c.criterion_id): c for r in card.results for c in r.criteria}
    ag = JudgeAgreement(total=0, agreed=0)
    for label in human_labels:
        if not (isinstance(label, dict) and isinstance(label.get("test_id"), str)
                and isinstance(label.get("criterion_id"), str)):
            ag.unmatched += 1  # same scalar-id contract as save_labels — never crash on a bad row
            continue
        key = (label["test_id"], label["criterion_id"])
        cr = judge.get(key)
        if cr is None:
            # The judge never ruled on this (test, criterion), so there is no
            # verdict to agree with. Counting it either way would make the rate
            # describe coverage it does not have; report it separately instead.
            ag.unmatched += 1
            continue
        human_passed = as_bool(label.get("passed"))
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
        "agreed": ag.agreed, "total": ag.total, "unmatched": ag.unmatched,
        "by_criterion": {cid: {"agreed": a, "total": t, "rate": round(a / t, 3) if t else 0.0}
                         for cid, (a, t) in ag.by_criterion.items()},
        "unreliable_criteria": ag.unreliable_criteria(),
        "disagreements": ag.disagreements,
    }
