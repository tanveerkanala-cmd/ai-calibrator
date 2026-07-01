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

from dataclasses import dataclass, field

from .models import Scorecard


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
