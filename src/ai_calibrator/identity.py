"""What makes two graded results the same question.

`compile` mints test ids positionally (t1, t2, …) and regenerates the whole t*
range every time it runs, so one id routinely names different content over time.
Every comparison between a saved run and either the current suite or another
saved run therefore has to ask what was ASKED, not just which slot asked it.
Otherwise the ordinary workflow — compile -> eval -> answer more questions ->
compile -> ci — quietly compares two different exams and reports the difference
as a result: a real regression hides because the question that caught it is
gone, and a deleted failure reads as a fix.

``TestResult.input_hash`` records the content. It is None on scorecards written
before the field existed, and None means "unknown", never "matches": those
results keep matching by id alone, so scorecards already on disk report exactly
as they always did, and every run from here on gets the stricter check.

This module is the single place that rule lives. `report`, `drift`, `ci`,
`finetune`, `train_engine` and `snapshot` all defer to it, because a fix that
teaches one surface to say "not comparable" while another still says "ACCEPT"
leaves the user with two answers and no way to tell which one is honest.
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import Scorecard, TestCase, TestResult, content_hash, test_input_hash

__all__ = [
    "content_hash",
    "hash_by_id",
    "hashes_compatible",
    "partition_shared",
    "recompiled_between",
    "result_matches_test",
    "test_input_hash",
]


def hashes_compatible(a: str | None, b: str | None) -> bool:
    """Whether two recorded content hashes can name the same question.

    Unknown (None — written before the field existed) is compatible with
    anything, which is what keeps pre-existing scorecards comparable. Two
    hashes that are both KNOWN and differ are two different questions, and no
    caller may score one against the other.
    """
    return a is None or b is None or a == b


def result_matches_test(result: TestResult, test: TestCase) -> bool:
    """Whether ``result`` is a verdict on ``test`` as the suite defines it NOW.

    Same slot AND same question. Matching on the id alone hands an old run's
    verdicts to tests that have never been executed.
    """
    if result.test_id != test.id:
        return False
    return hashes_compatible(result.input_hash, test_input_hash(test))


def hash_by_id(results: Iterable[TestResult]) -> dict[str, str | None]:
    """Recorded content hash per test id. Last write wins, matching how every
    caller already collapses a scorecard to a dict (duplicate ids are a
    separate defect `lint` reports)."""
    return {r.test_id: r.input_hash for r in results}


def partition_shared(
    baseline: Iterable[TestResult], candidate: Iterable[TestResult]
) -> tuple[set[str], list[str]]:
    """Split the ids two runs share into (comparable, incomparable).

    Comparable ids asked the same question in both runs and may be scored
    against each other. Incomparable ids are present in both runs under the
    same id but asked DIFFERENT questions — the suite was recompiled between
    them. Those are neither regressions nor fixes, and a caller that silently
    drops them turns "we stopped checking" into "we checked and it passed".
    """
    before, after = hash_by_id(baseline), hash_by_id(candidate)
    shared = before.keys() & after.keys()
    comparable = {t for t in shared if hashes_compatible(before[t], after[t])}
    return comparable, sorted(shared - comparable)


def recompiled_between(baseline: Scorecard, candidate: Scorecard) -> list[str]:
    """Ids the two scorecards share that no longer ask the same question."""
    return partition_shared(baseline.results, candidate.results)[1]


def restrict_to_comparable(
    baseline: Scorecard, candidate: Scorecard, ids: set[str]
) -> tuple[set[str], list[str]]:
    """Narrow an already-chosen id set to the ids that asked the same question.

    For callers that select ids on their own terms first (graded-only,
    held-out-of-training) and then need the content check applied to whatever
    survived. Returns (still comparable, excluded because the question changed).
    """
    comparable, changed = partition_shared(baseline.results, candidate.results)
    return ids & comparable, sorted(ids & set(changed))
