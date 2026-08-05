"""Golden-output snapshots — pin outputs and detect text changes."""

from ai_calibrator.models import Scorecard
from ai_calibrator.models import TestResult as Result  # aliased: avoids pytest collecting the model
from ai_calibrator.snapshot import compare, load_golden, outputs_of, save_golden


def _card(run_id, outs):
    return Scorecard(run_id=run_id, results=[Result(test_id=t, output=o) for t, o in outs.items()])


def test_outputs_of():
    # Each pin carries the question it answered, not just the answer: a test id
    # is a slot `compile` re-mints, so the output alone cannot say what was asked.
    assert outputs_of(_card("r", {"t1": "x", "t2": "y"})) == {
        "t1": {"output": "x", "input_hash": None},
        "t2": {"output": "y", "input_hash": None},
    }


def test_golden_distinguishes_a_replaced_test_from_a_removed_one():
    """`compile` re-minted t1 onto a different question. The pinned answer
    belongs to text this run never sent, so the pin has stopped checking
    anything — that is neither a clean match nor a `removed` test."""
    golden = {"t1": {"output": "Our policy is 30 days.", "input_hash": "aaaa000000000000"}}
    latest = {"t1": {"output": "Our policy is 30 days.", "input_hash": "bbbb111111111111"}}
    d = compare(golden, latest)
    # Identical text, so the old comparison saw no drift at all and passed.
    assert d.changed == [] and d.removed == [] and d.added == []
    assert d.replaced == ["t1"]
    assert d.drifted is True


def test_compare_still_matches_by_id_when_either_hash_is_none():
    """A golden pinned before the question was recorded is a bare string, and a
    result from a pre-hash scorecard records None. Unknown never blocks: these
    keep comparing by id exactly as they always did."""
    assert compare({"t1": "same"}, {"t1": {"output": "same", "input_hash": "aaaa000000000000"}}).drifted is False
    assert compare({"t1": {"output": "same", "input_hash": "aaaa000000000000"}}, {"t1": "same"}).drifted is False
    assert compare({"t1": "same"}, {"t1": "changed"}).changed == ["t1"]


def test_compare_detects_changed_added_removed():
    d = compare({"t1": "hello", "t2": "world", "t3": "gone"},
                {"t1": "hello", "t2": "WORLD", "t4": "new"})
    assert d.changed == ["t2"] and d.added == ["t4"] and d.removed == ["t3"]
    assert d.drifted is True


def test_no_drift_when_identical():
    d = compare({"t1": "a"}, {"t1": "a"})
    assert not d.drifted and not d.changed and not d.added and not d.removed


def test_added_only_is_not_drift():
    d = compare({"t1": "a"}, {"t1": "a", "t2": "b"})
    assert d.added == ["t2"] and not d.drifted   # a new test isn't a regression


def test_save_load_golden_roundtrip(tmp_path):
    save_golden(tmp_path, {"t1": "out1", "t2": "out2"})
    assert load_golden(tmp_path) == {"t1": "out1", "t2": "out2"}
    assert load_golden(tmp_path / "nope") is None
