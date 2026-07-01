"""Golden-output snapshots — pin outputs and detect text changes."""

from calibrator.models import Scorecard
from calibrator.models import TestResult as Result  # aliased: avoids pytest collecting the model
from calibrator.snapshot import compare, load_golden, outputs_of, save_golden


def _card(run_id, outs):
    return Scorecard(run_id=run_id, results=[Result(test_id=t, output=o) for t, o in outs.items()])


def test_outputs_of():
    assert outputs_of(_card("r", {"t1": "x", "t2": "y"})) == {"t1": "x", "t2": "y"}


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
