"""Project persistence — atomic save + load round-trip."""

import pytest

from calibrator.models import Project, TaskType
from calibrator.store import load_project, save_project


def test_save_load_roundtrip(tmp_path):
    p = Project(name="x", goal="answer questions", task_type=TaskType.SUPPORT_ASSISTANT)
    save_project(p, tmp_path / "x")
    loaded = load_project(tmp_path / "x")
    assert loaded.name == "x"
    assert loaded.goal == "answer questions"
    assert loaded.task_type is TaskType.SUPPORT_ASSISTANT
    assert (tmp_path / "x" / "project.yaml").exists()
    # the atomic temp file is renamed away, not left behind
    assert not (tmp_path / "x" / "project.yaml.tmp").exists()


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "nope")


def test_atomic_write_text_writes_completely_and_leaves_no_temp(tmp_path):
    from calibrator.store import atomic_write_text
    p = tmp_path / "sub" / "f.json"
    atomic_write_text(p, '{"a": 1}')
    assert p.read_text() == '{"a": 1}'
    assert list((tmp_path / "sub").glob("*.tmp")) == []


def test_atomic_write_text_never_truncates_on_failure(tmp_path, monkeypatch):
    import os

    from calibrator.store import atomic_write_text
    p = tmp_path / "f.txt"
    p.write_text("ORIGINAL")  # prior complete content

    def boom(_fd):
        raise OSError("simulated crash mid-write")
    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError):
        atomic_write_text(p, "NEW DATA " * 1000)
    assert p.read_text() == "ORIGINAL"            # prior version intact — never truncated
    assert list(tmp_path.glob("*.tmp")) == []      # scratch file cleaned up


def test_save_scorecard_is_atomic(tmp_path, monkeypatch):
    """The scorecard (eval's source of truth) must never be left truncated. (stress finding)"""
    import os

    from calibrator.eval import save_scorecard
    from calibrator.models import CriterionResult, Scorecard, TestResult
    card = Scorecard(run_id="run-0001", results=[
        TestResult(test_id="t1", output="x" * 10000, criteria=[CriterionResult(criterion_id="c", passed=True)])])

    def boom(_fd):
        raise OSError("crash")
    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        save_scorecard(tmp_path, card)

    d = tmp_path / "evals" / "run-0001"
    assert not (d / "scorecard.json").exists()     # never a truncated scorecard on disk
    assert list(d.glob("*.tmp")) == []
