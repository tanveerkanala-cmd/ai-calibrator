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
