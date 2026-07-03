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


def test_unknown_fields_survive_round_trip(tmp_path):
    """Fields from a newer version (or a typo) must survive load→save, not vanish.
    (audit finding: pydantic's default extra='ignore' silently destroyed them)"""
    import yaml as _yaml

    from calibrator.store import load_project, save_project

    d = tmp_path / "p"
    d.mkdir()
    (d / "project.yaml").write_text(_yaml.safe_dump({
        "name": "p", "goal": "g",
        "future_top_level": {"nested": [1, 2]},               # top level
        "spec": {"goal": "g", "spec_future": "keep me",       # spec level
                 "eval_criteria": [{"id": "c1", "description": "d", "crit_future": 7}]},
        "tests": [{"id": "t1", "input": "q", "test_future": True}],
    }))
    project = load_project(d)
    save_project(project, d)                                   # round-trip
    on_disk = _yaml.safe_load((d / "project.yaml").read_text())
    assert on_disk["future_top_level"] == {"nested": [1, 2]}
    assert on_disk["spec"]["spec_future"] == "keep me"
    assert on_disk["spec"]["eval_criteria"][0]["crit_future"] == 7
    assert on_disk["tests"][0]["test_future"] is True


def test_yaml_tab_error_is_friendly(tmp_path):
    """A trailing tab (classic hand-edit slip) must produce an actionable message
    with the location and a tab hint — not a raw pyyaml ScannerError."""
    import pytest as _pytest

    from calibrator.store import load_project

    d = tmp_path / "p"
    d.mkdir()
    (d / "project.yaml").write_text("name: Test Project\t\ngoal: g\n")
    with _pytest.raises(ValueError) as err:
        load_project(d)
    msg = str(err.value)
    assert "line 1" in msg and "not valid YAML" in msg
    assert "tab" in msg.lower() and "spaces" in msg


def test_write_project_gitignore(tmp_path):
    from calibrator.store import write_project_gitignore

    target = write_project_gitignore(tmp_path)
    body = target.read_text()
    assert "evals/" in body and ".env" in body and "*.key" in body
    # never clobbers user edits
    target.write_text("# mine\n")
    write_project_gitignore(tmp_path)
    assert target.read_text() == "# mine\n"
