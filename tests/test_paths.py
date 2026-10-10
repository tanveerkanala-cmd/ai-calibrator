"""paths.contained / run_dir — the backstop under every path component that came
from a CLI flag or an HTTP request (project names, run ids, upload filenames,
engine roles)."""

import os
from pathlib import Path

import pytest

from ai_calibrator.paths import contained, run_dir


def test_a_child_comes_back_absolute(tmp_path):
    assert contained(tmp_path, "a", "b.txt") == tmp_path / "a" / "b.txt"


@pytest.mark.parametrize("parts", [
    ("..",), ("",), (".",), ("a", "..", ".."), ("../sibling",), ("a/../../x",),
])
def test_refuses_anything_not_strictly_inside(tmp_path, parts):
    with pytest.raises(ValueError, match="points outside"):
        contained(tmp_path / "base", *parts)


def test_an_absolute_component_is_refused(tmp_path):
    # os.path.join discards everything before an absolute part — the classic escape.
    with pytest.raises(ValueError):
        contained(tmp_path / "base", str(tmp_path / "elsewhere"))


def test_a_sibling_sharing_the_prefix_is_not_a_child(tmp_path):
    # ".../base-evil" starts with ".../base" as a string; it must not pass.
    with pytest.raises(ValueError):
        contained(tmp_path / "base", "../base-evil")


def test_a_relative_base_compares_like_an_absolute_one(tmp_path, monkeypatch):
    # The CLI passes project dirs as given ("."); the check must not compare a
    # relative target against an absolute prefix and refuse every child.
    monkeypatch.chdir(tmp_path)
    assert contained(".", "evals", "r1") == tmp_path / "evals" / "r1"
    with pytest.raises(ValueError):
        contained("proj", "..")


def test_a_filesystem_root_base_is_not_doubled():
    root = os.path.abspath(os.sep)
    assert contained(root, "x") == Path(root) / "x"


def test_a_symlinked_child_is_still_inside(tmp_path):
    # Lexical on purpose: a project folder the user symlinked into the served
    # root keeps working. The check is where the NAME points, not the link.
    real = tmp_path / "real"
    real.mkdir()
    base = tmp_path / "root"
    base.mkdir()
    try:
        (base / "linked").symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert contained(base, "linked") == base / "linked"


@pytest.mark.parametrize("rid", ["", "  ", "a/b", "a\\b", "..", "x..y", None])
def test_run_dir_keeps_the_message_callers_show(tmp_path, rid):
    with pytest.raises(ValueError, match="invalid run id"):
        run_dir(tmp_path, rid)


def test_run_dir_accepts_a_real_run_id(tmp_path):
    assert run_dir(tmp_path, "20261010-142619") == tmp_path / "evals" / "20261010-142619"


def test_a_drive_relative_run_id_cannot_leave_the_project(tmp_path):
    """Joining "D:x" onto C:\\proj\\evals yields D:x — another drive, with no
    separator or ".." for the old shape check to catch. Both readers of a run
    id (the scorecard loader, and the label writer) go through run_dir."""
    if os.name != "nt":
        pytest.skip("platform-specific: drive-relative paths exist only on Windows")
    from ai_calibrator.drift import load_scorecard
    from ai_calibrator.judge_check import save_labels

    other = "C:" if str(tmp_path).upper().startswith("D:") else "D:"
    rid = f"{other}x"
    with pytest.raises(ValueError, match="invalid run id"):
        run_dir(tmp_path, rid)
    with pytest.raises(ValueError, match="invalid run id"):
        load_scorecard(tmp_path, rid)
    with pytest.raises(ValueError, match="invalid run id"):
        save_labels(tmp_path, rid, [])


def test_read_log_refuses_a_role_that_would_leave_logs(tmp_path):
    from ai_calibrator.train_engine import read_log

    with pytest.raises(ValueError):
        read_log(tmp_path, "../../outside")
