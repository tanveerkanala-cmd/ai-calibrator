"""Project persistence — a project is a directory of plain, git-friendly files.

v0 keeps the whole Project in one ``project.yaml`` for simplicity. As the spec
and artifacts grow (M3+), these can split into ``spec.yaml`` / ``build/`` per
the architecture's data model, but the load/save contract stays the same.

Writes are **atomic and durable**: each writer streams to its own uniquely
named temp file, fsyncs it, then ``os.replace``s it onto ``project.yaml`` (an
atomic rename on the same filesystem). A reader therefore always sees either the
complete old file or the complete new one — never a half-written or missing
file — even under heavy concurrency. ``project_lock`` serializes the
read-modify-write *logic* on top of that, so concurrent updaters can't lose each
other's changes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from .locking import FileLock
from .models import Project

PROJECT_FILE = "project.yaml"
LOCK_FILE = ".lock"


def project_lock(path: str | Path) -> FileLock:
    """Return an exclusive lock for a project directory's read-modify-write.

    Hold it across the whole ``load → mutate → save`` region so concurrent
    actors (API thread-pool requests, multiple CLI processes) serialize instead
    of clobbering one another::

        with project_lock(d):
            project = load_project(d)
            ...mutate...
            save_project(project, d)

    The lock is per-directory, so operations on *different* projects still run
    fully in parallel. Not re-entrant — never nest it for the same project.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return FileLock(directory / LOCK_FILE)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically (unique temp + fsync + replace).

    Reusable for artifact files written under a fixed name (e.g. the rightsize
    summary), so concurrent writers can't corrupt or half-write them."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    tmp: Path | None = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return target


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a rename survives a crash (POSIX)."""
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - e.g. Windows can't open a dir this way
        return
    try:
        os.fsync(dir_fd)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        pass
    finally:
        os.close(dir_fd)


def save_project(project: Project, path: str | Path) -> Path:
    """Write the project to ``<path>/project.yaml`` atomically and durably.

    Safe under concurrency: a per-writer temp file means concurrent savers never
    share (and so never clobber) the same scratch file, and the final
    ``os.replace`` is atomic — so a reader never observes a partial write and a
    crash mid-write can't corrupt the single source-of-truth file.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "materials").mkdir(exist_ok=True)
    target = directory / PROJECT_FILE
    data = yaml.safe_dump(project.model_dump(mode="json"), sort_keys=False)

    # Unique temp file in the SAME directory (so os.replace stays atomic — a
    # cross-filesystem rename is not). mkstemp gives each concurrent writer its
    # own path, eliminating the shared-".tmp"-file race entirely.
    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=PROJECT_FILE + ".", suffix=".tmp")
    tmp: Path | None = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())  # durability: bytes hit disk before the rename
        os.replace(tmp, target)  # atomic on the same filesystem
        tmp = None  # ownership transferred to `target`; nothing to clean up
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)  # an error before replace left a stray temp

    _fsync_dir(directory)
    return target


def load_project(path: str | Path) -> Project:
    """Load the project from ``<path>/project.yaml``."""
    target = Path(path) / PROJECT_FILE
    if not target.exists():
        raise FileNotFoundError(
            f"No calibrator project at {Path(path)} (missing {PROJECT_FILE}). "
            "Run `calibrate init` first."
        )
    data = yaml.safe_load(target.read_text())
    return Project.model_validate(data)
