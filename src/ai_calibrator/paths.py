"""The one place a path is built from a value that came from outside.

Project names, run ids, upload filenames and engine roles all arrive from a CLI
flag or an HTTP request and end up as directory or file components. Each has its
own validator (``validate_project_name``, the run-id checks, ``Path(...).name``,
the role allowlist), and those stay — they give the caller a precise 4xx. This
is the backstop under all of them: whatever a validator lets through, the joined
path must still land inside the directory it was joined onto.

Why a backstop on top of validators that already hold: a validator is a list of
shapes someone thought of. ``..`` and ``/`` were on every list here; the Windows
drive-relative form was not — joining ``"D:x"`` onto ``C:\\proj\\evals`` yields
``D:x``, a path on another drive, with no separator or ``..`` anywhere in it.
Checking where the path actually lands needs no list.

The check is lexical (``abspath``, not ``realpath``): it decides where a NAME
points, and leaves symlinks alone, so a project folder the user symlinked into
the served root keeps working exactly as before.
"""

from __future__ import annotations

import os
from pathlib import Path


def contained(base: str | Path, *parts: str) -> Path:
    """``base / parts...``, or ``ValueError`` if that lands outside ``base``.

    Strictly inside: a join that resolves to ``base`` itself (an empty or ``.``
    component) is refused too — every caller wants a child, never the parent
    back. Returns an absolute path."""
    root = os.path.abspath(base)
    target = os.path.abspath(os.path.join(root, *parts))
    # join(root, "") appends exactly one separator, so "/a/bc" can't pass for
    # a child of "/a/b", and a filesystem root ("/", "C:\\") isn't doubled.
    if not target.startswith(os.path.join(root, "")):
        raise ValueError(f"{os.path.join(*parts) if parts else ''!r} points outside {root}")
    return Path(target)


def run_dir(project_dir: str | Path, run_id: str) -> Path:
    """``<project>/evals/<run_id>``, for a run id a user or client supplied.

    A run id is ONE component: the shape check refuses separators and ``..``
    with the message callers already show, and ``contained`` refuses whatever
    else would leave ``evals/`` — the drive-relative form among them."""
    if not isinstance(run_id, str) or not run_id.strip() \
            or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError(f"invalid run id: {run_id!r}")
    try:
        return contained(Path(project_dir) / "evals", run_id)
    except ValueError:
        raise ValueError(f"invalid run id: {run_id!r}") from None
