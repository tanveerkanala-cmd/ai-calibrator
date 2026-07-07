"""Cross-process **and** cross-thread advisory file lock.

A project is a directory of plain files that several actors may touch at once:
concurrent HTTP requests in the local API (Starlette dispatches sync endpoints
to a thread pool) and, in principle, more than one `calibrate` process. A bare
read-modify-write on ``project.yaml`` therefore races — the losing writer's
changes vanish ("lost update"). This lock serializes those critical sections.

Design notes:
- POSIX uses ``fcntl.flock``; Windows uses ``msvcrt.locking``. Both are
  advisory — they only exclude other holders of *this* lock, which is exactly
  what we want.
- Every ``acquire`` opens its **own** file descriptor. ``flock`` associates the
  lock with the open file description, so two descriptors block each other even
  inside one process — giving cross-thread mutual exclusion for free.
- The lock is **not** re-entrant: acquiring it twice on the same path in one
  thread deadlocks. Callers hold a project's lock for exactly one
  read-modify-write region and never nest.
- If no locking facility is available, acquisition degrades to a no-op. Atomic
  file replacement (see ``store.save_project``) still prevents *corruption*; the
  lock only adds *serialization* of logical updates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:  # POSIX
    import fcntl

    _BACKEND = "fcntl"
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]
    try:
        import msvcrt

        _BACKEND = "msvcrt"
    except ImportError:  # pragma: no cover - no locking facility at all
        msvcrt = None  # type: ignore[assignment]
        _BACKEND = "none"


class FileLock:
    """An exclusive advisory lock bound to a lock file.

    Usage::

        with FileLock(path):
            ...  # critical section

    Blocks until the lock is acquired.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: Optional[int] = None

    def acquire(self) -> "FileLock":
        # Not re-entrant (see class docstring). A second acquire on the same
        # instance would overwrite self._fd — leaking the first descriptor and,
        # under fcntl, blocking forever waiting on a lock this process holds.
        # Fail fast with a clear programming error instead.
        if self._fd is not None:
            raise RuntimeError("FileLock is already held — it is not re-entrant; do not nest it")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_CREAT so the first caller materializes the lock file; the descriptor
        # stays open for the whole critical section and carries the lock.
        self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if _BACKEND == "fcntl":
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            elif _BACKEND == "msvcrt":  # pragma: no cover - Windows only
                # Lock one byte at offset 0; LK_LOCK blocks (retrying) until free.
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
            # _BACKEND == "none": best-effort no-op (atomic writes still apply).
        except OSError:
            # Never leak the descriptor if locking itself failed.
            os.close(self._fd)
            self._fd = None
            raise
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if _BACKEND == "fcntl":
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            elif _BACKEND == "msvcrt":  # pragma: no cover - Windows only
                try:
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc: object) -> bool:
        self.release()
        return False
