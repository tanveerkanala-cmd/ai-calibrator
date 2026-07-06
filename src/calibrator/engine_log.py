"""Opt-in local logging of engine decisions — fuel for the Engine-Trainer.

When a project has ``log_interactions`` on, each call to a wrapped engine for a
role is appended to ``<project>/logs/<role>.jsonl`` as
``{role, system, prompt, schema, output}``. Those logs are the labeled dataset to
later fine-tune a LOCAL model that reproduces a cloud role (see
:mod:`calibrator.train_engine`) — the self-bootstrapping path to running the tool
privately and free on your own engines.

Logging is OFF by default, entirely local (the ``logs/`` dir is gitignored), and
best-effort: a logging failure never breaks the pipeline. Appends are serialized
in practice because callers hold the project lock across the engine calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engines.base import Engine


class LoggingEngine(Engine):
    """Wraps an engine and records each completion for ``role`` to a JSONL log."""

    def __init__(self, inner: Engine, role: str, log_dir: str | Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.role = role
        self.log_path = Path(log_dir) / f"{role}.jsonl"

    def complete(self, prompt: str, *, system: str | None = None, schema: dict | None = None) -> Any:
        out = self.inner.complete(prompt, system=system, schema=schema)
        try:
            self._record(prompt, system, schema, out)
        except Exception:
            # Best-effort by contract: logging must NEVER break the pipeline —
            # not just on OSError (disk) but also e.g. a TypeError serializing an
            # exotic output. The eval/compile result is what matters.
            pass
        return out

    def _record(self, prompt: str, system: str | None, schema: dict | None, output: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"role": self.role, "system": system, "prompt": prompt,
                  "schema": schema, "output": output}
        line = json.dumps(record, default=str)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def wrap_engine(engine: Engine, role: str, project_dir: str | Path, *, enabled: bool) -> Engine:
    """Return a logging wrapper around ``engine`` for ``role`` if ``enabled``,
    else the engine unchanged. The single switch the pipeline uses."""
    if not enabled:
        return engine
    return LoggingEngine(engine, role, Path(project_dir) / "logs")
