"""Local default engine — talks to an Ollama server on the user's machine.

No API key, no cost, fully private. Ollama constrains output to the JSON schema
when one is given; the shared ``call_json`` adds tolerant parsing + one retry on
top, so a flaky local model degrades gracefully instead of erroring.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import Engine, call_json

DEFAULT_TIMEOUT = 120.0


def _default_timeout() -> float:
    """Env-overridable: a slow machine (or a busy shared model) can need more
    than 120s for a big extraction — without a knob the user is simply blocked.
    Set CALIBRATOR_OLLAMA_TIMEOUT (seconds)."""
    raw = os.getenv("CALIBRATOR_OLLAMA_TIMEOUT")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass  # ignore junk; fall through to the default
    return DEFAULT_TIMEOUT


class OllamaEngine(Engine):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float | None = None,
    ) -> None:
        self.name = f"{model}@ollama"
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout if timeout is not None else _default_timeout()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Any:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if schema is not None:
            payload["format"] = schema  # Ollama constrains output to the schema

        def _call() -> str:
            # Every failure mode a real server can produce must surface as a
            # FRIENDLY RuntimeError (the CLI/API show it verbatim) — never a raw
            # httpx/json traceback. (audit: provider-adapter failure modes)
            try:
                resp = httpx.post(
                    f"{self.host}/api/chat", json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Could not reach Ollama at {self.host}. Is it running?\n"
                    f"  Try:  ollama serve   (and)  ollama pull {self.model}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise RuntimeError(
                    f"Ollama at {self.host} did not respond within {self.timeout:g}s "
                    f"(model {self.model!r} may still be loading, or the machine is overloaded)."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                hint = ""
                if status == 404:
                    hint = f"\n  Try:  ollama pull {self.model}   (the model may not be installed)"
                elif status in (401, 403):
                    hint = "\n  This Ollama server requires authentication this tool did not send."
                elif status >= 500:
                    hint = "\n  The Ollama server hit an internal error — check its logs."
                body = exc.response.text[:200]
                raise RuntimeError(
                    f"Ollama returned HTTP {status} for model {self.model!r}."
                    f"{hint}" + (f"\n  Server said: {body}" if body else "")
                ) from exc
            except httpx.HTTPError as exc:  # anything else transport-level
                raise RuntimeError(f"Ollama request to {self.host} failed: {exc}") from exc
            try:
                data = resp.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Ollama returned invalid JSON (truncated or corrupted response): "
                    f"{resp.text[:200]!r}"
                ) from exc
            try:
                return data["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Unexpected Ollama response (missing message.content): {str(data)[:200]}"
                ) from exc

        return call_json(_call) if schema is not None else _call()
