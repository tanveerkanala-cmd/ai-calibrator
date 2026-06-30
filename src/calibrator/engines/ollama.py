"""Local default engine — talks to an Ollama server on the user's machine.

No API key, no cost, fully private. Ollama constrains output to the JSON schema
when one is given; the shared ``call_json`` adds tolerant parsing + one retry on
top, so a flaky local model degrades gracefully instead of erroring.
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import Engine, call_json


class OllamaEngine(Engine):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.name = f"{model}@ollama"
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

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
            return resp.json()["message"]["content"]

        return call_json(_call) if schema is not None else _call()
