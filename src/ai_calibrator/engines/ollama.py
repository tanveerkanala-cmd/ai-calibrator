"""Local default engine — talks to an Ollama server on the user's machine.

No API key, no cost, fully private. Ollama constrains output to the JSON schema
when one is given; the shared ``call_json`` adds tolerant parsing + one retry on
top, so a flaky local model degrades gracefully instead of erroring.
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

from .base import Engine, EngineError, EngineTimeout, call_json

DEFAULT_TIMEOUT = 120.0

# Ollama defaults a model's context to whatever the Modelfile says — commonly
# 2048 or 4096 tokens — and SILENTLY DROPS whatever does not fit, answering 200
# with done_reason "stop". Nothing in the response says the prompt was cut, so a
# 32k-character ingest can be answered from its last few thousand characters and
# every fact, gap, spec and test downstream is built from that fragment. The
# engine therefore sizes num_ctx to the prompt it is about to send.
DEFAULT_NUM_CTX = 8192
MAX_NUM_CTX = 32768
# Rough bytes-per-token for prose. Deliberately pessimistic: over-reserving costs
# memory on the server, under-reserving costs silent truncation.
_BYTES_PER_TOKEN = 3.0


def _num_ctx_for(text: str) -> int:
    """A context window big enough for ``text`` plus room to answer in."""
    override = os.getenv("CALIBRATOR_OLLAMA_NUM_CTX")
    if override:
        try:
            n = int(override)
            if n > 0:
                return n
        except ValueError:
            pass
    needed = int(len(text) / _BYTES_PER_TOKEN) + 1024   # + headroom for the reply
    return max(DEFAULT_NUM_CTX, min(needed, MAX_NUM_CTX))


def _default_timeout() -> float:
    """Env-overridable: a slow machine (or a busy shared model) can need more
    than 120s for a big extraction — without a knob the user is simply blocked.
    Set CALIBRATOR_OLLAMA_TIMEOUT (seconds)."""
    raw = os.getenv("CALIBRATOR_OLLAMA_TIMEOUT")
    if raw:
        try:
            value = float(raw)
            if math.isfinite(value) and value > 0:  # reject inf / nan / 1e999 → no timeout
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
            "options": {"num_ctx": _num_ctx_for("".join(m["content"] for m in messages))},
        }
        if schema is not None:
            payload["format"] = schema  # Ollama constrains output to the schema
            # A thinking model spends its output budget on unconstrained
            # thinking BEFORE the grammar-constrained JSON — invisible,
            # unbounded, and it flakily starves the actual output past
            # num_predict, killing the call as a truncation. A structured
            # call's entire product is the JSON, so thinking is off here.
            # Plain calls are left alone: the subject's answers are the thing
            # being measured. Ollama accepts think=false on non-thinking
            # models without complaint.
            payload["think"] = False

        def _call() -> str:
            # Every failure mode a real server can produce must surface as a
            # FRIENDLY RuntimeError (the CLI/API show it verbatim) — never a raw
            # httpx/json traceback.
            try:
                resp = httpx.post(
                    f"{self.host}/api/chat", json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise EngineError(
                    f"Could not reach Ollama at {self.host}. Is it running?\n"
                    f"  Try:  ollama serve   (and)  ollama pull {self.model}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise EngineTimeout(
                    f"Ollama at {self.host} did not respond within {self.timeout:g}s "
                    f"(model {self.model!r} may still be loading, or the machine is overloaded).\n"
                    f"  Try raising the limit:  CALIBRATOR_OLLAMA_TIMEOUT={max(300, int(self.timeout) * 2)} "
                    "calibrate <command> …\n"
                    f"  (big extractions on a large or busy local model can exceed the {DEFAULT_TIMEOUT:g}s default.)"
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
                raise EngineError(
                    f"Ollama returned HTTP {status} for model {self.model!r}."
                    f"{hint}" + (f"\n  Server said: {body}" if body else "")
                ) from exc
            except httpx.HTTPError as exc:  # anything else transport-level
                raise EngineError(f"Ollama request to {self.host} failed: {exc}") from exc
            try:
                data = resp.json()
            except ValueError as exc:
                raise EngineError(
                    f"Ollama returned invalid JSON (truncated or corrupted response): "
                    f"{resp.text[:200]!r}"
                ) from exc
            # A cut-off answer is an error, not an answer: returned as if it were
            # finished, it is graded and certified as the whole answer.
            # The server reports how much of the prompt it actually evaluated.
            # If that is materially less than what was sent, the rest was dropped
            # on the floor and the answer is to a question nobody asked — and
            # done_reason stays "stop", so the check below never sees it.
            if isinstance(data, dict):
                sent = len("".join(m["content"] for m in messages))
                seen = data.get("prompt_eval_count")
                if isinstance(seen, int) and seen > 0 and sent / _BYTES_PER_TOKEN > seen * 2:
                    raise EngineError(
                        f"Ollama evaluated only {seen} prompt token(s) of roughly "
                        f"{int(sent / _BYTES_PER_TOKEN)} sent — model {self.model!r} "
                        "silently dropped most of the input.\n"
                        "  Raise its context length (CALIBRATOR_OLLAMA_NUM_CTX), or "
                        "use a model with a larger context."
                    )
            if isinstance(data, dict) and data.get("done_reason") == "length":
                raise EngineError(
                    f"Ollama response truncated — model {self.model!r} hit its output limit.\n"
                    f"  Raise num_predict (or the context length) for {self.model!r}, "
                    "or split the work into smaller steps."
                )
            try:
                return data["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise EngineError(
                    f"Unexpected Ollama response (missing message.content): {str(data)[:200]}"
                ) from exc

        return call_json(_call) if schema is not None else _call()
