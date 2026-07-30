"""The Engine interface + role registry + factory.

Every "intelligent" step in the pipeline is an Engine call behind this one
interface, so a role can be powered by a local model, a cloud model (BYO key),
or your own fine-tuned model — interchangeably. Engine specs are strings of the
form ``model@provider`` (provider defaults to ``ollama``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class Role(str, Enum):
    """The roles an engine can fill. (The *subject* model being configured is
    separate — it is not an engine.)"""
    EXTRACTOR = "extractor"
    INTERVIEWER = "interviewer"
    PREDICTOR = "predictor"
    COMPILER = "compiler"
    JUDGE = "judge"


# --- Typed engine failures --------------------------------------------------
# All subclass RuntimeError so every existing `except RuntimeError` and message
# display keeps working; the specific types let the HTTP API map an upstream
# failure to the right status class (a provider timeout is 504, not a client 400).

class EngineError(RuntimeError):
    """An engine/provider failure (upstream), as opposed to bad user input."""

    # The upstream HTTP status, when the adapter knew one. Callers use it to tell
    # "this model rejected the request" (400 — e.g. no strict-json_schema support,
    # safe to retry unconstrained) from a timeout / 429 / 5xx, which must never
    # silently downgrade to an unconstrained call. Declared here rather than
    # assigned ad hoc so both sides of that decision share one contract.
    status: int | None = None


class EngineTimeout(EngineError):
    """The engine/provider did not respond in time."""


class EngineAuthError(EngineError):
    """Missing or invalid credentials for the engine."""


class EngineOutputError(EngineError):
    """The engine returned output that could not be parsed as the required JSON.

    ``raw`` carries the model's actual (truncated) text so a caller can save it
    for debugging instead of dumping it into a user-facing message.
    """

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def missing_credentials_message(provider: str, name: str) -> str:
    """The shared, actionable text for a missing/invalid cloud key — lists every
    way forward (a key, a login, or a free local model) so a first-time user is
    never stranded on raw SDK jargon."""
    if provider == "anthropic":
        key, login = "ANTHROPIC_API_KEY=sk-ant-...", "calibrate login claude"
    else:
        key, login = "OPENAI_API_KEY=sk-...", "see `calibrate login openai`"
    return (
        f"No usable {provider} credentials for {name}. Fix one of:\n"
        f"  • export {key}          (your own key)\n"
        f"  • {login}\n"
        "  • bind a free local model:  calibrate engines <project> --all <model>@ollama\n"
        "Run `calibrate auth` to see what's currently configured."
    )


class Engine(ABC):
    """A text-in / text-(or-JSON)-out model behind a uniform interface."""

    name: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Any:
        """Return the model's completion.

        If ``schema`` (a JSON Schema) is given, the engine constrains output to
        valid JSON and returns the parsed object; otherwise returns a string.
        """
        raise NotImplementedError


def _extract_json(text: str) -> str | None:
    """Best-effort: pull a JSON object/array out of fenced or prose-wrapped text."""
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if "\n" in t:
            first, rest = t.split("\n", 1)
            if first.strip().lower() in ("", "json") or first.strip().isalpha():
                t = rest
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
        t = t.strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    close = "}" if t[start] == "{" else "]"
    end = t.rfind(close)
    if end <= start:
        return None
    return t[start : end + 1]


def loads_tolerant(text: Any) -> Any:
    """``json.loads``, but tolerant of markdown fences / surrounding prose.

    Parses an engine's TEXT output — so a non-str (e.g. bytes, which
    ``json.loads`` would silently auto-decode) is a contract violation, rejected
    with ValueError rather than accepted."""
    if not isinstance(text, str):
        raise ValueError(f"expected engine text output, got {type(text).__name__}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _extract_json(text)
        if candidate is not None:
            return json.loads(candidate)  # JSONDecodeError (a ValueError) propagates
        raise ValueError("no JSON found in engine output") from None


def require_object(out: Any, what: str = "engine") -> dict:
    """Ensure a schema-constrained engine response is a JSON object (``dict``).

    Every schema the pipeline sends is ``type: object``, so an engine that
    honors it returns a ``dict``. A model that ignores the schema can still emit
    a valid-but-wrong JSON *value* — a string, number, or array. Without this
    guard the next line (``out.get(...)``) raises a cryptic ``AttributeError``
    deep inside a pipeline stage. Fail loudly and specifically instead.

    Built-in adapters already route through :func:`call_json` (which enforces
    the same invariant), so this primarily hardens against custom third-party
    ``Engine`` implementations that bypass it.
    """
    if not isinstance(out, dict):
        raise RuntimeError(
            f"{what} did not return a JSON object (got {type(out).__name__}); "
            "the model ignored the requested schema."
        )
    return out


def _parse_object(text: Any) -> dict:
    """Tolerantly parse ``text`` to JSON and require a JSON object.

    A non-object result (string / number / array) is treated as a parse failure
    — i.e. a :class:`ValueError` — so :func:`call_json`'s retry-then-clear-error
    path handles it uniformly with malformed JSON.
    """
    obj = loads_tolerant(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def call_json(call) -> dict:
    """Call a model (``call() -> text``) and parse a JSON **object** tolerantly.

    If the text isn't valid JSON, or is valid JSON but not an object, retry the
    call ONCE (constrained decoding usually re-yields a valid object), then give
    up with a clear error. API / connection errors raised by ``call`` propagate
    unchanged — only parse/shape failures trigger the retry, so genuine errors
    aren't masked. Always returns a ``dict`` or raises.
    """
    text = call()
    try:
        return _parse_object(text)
    except ValueError:
        pass
    text = call()
    try:
        return _parse_object(text)
    except ValueError as exc:
        raise EngineOutputError(
            "the model returned unreadable output (not valid JSON) after a retry — "
            "common with small local models. Retry, or bind a stronger model for "
            "this role with `calibrate engines <project> <role> <model>`.",
            raw=str(text),
        ) from exc


def parse_engine_spec(spec: str) -> tuple[str, str]:
    """``"qwen2.5:14b@ollama"`` -> ``("qwen2.5:14b", "ollama")``.

    Provider defaults to ``ollama`` when omitted.
    """
    if "@" in spec:
        model, provider = spec.split("@", 1)
        return model.strip(), provider.strip().lower()
    return spec.strip(), "ollama"


VALID_PROVIDERS = ("anthropic", "openai", "ollama")


def validate_engine_spec(spec: str) -> str:
    """Validate a ``model@provider`` string WITHOUT constructing the engine
    (no SDK import, no key needed) — for rebinding roles. Returns ``spec`` or
    raises ValueError with an actionable message."""
    model, provider = parse_engine_spec(spec)
    if not model:
        raise ValueError(f"Invalid engine spec {spec!r}: no model name. Use `model@provider` "
                         "(e.g. `gpt-4o-mini@openai`) or just `model` for local Ollama.")
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r} in {spec!r}. "
                         f"Valid providers: {', '.join(VALID_PROVIDERS)}.")
    return spec


def get_engine(spec: str) -> Engine:
    """Build an Engine from a ``model@provider`` spec."""
    model, provider = parse_engine_spec(spec)
    if not model:
        raise ValueError(
            f"Invalid engine spec {spec!r}: no model name. "
            "Use `model@provider` (e.g. `claude-opus-4-8@anthropic`) or just `model` for local Ollama."
        )
    if provider == "ollama":
        from .ollama import OllamaEngine
        return OllamaEngine(model)
    if provider == "anthropic":
        from .anthropic import AnthropicEngine
        return AnthropicEngine(model)
    if provider == "openai":
        from .openai import OpenAIEngine
        return OpenAIEngine(model)
    raise ValueError(
        f"Unknown engine provider {provider!r} in {spec!r}. "
        "Valid providers: anthropic, openai, ollama."
    )
