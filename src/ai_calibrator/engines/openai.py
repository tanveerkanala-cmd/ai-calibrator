"""Optional cloud engine — OpenAI (and OpenAI-compatible) products (BYO key).

Bind any role to ``<model>@openai`` (e.g. ``gpt-4o@openai``). Works with any
OpenAI-compatible endpoint via ``OPENAI_BASE_URL``. Strict ``json_schema`` is the
primary structured-output path; if a model doesn't support it, this falls back
to a plain JSON-instruction call. Install: pip install -e '.[cloud]'
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    Engine,
    EngineAuthError,
    EngineError,
    EngineTimeout,
    call_json,
    missing_credentials_message,
)


def _friendly_openai_error(name: str, model: str, exc: Exception) -> EngineError | None:
    """Map a raw openai-SDK exception to an actionable EngineError, or None if
    it is not an OpenAI SDK error (then the caller re-raises the original).

    Confirmed live: a wrong key / typo'd model otherwise surfaced a raw
    openai.NotFoundError / AuthenticationError traceback — same class of gap the
    Ollama adapter closed."""
    import openai
    if isinstance(exc, openai.AuthenticationError):
        return EngineAuthError(missing_credentials_message("openai", name))
    if isinstance(exc, openai.PermissionDeniedError):
        return EngineError(f"OpenAI denied access (403) to {model!r} for {name}.")
    if isinstance(exc, openai.NotFoundError):
        return EngineError(f"OpenAI has no model {model!r} (404), or your key lacks access to it.")
    if isinstance(exc, openai.RateLimitError):
        return EngineError(f"OpenAI rate limit or quota hit (429) for {name} — "
                           "slow down or check your plan/billing.")
    if isinstance(exc, openai.APITimeoutError):
        return EngineTimeout(f"OpenAI request timed out for {name} — the API may be slow; retry.")
    if isinstance(exc, openai.APIConnectionError):
        return EngineError(f"Could not reach OpenAI for {name} (network/endpoint issue): {exc}")
    if isinstance(exc, openai.APIError):
        status = getattr(exc, "status_code", "?")
        err = EngineError(f"OpenAI API error ({status}) for {name}: {getattr(exc, 'message', exc)}")
        # Carry the HTTP status so callers can distinguish "this model rejected the
        # request" (400 — e.g. no strict-json_schema support, safe to retry
        # unconstrained) from a timeout / 429 / 5xx, which must never silently
        # downgrade to an unconstrained call.
        err.status = status if isinstance(status, int) else None
        return err
    # The base SDK error (NOT an APIError) is what a missing / empty key raises
    # before any HTTP call — treat it as missing credentials, not a raw traceback.
    if isinstance(exc, openai.OpenAIError):
        return EngineAuthError(missing_credentials_message("openai", name))
    return None


class OpenAIEngine(Engine):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "The OpenAI cloud engine needs the `openai` package.\n"
                "  Install it with:  pip install -e '.[cloud]'  (in your ai-calibrator clone)"
            ) from exc

        self.name = f"{model}@openai"
        self.model = model
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            self._client = OpenAI(**kwargs)
        except Exception as exc:
            raise EngineAuthError(missing_credentials_message("openai", self.name)) from exc

    def _chat(self, messages: list[dict], response_format: dict | None = None) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        # No max_tokens / temperature — keeps the adapter model-agnostic across
        # the chat and reasoning model families.
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # map raw SDK errors (auth/404/rate/timeout/conn) to friendly
            friendly = _friendly_openai_error(self.name, self.model, exc)
            if friendly is not None:
                raise friendly from exc
            raise
        try:
            choice = resp.choices[0]
            message = choice.message
        except (IndexError, AttributeError, TypeError, KeyError) as exc:
            # TypeError/KeyError too: choices may be None or a non-list on a
            # malformed (or OpenAI-compatible) response, not just empty/missing.
            raise EngineError(
                f"OpenAI returned no usable choices for {self.name} (empty or malformed response)."
            ) from exc
        if message is None:  # a valid response can still carry a null message
            raise EngineError(f"OpenAI returned an empty message for {self.name} (malformed response).")
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise EngineError(f"OpenAI declined the request ({self.name}): {refusal}")
        # A cut-off answer is an error, exactly as it is on the Anthropic adapter:
        # returned as if it were finished, it is graded and certified as the whole
        # answer, and on the schema path its half-written JSON is diagnosed as a
        # weak model instead of an exhausted output budget. Match "length" strictly
        # — OpenAI-compatible endpoints omit the field or send their own values.
        if getattr(choice, "finish_reason", None) == "length":
            raise EngineError(
                f"OpenAI response truncated — {self.model!r} hit its output limit ({self.name}).\n"
                "  Bind a model with a larger output budget, or split the work into smaller steps."
            )
        return message.content or ""

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

        if schema is None:
            return self._chat(messages)

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": True},
        }

        def _call() -> str:
            try:
                return self._chat(messages, response_format=response_format)
            except EngineError as exc:
                # ONLY "this model doesn't support strict json_schema" may fall back.
                # Catching everything meant a transient timeout, a 429, a 5xx or an
                # auth failure silently dropped schema enforcement, and an
                # unconstrained-but-parseable reply was then accepted as a
                # successfully constrained result.
                if getattr(exc, "status", None) != 400:
                    raise
                instructed = messages + [
                    {
                        "role": "system",
                        "content": "Respond with ONLY valid JSON matching this schema: "
                        + json.dumps(schema),
                    }
                ]
                return self._chat(instructed)

        return call_json(_call)
