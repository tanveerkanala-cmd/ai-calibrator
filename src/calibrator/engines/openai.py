"""Optional cloud engine — OpenAI (and OpenAI-compatible) products (BYO key).

Bind any role to ``<model>@openai`` (e.g. ``gpt-4o@openai``). Works with any
OpenAI-compatible endpoint via ``OPENAI_BASE_URL``. Strict ``json_schema`` is the
primary structured-output path; if a model doesn't support it, this falls back
to a plain JSON-instruction call. Install: pip install -e '.[cloud]'
"""

from __future__ import annotations

import json
from typing import Any

from .base import Engine, call_json


def _friendly_openai_error(name: str, model: str, exc: Exception) -> RuntimeError | None:
    """Map a raw openai-SDK exception to an actionable RuntimeError, or None if
    it is not an OpenAI API error (then the caller re-raises the original).

    Confirmed live: a wrong key / typo'd model otherwise surfaced a raw
    openai.NotFoundError / AuthenticationError traceback — same class of gap the
    Ollama adapter closed."""
    import openai
    if isinstance(exc, openai.AuthenticationError):
        return RuntimeError(f"OpenAI rejected the API key (401) for {name}. "
                            "Check OPENAI_API_KEY (https://platform.openai.com/api-keys).")
    if isinstance(exc, openai.PermissionDeniedError):
        return RuntimeError(f"OpenAI denied access (403) to {model!r} for {name}.")
    if isinstance(exc, openai.NotFoundError):
        return RuntimeError(f"OpenAI has no model {model!r} (404), or your key lacks access to it.")
    if isinstance(exc, openai.RateLimitError):
        return RuntimeError(f"OpenAI rate limit or quota hit (429) for {name} — "
                            "slow down or check your plan/billing.")
    if isinstance(exc, openai.APITimeoutError):
        return RuntimeError(f"OpenAI request timed out for {name} — the API may be slow; retry.")
    if isinstance(exc, openai.APIConnectionError):
        return RuntimeError(f"Could not reach OpenAI for {name} (network/endpoint issue): {exc}")
    if isinstance(exc, openai.APIError):
        status = getattr(exc, "status_code", "?")
        return RuntimeError(f"OpenAI API error ({status}) for {name}: {getattr(exc, 'message', exc)}")
    return None


class OpenAIEngine(Engine):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "The OpenAI cloud engine needs the `openai` package.\n"
                "  Install it with:  pip install -e '.[cloud]'"
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
            raise RuntimeError(
                "No OpenAI credentials found. Set OPENAI_API_KEY "
                "(https://platform.openai.com/api-keys) — see `calibrate login openai`."
            ) from exc

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
            message = resp.choices[0].message
        except (IndexError, AttributeError, TypeError, KeyError) as exc:
            # TypeError/KeyError too: choices may be None or a non-list on a
            # malformed (or OpenAI-compatible) response, not just empty/missing.
            raise RuntimeError(
                f"OpenAI returned no usable choices for {self.name} (empty or malformed response)."
            ) from exc
        if message is None:  # a valid response can still carry a null message
            raise RuntimeError(f"OpenAI returned an empty message for {self.name} (malformed response).")
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"OpenAI declined the request ({self.name}): {refusal}")
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
            except Exception:
                # The model may not support strict json_schema — fall back to a
                # plain call with a JSON-only instruction. (Auth/other errors
                # re-raise from this second call, so they aren't masked.)
                instructed = messages + [
                    {
                        "role": "system",
                        "content": "Respond with ONLY valid JSON matching this schema: "
                        + json.dumps(schema),
                    }
                ]
                return self._chat(instructed)

        return call_json(_call)
