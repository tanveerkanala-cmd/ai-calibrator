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
        resp = self._client.chat.completions.create(**kwargs)
        try:
            message = resp.choices[0].message
        except (IndexError, AttributeError) as exc:
            raise RuntimeError(
                f"OpenAI returned no choices for {self.name} (empty or malformed response)."
            ) from exc
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
