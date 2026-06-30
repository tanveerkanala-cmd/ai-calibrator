"""Optional cloud engine — Anthropic Claude via the official SDK (BYO key / login).

Opt-in quality upgrade over the local default. Credentials resolve from
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile — so
`calibrate login claude` makes this work with no key. Install: pip install -e '.[cloud]'
"""

from __future__ import annotations

from typing import Any

from .base import Engine, call_json


class AnthropicEngine(Engine):
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 16000) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "The Anthropic cloud engine needs the `anthropic` package.\n"
                "  Install it with:  pip install -e '.[cloud]'"
            ) from exc

        self.name = f"{model}@anthropic"
        self.model = model
        self.max_tokens = max_tokens
        try:
            self._client = (
                anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            )
        except Exception as exc:
            raise RuntimeError(
                "No Claude credentials found. Log in with `calibrate login claude` "
                "(browser/OAuth — no key needed) or set ANTHROPIC_API_KEY."
            ) from exc

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # No temperature / thinking: sampling params are rejected on Opus 4.8 and
        # these calls don't need extended thinking — omitting both is safe.
        if system:
            # Cache the system prompt — reused verbatim across many calls (e.g.
            # the judge's rubric over every test case), cutting repeated cost ~90%.
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        def _call() -> str:
            resp = self._client.messages.create(**kwargs)
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"Claude declined the request ({self.name}).")
            if resp.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"Claude response truncated at max_tokens={self.max_tokens} ({self.name}); "
                    "increase max_tokens for this engine."
                )
            return next((b.text for b in resp.content if b.type == "text"), "")

        return call_json(_call) if schema is not None else _call()
