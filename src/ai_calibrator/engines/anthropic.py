"""Optional cloud engine — Anthropic Claude via the official SDK (BYO key / login).

Opt-in quality upgrade over the local default. Credentials resolve from
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile — so
`calibrate login claude` makes this work with no key. Install: pip install -e '.[cloud]'
"""

from __future__ import annotations

from typing import Any

from .base import (
    Engine,
    EngineAuthError,
    EngineError,
    EngineTimeout,
    call_json,
    missing_credentials_message,
)


def _friendly_anthropic_error(exc: Exception, name: str) -> EngineError | None:
    """Map a raw anthropic-SDK exception to an actionable EngineError, or None if
    it is not an Anthropic SDK error (then the caller re-raises the original).
    Mirrors the OpenAI adapter so the DEFAULT cloud engine never surfaces a raw
    SDK traceback on a bad key / model / rate limit."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return None
    if isinstance(exc, anthropic.AuthenticationError):
        return EngineAuthError(missing_credentials_message("anthropic", name))
    if isinstance(exc, anthropic.PermissionDeniedError):
        return EngineError(f"Anthropic denied access (403) for {name} — your key may lack this model.")
    if isinstance(exc, anthropic.NotFoundError):
        return EngineError(f"Anthropic has no such model for {name} (404), or your key lacks access.")
    if isinstance(exc, anthropic.RateLimitError):
        return EngineError(f"Anthropic rate limit or quota hit (429) for {name} — slow down or check billing.")
    if isinstance(exc, anthropic.APITimeoutError):
        return EngineTimeout(f"Anthropic request timed out for {name} — the API may be slow; retry.")
    if isinstance(exc, anthropic.APIConnectionError):
        return EngineError(f"Could not reach Anthropic for {name} (network/endpoint issue): {exc}")
    if isinstance(exc, anthropic.APIError):
        status = getattr(exc, "status_code", "?")
        return EngineError(f"Anthropic API error ({status}) for {name}: {getattr(exc, 'message', exc)}")
    # The base SDK error (NOT an APIError) is what a missing / empty key raises at
    # request-build time ("Could not resolve authentication method …") — the exact
    # raw jargon a keyless first run used to hit. Treat it as missing credentials.
    if isinstance(exc, anthropic.AnthropicError):
        return EngineAuthError(missing_credentials_message("anthropic", name))
    return None


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
            raise EngineAuthError(missing_credentials_message("anthropic", self.name)) from exc

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
            try:
                resp = self._client.messages.create(**kwargs)
            except Exception as exc:  # map SDK errors (auth/rate-limit/conn/timeout) to friendly text
                friendly = _friendly_anthropic_error(exc, self.name)
                if friendly is not None:
                    raise friendly from exc
                raise
            if resp.stop_reason == "refusal":
                raise EngineError(f"Claude declined the request ({self.name}).")
            if resp.stop_reason == "max_tokens":
                raise EngineError(
                    f"Claude response truncated at max_tokens={self.max_tokens} ({self.name}); "
                    "increase max_tokens for this engine."
                )
            # content can be None/empty on a malformed or OpenAI-compatible proxy
            # response — don't let that raise a raw TypeError.
            blocks = resp.content or []
            # `or ""` guards a text block whose .text is None — the contract is a str.
            return next((b.text or "" for b in blocks if getattr(b, "type", None) == "text"), "")

        return call_json(_call) if schema is not None else _call()
