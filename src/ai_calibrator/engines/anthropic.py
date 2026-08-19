"""Optional cloud engine — Anthropic Claude via the official SDK (BYO key / login).

Opt-in quality upgrade over the local default. Credentials resolve from
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile — so
`calibrate login claude` makes this work with no key. Install: pip install -e '.[cloud]'
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .base import (
    Engine,
    EngineAuthError,
    EngineError,
    EngineTimeout,
    call_json,
    missing_credentials_message,
    prune_schema_constraints,
)

DEFAULT_MAX_TOKENS = 16000

# The SDK refuses a NON-STREAMING request whose max_tokens implies more than 10
# minutes of generation — it budgets 3600s per 128k tokens and raises before any
# HTTP call. These are non-streaming calls, so anything above this is not a
# bigger budget, it is a guaranteed failure on every request.
MAX_NONSTREAMING_TOKENS = 21_333


def _default_max_tokens() -> int:
    """Env-overridable: a long compile or a big export can need more than 16k of
    output, and with no knob the truncation error below is a dead end. Set
    CALIBRATOR_ANTHROPIC_MAX_TOKENS (tokens), up to MAX_NONSTREAMING_TOKENS."""
    raw = os.getenv("CALIBRATOR_ANTHROPIC_MAX_TOKENS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                if value > MAX_NONSTREAMING_TOKENS:
                    # Say so. Silently honouring a smaller number than the one the
                    # operator set makes the next truncation inexplicable.
                    print(f"CALIBRATOR_ANTHROPIC_MAX_TOKENS={value} exceeds what one "
                          f"non-streaming request can carry; using {MAX_NONSTREAMING_TOKENS}.",
                          file=sys.stderr)
                return min(value, MAX_NONSTREAMING_TOKENS)
        except ValueError:
            pass  # ignore junk; fall through to the default
    return DEFAULT_MAX_TOKENS


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
    # Belt and braces: the SDK signals an unresolvable credential with a plain
    # builtin TypeError, which no isinstance check above can catch. Match on what
    # it is about rather than on its class.
    if isinstance(exc, TypeError) and any(
        token in str(exc).lower() for token in ("authentication", "api_key", "auth_token")
    ):
        return EngineAuthError(missing_credentials_message("anthropic", name))
    return None


def _credentials_available() -> bool:
    """Is there any credential the SDK could authenticate with?

    Mirrors what `calibrate auth` reports, so the two never disagree about
    whether Claude is usable."""
    import os
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    from ..auth import _anthropic_cli  # the verified Anthropic CLI, not Apache Ant
    return _anthropic_cli() is not None


class AnthropicEngine(Engine):
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "The Anthropic cloud engine needs the `anthropic` package.\n"
                "  Install it with:  pip install -e '.[cloud]'  (in your ai-calibrator clone)"
            ) from exc

        self.name = f"{model}@anthropic"
        self.model = model
        self.max_tokens = max_tokens if max_tokens is not None else _default_max_tokens()
        # Resolve credentials HERE rather than letting the SDK fail later. The
        # client constructor succeeds without a key and defers the failure to the
        # first request, where the SDK raises a plain builtin (not an
        # AnthropicError) — so the friendly-error mapper below never saw it and a
        # keyless first run on the DEFAULT engine got raw SDK jargon. Anthropic is
        # the default binding, so this is the single most likely first experience.
        if not api_key and not _credentials_available():
            raise EngineAuthError(missing_credentials_message("anthropic", self.name))
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
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": prune_schema_constraints(schema)}
            }

        def _call() -> str:
            try:
                resp = self._client.messages.create(**kwargs)
            except ValueError as exc:
                # The SDK validates the output budget before it sends, and does it
                # with a plain ValueError — which _friendly_anthropic_error can't
                # map, so it surfaced as raw SDK text and was reported as bad user
                # input rather than as an engine failure. Some models cap
                # non-streaming output far below MAX_NONSTREAMING_TOKENS.
                #
                # A ValueError is not proof of that, though: the SDK also raises
                # one when a proxy answers with a content type it cannot parse. Say
                # what is known, and only advise lowering the cap when the message
                # is about the token budget.
                reason = str(exc).lower()
                budget = "max_tokens" in reason or "streaming" in reason
                raise EngineError(
                    f"Anthropic rejected the request for {self.name} before sending it: {exc}"
                    + ("\n  If this model caps non-streaming output, lower it:  "
                       "CALIBRATOR_ANTHROPIC_MAX_TOKENS=8192 calibrate <command> …" if budget else "")
                ) from exc
            except Exception as exc:  # map SDK errors (auth/rate-limit/conn/timeout) to friendly text
                friendly = _friendly_anthropic_error(exc, self.name)
                if friendly is not None:
                    raise friendly from exc
                raise
            if resp.stop_reason == "refusal":
                raise EngineError(f"Claude declined the request ({self.name}).")
            if resp.stop_reason == "max_tokens":
                # Only ever suggest a limit the SDK will accept — the old advice
                # (always double) printed 32000 at the default, which every
                # subsequent call would have failed on.
                headroom = min(self.max_tokens * 2, MAX_NONSTREAMING_TOKENS)
                raise EngineError(
                    f"Claude response truncated at max_tokens={self.max_tokens} ({self.name}).\n"
                    + (f"  Try raising the limit:  CALIBRATOR_ANTHROPIC_MAX_TOKENS={headroom} "
                       "calibrate <command> …"
                       if headroom > self.max_tokens else
                       "  That is this tool's ceiling for one non-streaming request — split the "
                       "input, or raise it and stream if your model supports a larger budget.")
                )
            # content can be None/empty on a malformed or OpenAI-compatible proxy
            # response — don't let that raise a raw TypeError.
            blocks = resp.content or []
            # `or ""` guards a text block whose .text is None — the contract is a str.
            return next((b.text or "" for b in blocks if getattr(b, "type", None) == "text"), "")

        return call_json(_call) if schema is not None else _call()
