"""Deterministic eval checks — §9 layer 1: exact, cheap, no LLM.

A criterion with a ``check`` is graded by code, not the judge: free and perfectly
reliable for objectively-verifiable behavior (a required or forbidden term, a
length limit, a format/regex, non-empty). This is the reliability floor under the
noisy LLM-judge — use it wherever a criterion can be made objective.

Regex uses the third-party ``regex`` engine (not stdlib ``re``): it resists most
catastrophic backtracking outright, and its cooperative ``timeout`` bounds the
residual cases so an owner-authored pattern can never hang the eval (stdlib ``re``
cannot be interrupted mid-backtrack).
"""

from __future__ import annotations

import regex

from .models import Check

# Wall-clock ceiling for a single regex check. A legitimate check matches in
# microseconds; only catastrophic backtracking approaches this, and we fail it
# with an actionable message rather than hang.
REGEX_TIMEOUT = 1.0


def _as_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def run_check(check: Check, output: str) -> tuple[bool, str]:
    """Grade ``output`` against a deterministic ``check`` → (passed, rationale)."""
    import unicodedata
    # NFC-normalize both sides so a "contains"/"not_contains" check can't be
    # bypassed by emitting a different Unicode normalization of the same glyph
    # (e.g. composed vs decomposed "é") — the check must be truly deterministic.
    text = unicodedata.normalize("NFC", output or "")
    kind = check.kind
    value = unicodedata.normalize("NFC", check.value)

    if kind == "contains":
        ok = value.lower() in text.lower()
        return ok, f"{'contains' if ok else 'missing required'} {value!r}"
    if kind == "not_contains":
        ok = value.lower() not in text.lower()
        return ok, f"{'absent' if ok else 'contains forbidden'} {value!r}"
    if kind == "regex":
        try:
            ok = regex.search(value, text, timeout=REGEX_TIMEOUT) is not None
        except regex.error as exc:
            return False, f"invalid regex {value!r}: {exc}"
        except TimeoutError:
            return False, (f"regex {value!r} timed out (>{REGEX_TIMEOUT:g}s) — likely catastrophic "
                           "backtracking; simplify it (avoid nested quantifiers like (a+)+ or (a|aa)+)")
        return ok, f"regex {value!r} {'matched' if ok else 'did not match'}"
    if kind == "max_chars":
        limit = _as_int(value)
        if limit is None or limit < 0:
            return False, f"max_chars needs a non-negative integer, got {value!r}"
        ok = len(text) <= limit
        return ok, f"length {len(text)} {'<=' if ok else '>'} {limit}"
    if kind == "min_chars":
        limit = _as_int(value)
        if limit is None or limit < 0:
            return False, f"min_chars needs a non-negative integer, got {value!r}"
        ok = len(text) >= limit
        return ok, f"length {len(text)} {'>=' if ok else '<'} {limit}"
    if kind == "non_empty":
        ok = bool(text.strip())
        return ok, "non-empty" if ok else "empty output"
    return False, f"unknown check kind {kind!r}"  # unreachable via the Literal type
