"""Deterministic eval checks — §9 layer 1: exact, cheap, no LLM.

A criterion with a ``check`` is graded by code, not the judge: free and perfectly
reliable for objectively-verifiable behavior (a required or forbidden term, a
length limit, a format/regex, non-empty). This is the reliability floor under the
noisy LLM-judge — use it wherever a criterion can be made objective.
"""

from __future__ import annotations

import re

from .models import Check


def _as_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def run_check(check: Check, output: str) -> tuple[bool, str]:
    """Grade ``output`` against a deterministic ``check`` → (passed, rationale)."""
    text = output or ""
    kind, value = check.kind, check.value

    if kind == "contains":
        ok = value.lower() in text.lower()
        return ok, f"{'contains' if ok else 'missing required'} {value!r}"
    if kind == "not_contains":
        ok = value.lower() not in text.lower()
        return ok, f"{'absent' if ok else 'contains forbidden'} {value!r}"
    if kind == "regex":
        try:
            ok = re.search(value, text) is not None
        except re.error as exc:
            return False, f"invalid regex {value!r}: {exc}"
        return ok, f"regex {value!r} {'matched' if ok else 'did not match'}"
    if kind == "max_chars":
        limit = _as_int(value)
        if limit is None:
            return False, f"max_chars needs an integer, got {value!r}"
        ok = len(text) <= limit
        return ok, f"length {len(text)} {'<=' if ok else '>'} {limit}"
    if kind == "min_chars":
        limit = _as_int(value)
        if limit is None:
            return False, f"min_chars needs an integer, got {value!r}"
        ok = len(text) >= limit
        return ok, f"length {len(text)} {'>=' if ok else '<'} {limit}"
    if kind == "non_empty":
        ok = bool(text.strip())
        return ok, "non-empty" if ok else "empty output"
    return False, f"unknown check kind {kind!r}"  # unreachable via the Literal type
