"""Deterministic eval checks — the first grading layer: exact, cheap, no LLM.

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
        except Exception as exc:
            # The timeout bounds TIME, not MEMORY. `regex`'s recursive patterns
            # ((?R), (?1)) can allocate until the allocator gives up and raise
            # MemoryError — an ordinary Exception that escaped both handlers
            # above, every caller, and `run_eval`, destroying the whole graded
            # run and 502-ing every answer under `calibrate run --guard`. One
            # owner-authored pattern must fail its own criterion, not the run.
            return False, f"regex {value!r} could not be evaluated: {type(exc).__name__}: {exc}"
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


def run_check_turns(check: Check, replies: list[str]) -> tuple[bool, str]:
    """Grade a conversation's assistant replies → (passed, rationale).

    Each reply is a whole answer, so the length and emptiness checks apply to
    every reply on its own rather than to their concatenation: three 49-character
    replies do not violate ``max_chars 50``, and a turn that says nothing fails
    ``non_empty`` however chatty its neighbours were. That is also the granularity
    the runtime guard enforces on a live answer, so what the eval certifies is
    what serving allows.

    ``contains`` is the exception: it asks whether the conversation *carries* a
    term, which any one turn can settle — a closing "happy to help!" must not
    fail a criterion the substantive turn already satisfied.

    ``regex`` deliberately does NOT join it, positive-sounding though it is. It
    is the only kind that can express a pattern-based BAN or a per-answer format
    rule (``not_contains`` takes a literal substring only), so reading it as "any
    turn carries it" would pass a conversation in which a reply broke the rule.
    Per-reply is also what ``runtime._guard_failures`` enforces on a live answer,
    and the eval must not certify what serving would flag.
    """
    if len(replies) <= 1:  # single-turn: graded exactly as it always has been
        return run_check(check, replies[0] if replies else "")
    verdicts = [run_check(check, reply) for reply in replies]
    if check.kind == "contains":
        for i, (ok, why) in enumerate(verdicts, start=1):
            if ok:
                return True, f"turn {i}: {why}"
        return verdicts[0]  # no turn carried it — every rationale says the same thing
    for i, (ok, why) in enumerate(verdicts, start=1):
        if not ok:
            return False, f"turn {i}: {why}"
    # Never a summed measurement here: no single answer ever had it.
    return True, f"all {len(replies)} turns pass {check.kind}" + (f" {check.value!r}" if check.value else "")
