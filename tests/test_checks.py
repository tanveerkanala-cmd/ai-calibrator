"""Deterministic eval checks (the code-graded layer)."""

from ai_calibrator.checks import run_check
from ai_calibrator.models import Check


def test_contains():
    assert run_check(Check(kind="contains", value="30-day"), "our 30-DAY policy")[0] is True
    assert run_check(Check(kind="contains", value="refund"), "no mention")[0] is False


def test_not_contains():
    assert run_check(Check(kind="not_contains", value="guarantee"), "we promise nothing")[0] is True
    assert run_check(Check(kind="not_contains", value="guarantee"), "a guarantee!")[0] is False


def test_regex_and_invalid_regex():
    assert run_check(Check(kind="regex", value=r"\d+ days"), "within 30 days")[0] is True
    ok, why = run_check(Check(kind="regex", value="[unclosed"), "x")
    assert ok is False and "invalid regex" in why


def test_max_and_min_chars():
    assert run_check(Check(kind="max_chars", value="10"), "short")[0] is True
    assert run_check(Check(kind="max_chars", value="3"), "toolong")[0] is False
    assert run_check(Check(kind="min_chars", value="3"), "ok")[0] is False
    assert run_check(Check(kind="max_chars", value="notint"), "x")[0] is False  # non-numeric → fail cleanly


def test_non_empty():
    assert run_check(Check(kind="non_empty"), "hi")[0] is True
    assert run_check(Check(kind="non_empty"), "   ")[0] is False


def test_regex_catastrophic_pattern_times_out_instead_of_hanging():
    """A catastrophic-backtracking pattern must fail fast, not hang the eval
    (regression: stdlib re hung ~10s on such patterns; the regex-engine timeout bounds it)."""
    import time

    from ai_calibrator.checks import REGEX_TIMEOUT

    start = time.perf_counter()
    ok, why = run_check(Check(kind="regex", value=r"(a|aa)+$"), "a" * 40 + "!")
    elapsed = time.perf_counter() - start

    assert ok is False
    assert "timed out" in why
    assert elapsed < REGEX_TIMEOUT + 2.0  # bounded — nowhere near an unbounded hang


def test_length_checks_exact_boundary():
    """max_chars '<=' vs '<' (and min_chars '>=' vs '>') — the boundary itself
    must pass."""
    assert run_check(Check(kind="max_chars", value="5"), "12345")[0] is True    # == limit passes
    assert run_check(Check(kind="max_chars", value="5"), "123456")[0] is False
    assert run_check(Check(kind="min_chars", value="5"), "12345")[0] is True    # == limit passes
    assert run_check(Check(kind="min_chars", value="5"), "1234")[0] is False


def test_run_check_fails_the_criterion_when_the_regex_engine_raises():
    """The match timeout bounds TIME, not MEMORY.

    A recursive pattern can allocate until the allocator gives up and raises
    MemoryError — an ordinary Exception that escaped the `regex.error` and
    `TimeoutError` handlers, every caller, and `run_eval`, taking the whole
    graded run down with it and 502-ing every answer under `run --guard`. One
    owner-authored pattern must fail its own criterion, not the run.

    Monkeypatched rather than reproduced: the real trigger allocates ~550MB.
    """
    import regex as regex_mod

    from ai_calibrator.checks import run_check
    from ai_calibrator.models import Check

    real_search = regex_mod.search

    def _boom(*a, **kw):
        raise MemoryError()

    regex_mod.search = _boom
    try:
        ok, detail = run_check(Check(kind="regex", value="(?R)"), "some output")
    finally:
        regex_mod.search = real_search

    assert ok is False
    assert "MemoryError" in detail and "could not be evaluated" in detail
