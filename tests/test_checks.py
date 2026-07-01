"""Deterministic eval checks (§9 layer 1)."""

from calibrator.checks import run_check
from calibrator.models import Check


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

    from calibrator.checks import REGEX_TIMEOUT

    start = time.perf_counter()
    ok, why = run_check(Check(kind="regex", value=r"(a|aa)+$"), "a" * 40 + "!")
    elapsed = time.perf_counter() - start

    assert ok is False
    assert "timed out" in why
    assert elapsed < REGEX_TIMEOUT + 2.0  # bounded — nowhere near an unbounded hang
