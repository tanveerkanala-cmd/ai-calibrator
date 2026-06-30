"""Engine robustness layer: tolerant JSON parsing + single repair retry."""

import pytest

from calibrator.engines.base import call_json, loads_tolerant, parse_engine_spec


def test_parse_engine_spec():
    assert parse_engine_spec("gpt-4o@openai") == ("gpt-4o", "openai")
    assert parse_engine_spec("claude-opus-4-8@anthropic") == ("claude-opus-4-8", "anthropic")
    assert parse_engine_spec("qwen2.5:14b") == ("qwen2.5:14b", "ollama")  # provider defaults


def test_loads_clean_json():
    assert loads_tolerant('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_loads_markdown_fenced():
    assert loads_tolerant('```json\n{"a": 1}\n```') == {"a": 1}
    assert loads_tolerant('```\n{"a": 2}\n```') == {"a": 2}


def test_loads_prose_wrapped():
    text = 'Sure — here is the result:\n{"ok": true, "n": 3}\nHope that helps!'
    assert loads_tolerant(text) == {"ok": True, "n": 3}


def test_loads_garbage_raises():
    with pytest.raises(ValueError):
        loads_tolerant("not json at all")


def test_call_json_first_try():
    assert call_json(lambda: '{"ok": 1}') == {"ok": 1}


def test_call_json_retries_then_succeeds():
    seq = iter(["oops, not json", '{"ok": true}'])
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        return next(seq)

    assert call_json(call) == {"ok": True}
    assert calls["n"] == 2  # retried exactly once


def test_call_json_gives_up_after_retry():
    with pytest.raises(RuntimeError):
        call_json(lambda: "still not json")


def test_call_json_does_not_retry_on_api_error():
    """Non-parse errors from the call propagate unchanged (not masked as JSON)."""
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise ConnectionError("network down")

    with pytest.raises(ConnectionError):
        call_json(call)
    assert calls["n"] == 1  # no retry on a genuine API/connection error
