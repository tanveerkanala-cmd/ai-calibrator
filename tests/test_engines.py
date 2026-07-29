"""Engine robustness layer: tolerant JSON parsing + single repair retry."""

import pytest

from ai_calibrator.engines.base import call_json, loads_tolerant, parse_engine_spec


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




def test_anthropic_max_tokens_is_env_overridable(monkeypatch):
    """A hard-coded 16k cap with no knob makes the truncation error a dead end —
    the message tells the user to raise it, so there has to be a way to."""
    from ai_calibrator.engines.anthropic import DEFAULT_MAX_TOKENS, _default_max_tokens
    monkeypatch.delenv("CALIBRATOR_ANTHROPIC_MAX_TOKENS", raising=False)
    assert _default_max_tokens() == DEFAULT_MAX_TOKENS
    monkeypatch.setenv("CALIBRATOR_ANTHROPIC_MAX_TOKENS", "20000")
    assert _default_max_tokens() == 20000
    for junk in ("junk", "0", "-5", "1e999"):
        monkeypatch.setenv("CALIBRATOR_ANTHROPIC_MAX_TOKENS", junk)
        assert _default_max_tokens() == DEFAULT_MAX_TOKENS, junk   # junk → default, never crash


def test_anthropic_truncation_error_names_the_knob():
    """The truncation message must name something the reader can actually change;
    "increase max_tokens for this engine" named nothing."""
    from ai_calibrator.engines.anthropic import AnthropicEngine

    class Block:
        type = "text"
        text = "partial"

    class Resp:
        stop_reason = "max_tokens"
        content = [Block()]

    class Messages:
        def create(self, **kwargs):
            return Resp()

    # The anthropic SDK is an optional extra, so build the adapter without __init__.
    eng = AnthropicEngine.__new__(AnthropicEngine)
    eng.name = "claude-x@anthropic"
    eng.model = "claude-x"
    eng.max_tokens = 100
    eng._client = type("Client", (), {"messages": Messages()})()
    with pytest.raises(RuntimeError, match="CALIBRATOR_ANTHROPIC_MAX_TOKENS=200"):
        eng.complete("hi")
