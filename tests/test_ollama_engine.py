"""OllamaEngine adapter — verified with mocked httpx (no server needed).

Closes a real coverage gap: the local engine that powers the tool out-of-the-box
had no unit tests. Covers the happy paths, the JSON-schema path through
call_json, and the error paths (connection refused, HTTP error, malformed
response) — including the clear-error fix for missing response keys.
"""

import httpx
import pytest

import ai_calibrator.engines.ollama as ollama_mod
from ai_calibrator.engines.ollama import OllamaEngine


class FakeResp:
    def __init__(self, data=None, ok=True, status=500, raw=None):
        self._data = {} if data is None else data
        self._ok = ok
        self._status = status
        self.text = raw if raw is not None else ""

    def raise_for_status(self):
        if not self._ok:
            raise httpx.HTTPStatusError(
                str(self._status), request=httpx.Request("POST", "http://localhost:11434/api/chat"),
                response=httpx.Response(self._status, text=self.text))

    def json(self):
        if self.text and self._data == {}:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._data


def _patch(monkeypatch, resp=None, exc=None):
    def fake_post(*a, **k):
        if exc is not None:
            raise exc
        return resp
    monkeypatch.setattr(ollama_mod.httpx, "post", fake_post)


def test_complete_plain_text(monkeypatch):
    _patch(monkeypatch, FakeResp({"message": {"content": "hello world"}}))
    eng = OllamaEngine("gemma")
    assert eng.complete("hi") == "hello world"
    assert eng.name == "gemma@ollama"


def test_complete_with_schema_parses_json(monkeypatch):
    _patch(monkeypatch, FakeResp({"message": {"content": '{"facts": ["a"], "gaps": []}'}}))
    out = OllamaEngine("gemma").complete("hi", schema={"type": "object"})
    assert out == {"facts": ["a"], "gaps": []}


def test_connect_error_is_friendly(monkeypatch):
    _patch(monkeypatch, exc=httpx.ConnectError("refused"))
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        OllamaEngine("gemma").complete("hi")


def test_missing_message_key_is_clear_error(monkeypatch):
    _patch(monkeypatch, FakeResp({"error": "model not found"}))  # no 'message'
    with pytest.raises(RuntimeError, match="missing message.content"):
        OllamaEngine("gemma").complete("hi")


def test_missing_content_key_is_clear_error(monkeypatch):
    _patch(monkeypatch, FakeResp({"message": {}}))  # no 'content'
    with pytest.raises(RuntimeError, match="missing message.content"):
        OllamaEngine("gemma").complete("hi")


def test_non_dict_message_is_clear_error(monkeypatch):
    _patch(monkeypatch, FakeResp({"message": "oops"}))  # string, not a dict → TypeError path
    with pytest.raises(RuntimeError, match="missing message.content"):
        OllamaEngine("gemma").complete("hi")


def test_http_errors_are_friendly_not_raw(monkeypatch):
    """Audit finding: 4xx/5xx surfaced as raw httpx.HTTPStatusError (and the old
    version of this test ENSHRINED that). Every status must be a RuntimeError
    with an actionable message."""
    for status, expect in [(404, "ollama pull"), (401, "authentication"),
                           (429, "HTTP 429"), (500, "internal error"), (529, "HTTP 529")]:
        _patch(monkeypatch, FakeResp(ok=False, status=status))
        with pytest.raises(RuntimeError, match=expect):
            OllamaEngine("gemma").complete("hi")


def test_timeout_is_friendly(monkeypatch):
    _patch(monkeypatch, exc=httpx.ReadTimeout("timed out"))
    with pytest.raises(RuntimeError, match="did not respond within"):
        OllamaEngine("gemma").complete("hi")
    _patch(monkeypatch, exc=httpx.ConnectTimeout("timed out"))
    with pytest.raises(RuntimeError, match="did not respond within"):
        OllamaEngine("gemma").complete("hi")


def test_invalid_json_body_is_friendly(monkeypatch):
    _patch(monkeypatch, FakeResp(raw="<html>gateway error</html>"))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        OllamaEngine("gemma").complete("hi")


def test_other_transport_errors_are_friendly(monkeypatch):
    _patch(monkeypatch, exc=httpx.RemoteProtocolError("connection torn down"))
    with pytest.raises(RuntimeError, match="request .* failed"):
        OllamaEngine("gemma").complete("hi")


def test_schema_with_unparseable_output_retries_then_raises(monkeypatch):
    _patch(monkeypatch, FakeResp({"message": {"content": "not json at all"}}))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        OllamaEngine("gemma").complete("hi", schema={"type": "object"})


def test_timeout_env_override(monkeypatch):
    """Journey finding: 120s was hardcoded — a slow machine had no knob."""
    monkeypatch.delenv("CALIBRATOR_OLLAMA_TIMEOUT", raising=False)
    assert OllamaEngine("m").timeout == 120.0
    monkeypatch.setenv("CALIBRATOR_OLLAMA_TIMEOUT", "420")
    assert OllamaEngine("m").timeout == 420.0
    monkeypatch.setenv("CALIBRATOR_OLLAMA_TIMEOUT", "junk")
    assert OllamaEngine("m").timeout == 120.0          # junk → default, never crash
    monkeypatch.setenv("CALIBRATOR_OLLAMA_TIMEOUT", "-5")
    assert OllamaEngine("m").timeout == 120.0
    assert OllamaEngine("m", timeout=7.0).timeout == 7.0  # explicit arg still wins


def test_engine_spec_errors_are_actionable():
    from ai_calibrator.engines.base import get_engine
    import pytest as _pytest
    with _pytest.raises(ValueError, match="Valid providers: anthropic, openai, ollama"):
        get_engine("some-model@bogus")
    with _pytest.raises(ValueError, match="no model name"):
        get_engine("@ollama")
