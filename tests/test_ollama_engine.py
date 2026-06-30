"""OllamaEngine adapter — verified with mocked httpx (no server needed).

Closes a real coverage gap: the local engine that powers the tool out-of-the-box
had no unit tests. Covers the happy paths, the JSON-schema path through
call_json, and the error paths (connection refused, HTTP error, malformed
response) — including the clear-error fix for missing response keys.
"""

import httpx
import pytest

import calibrator.engines.ollama as ollama_mod
from calibrator.engines.ollama import OllamaEngine


class FakeResp:
    def __init__(self, data=None, ok=True):
        self._data = {} if data is None else data
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise httpx.HTTPStatusError(
                "500", request=httpx.Request("POST", "http://localhost:11434/api/chat"),
                response=httpx.Response(500))

    def json(self):
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


def test_http_error_propagates(monkeypatch):
    _patch(monkeypatch, FakeResp(ok=False))
    with pytest.raises(httpx.HTTPStatusError):
        OllamaEngine("gemma").complete("hi")


def test_schema_with_unparseable_output_retries_then_raises(monkeypatch):
    _patch(monkeypatch, FakeResp({"message": {"content": "not json at all"}}))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        OllamaEngine("gemma").complete("hi", schema={"type": "object"})
