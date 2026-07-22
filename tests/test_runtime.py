"""`calibrate run` — the OpenAI-compatible runtime serving the calibrated AI."""

import json

import pytest

pytest.importorskip("fastapi")  # runtime needs the `api` extra

from fastapi.testclient import TestClient  # noqa: E402

from ai_calibrator.models import BehaviorSpec, Check, EvalCriterion, Project, Weight  # noqa: E402
from ai_calibrator.runtime import create_ai_app, encode_messages  # noqa: E402
from ai_calibrator.store import save_project  # noqa: E402


class RecordingEngine:
    name = "fake@test"

    def __init__(self, replies=("the answer",)):
        self.replies = list(replies)
        self.calls = []  # (prompt, system)

    def complete(self, prompt, *, system=None, schema=None):
        self.calls.append((prompt, system))
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


def _seed(tmp_path, check=None):
    p = Project(name="my-ai", goal="answer return questions")
    p.spec = BehaviorSpec(goal="answer return questions",
                          standards=["Always cite the 30-day window."],
                          eval_criteria=[EvalCriterion(id="c1", description="cites the window",
                                                       weight=Weight.HIGH, check=check)])
    save_project(p, tmp_path)
    return p


def _client(tmp_path, engine, guard=False):
    return TestClient(create_ai_app(tmp_path, engine=engine, guard=guard))


def test_chat_completion_shape_and_calibrated_system_prompt(tmp_path):
    _seed(tmp_path)
    eng = RecordingEngine(["Sure — 30 days."])
    r = _client(tmp_path, eng).post("/v1/chat/completions", json={
        "model": "whatever", "messages": [{"role": "user", "content": "how long?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "my-ai"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Sure — 30 days."}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] >= 2
    # the calibrated spec IS the system prompt
    prompt, system = eng.calls[0]
    assert "Always cite the 30-day window." in system
    assert prompt == "User: how long?\nAssistant:"


def test_history_encoding_matches_eval_harness(tmp_path):
    """What you tested is what you serve: live chats use conversation_prompt."""
    from ai_calibrator.eval import conversation_prompt

    _seed(tmp_path)
    eng = RecordingEngine()
    _client(tmp_path, eng).post("/v1/chat/completions", json={"messages": [
        {"role": "system", "content": "you are a pirate"},   # ignored — spec wins
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how long?"},
    ]})
    prompt, system = eng.calls[0]
    assert prompt == conversation_prompt(["User: hi", "Assistant: hello"], "how long?")
    assert "pirate" not in (system or "")  # client system message dropped


def test_bad_requests_are_400_not_500(tmp_path):
    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine())
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}).status_code == 400
    assert c.post("/v1/chat/completions", json={"messages": "nope"}).status_code == 400
    assert c.post("/v1/chat/completions", content=b"not json",
                  headers={"Content-Type": "application/json"}).status_code == 400
    big = "x" * 200_001
    assert c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": big}]}).status_code == 400


def test_engine_failure_is_502(tmp_path):
    _seed(tmp_path)

    class Boom:
        name = "boom@test"

        def complete(self, *a, **k):
            raise RuntimeError("provider down")

    r = _client(tmp_path, Boom()).post("/v1/chat/completions",
                                       json={"messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 502 and "provider down" in r.json()["detail"]


def test_models_and_root_endpoints(tmp_path):
    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine())
    models = c.get("/v1/models").json()
    assert models["data"][0]["id"] == "my-ai"
    root = c.get("/").json()
    assert root["certification"] == "none" and root["openai_base_url"] == "/v1"


def test_streaming_emits_valid_sse(tmp_path):
    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine(["hello " * 60]))  # long enough for >1 chunk
    r = c.post("/v1/chat/completions", json={"stream": True,
                                             "messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(ch["object"] == "chat.completion.chunk" for ch in chunks)
    content = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
    assert content == ("hello " * 60).strip()  # output is stripped, same as the eval harness
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_guard_retries_then_flags(tmp_path):
    """--guard re-checks live answers: violating answer → one retry → flagged header + log."""
    _seed(tmp_path, check=Check(kind="contains", value="30"))

    # fails, then fixes itself on retry
    eng = RecordingEngine(["no window mentioned", "the window is 30 days"])
    r = _client(tmp_path, eng, guard=True).post("/v1/chat/completions",
                                                json={"messages": [{"role": "user", "content": "q"}]})
    assert len(eng.calls) == 2
    assert r.json()["choices"][0]["message"]["content"] == "the window is 30 days"
    assert r.headers["x-calibrate-guard"].startswith("passed-after-retry")

    # fails twice → flagged (never blocked) + logged
    eng2 = RecordingEngine(["nope", "still nope"])
    r2 = _client(tmp_path, eng2, guard=True).post("/v1/chat/completions",
                                                  json={"messages": [{"role": "user", "content": "q"}]})
    assert r2.status_code == 200  # flag, don't block
    assert r2.headers["x-calibrate-guard"] == "failed:c1"
    logged = (tmp_path / "logs" / "guard.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(logged[-1])["failed"] == ["c1"]

    # passing answer → passed header, single call
    eng3 = RecordingEngine(["30 days, always"])
    r3 = _client(tmp_path, eng3, guard=True).post("/v1/chat/completions",
                                                  json={"messages": [{"role": "user", "content": "q"}]})
    assert r3.headers["x-calibrate-guard"] == "passed" and len(eng3.calls) == 1


def test_guard_off_never_rechecks(tmp_path):
    _seed(tmp_path, check=Check(kind="contains", value="30"))
    eng = RecordingEngine(["no mention"])
    r = _client(tmp_path, eng, guard=False).post("/v1/chat/completions",
                                                 json={"messages": [{"role": "user", "content": "q"}]})
    assert "x-calibrate-guard" not in r.headers and len(eng.calls) == 1


def test_encode_messages_contract():
    with pytest.raises(ValueError):
        encode_messages([])
    with pytest.raises(ValueError):
        encode_messages([{"role": "assistant", "content": "a"}])
    assert encode_messages([{"role": "user", "content": "hi"}]) == "User: hi\nAssistant:"


def test_feedback_by_completion_id_and_explicit(tmp_path):
    """The flywheel capture point: feedback lands durably in logs/feedback.jsonl."""
    from ai_calibrator.flywheel import read_feedback

    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine(["Any time!"]))
    cid = c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "return after 40 days?"}]}).json()["id"]

    r = c.post("/v1/feedback", json={"completion_id": cid, "verdict": "down",
                                     "correction": "No — 30 days.", "reason": "wrong"})
    assert r.status_code == 200 and r.json()["recorded"] is True
    rec = read_feedback(tmp_path)[0]
    assert rec["turns"] == ["hi", "return after 40 days?"]     # user turns only
    assert rec["output"] == "Any time!" and rec["correction"] == "No — 30 days."

    # explicit body (no id needed — e.g. after a server restart)
    r2 = c.post("/v1/feedback", json={"input": "q", "output": "a", "verdict": "up"})
    assert r2.status_code == 200
    assert read_feedback(tmp_path)[1]["turns"] == ["q"]

    # friendly failures
    assert c.post("/v1/feedback", json={"completion_id": "nope", "verdict": "down"}).status_code == 404
    assert c.post("/v1/feedback", json={"input": "q", "output": "a", "verdict": "meh"}).status_code == 400
    assert c.post("/v1/feedback", json={"verdict": "down"}).status_code == 400


def test_flywheel_end_to_end_chat_feedback_absorb(tmp_path):
    """use → flag → absorb → pinned: the full loop, and certification goes stale."""
    from ai_calibrator.ci import certification_status, save_gate, CiResult, CiStage
    from ai_calibrator.flywheel import absorb_feedback
    from ai_calibrator.store import load_project

    p = _seed(tmp_path)
    # fake a passing gate for the CURRENT config so we can watch it go stale
    save_gate(p, CiResult(stages=[CiStage("lint", "pass", "")], run_id="run-0001",
                          pass_rate=1.0, weighted_score=1.0), tmp_path)
    assert certification_status(p, tmp_path)[0] == "pass"

    c = _client(tmp_path, RecordingEngine(["Sure, any time!"]))
    cid = c.post("/v1/chat/completions",
                 json={"messages": [{"role": "user", "content": "return after 40 days?"}]}).json()["id"]
    c.post("/v1/feedback", json={"completion_id": cid, "verdict": "down",
                                 "correction": "No — the window is 30 days."})

    project = load_project(tmp_path)
    result = absorb_feedback(project, tmp_path)
    assert result.tests_added == 1 and result.examples_added == 1
    assert project.tests[-1].input == "return after 40 days?"
    assert project.spec.examples[-1].good_output == "No — the window is 30 days."
    # the AI learned something → its old certification no longer applies
    assert certification_status(project, tmp_path)[0] == "stale"


def test_tool_calling_rejected_not_silently_dropped(tmp_path):
    """Tool/function messages were once silently dropped from context. They
    must be rejected loudly — a lost tool result corrupts the conversation
    invisibly."""
    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine())
    r = c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "call the tool"},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "so?"}]})
    assert r.status_code == 400 and "tool" in r.json()["detail"]

    # assistant messages carrying tool_calls are the same protocol family
    r2 = c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        {"role": "user", "content": "so?"}]})
    assert r2.status_code == 400

    # unknown roles are loud too
    r3 = c.post("/v1/chat/completions", json={"messages": [{"role": "wizard", "content": "q"}]})
    assert r3.status_code == 400 and "role" in r3.json()["detail"]


def test_content_parts_form_is_accepted(tmp_path):
    """Many OpenAI SDK wrappers always send content as [{"type":"text",...}]."""
    _seed(tmp_path)
    eng = RecordingEngine(["ok"])
    c = _client(tmp_path, eng)
    r = c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": [{"type": "text", "text": "how "}, {"type": "text", "text": "long?"}]}]})
    assert r.status_code == 200
    assert eng.calls[0][0] == "User: how long?\nAssistant:"

    # non-text parts are rejected with a reason, not silently dropped
    r2 = c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x"}}]}]})
    assert r2.status_code == 400 and "text" in r2.json()["detail"]

    # null content is loud, not a silent drop
    r3 = c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": None},
        {"role": "user", "content": "so?"}]})
    assert r3.status_code == 400


def test_completion_ids_unique_past_ring_buffer(tmp_path):
    """Ids derived from len(recent) plateau at the 512 cap and would collide
    within a second — routing feedback to the wrong conversation."""
    from ai_calibrator.runtime import RECENT_COMPLETIONS

    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine(["ok"]))
    ids = set()
    for _ in range(RECENT_COMPLETIONS + 25):   # push well past the ring cap
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
        ids.add(r.json()["id"])
    assert len(ids) == RECENT_COMPLETIONS + 25  # every id distinct


def test_cross_origin_post_is_blocked(tmp_path):
    """The serving endpoint spends the owner's engine key — a malicious page must
    not be able to drive it via a no-preflight cross-origin POST (CSRF)."""
    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine(["ok"]))
    payload = {"messages": [{"role": "user", "content": "q"}]}
    # no Origin (scripts/SDKs/TestClient) and same-origin Origin are allowed
    assert c.post("/v1/chat/completions", json=payload).status_code == 200
    assert c.post("/v1/chat/completions", json=payload,
                  headers={"Origin": "http://testserver"}).status_code == 200
    # cross-origin Origin on a mutating request is rejected (CSRF guard)
    assert c.post("/v1/chat/completions", json=payload,
                  headers={"Origin": "https://evil.example"}).status_code == 403


def test_foreign_host_is_blocked(tmp_path):
    """DNS rebinding: a Host outside the allowlist is rejected outright."""
    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine(["ok"]))
    assert c.get("/").status_code == 200  # default Host "testserver" allowed
    assert c.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_csrf_guard_stays_on_when_host_is_widened(tmp_path):
    """--host <addr> widens the allowlist to that one host; the cross-origin
    guard must survive the widening (mirrors the serve/api guarantee)."""
    _seed(tmp_path)
    app = create_ai_app(tmp_path, engine=RecordingEngine(["ok"]), allowed_hosts=["192.168.1.50"])
    client = TestClient(app, base_url="http://192.168.1.50")
    payload = {"messages": [{"role": "user", "content": "q"}]}
    assert client.post("/v1/chat/completions", json=payload,
                       headers={"Origin": "http://192.168.1.50:8600"}).status_code == 200
    assert client.post("/v1/chat/completions", json=payload,
                       headers={"Origin": "https://evil.example"}).status_code == 403


def test_tools_param_rejected_with_400(tmp_path):
    """Tool/function calling isn't supported; a top-level `tools` must be a clear
    400, not a silently-dropped 200 (the documented contract)."""
    _seed(tmp_path)
    c = _client(tmp_path, RecordingEngine(["ok"]))
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}]})
    assert r.status_code == 400 and "tool" in r.json()["detail"].lower()
    # a plain request (no tools) still works
    assert c.post("/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 200
