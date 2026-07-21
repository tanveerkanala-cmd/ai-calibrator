"""`calibrate run` — serve the calibrated AI itself (OpenAI-compatible).

The export bundle is a *dead artifact*; this is the live one. It serves
``POST /v1/chat/completions`` (plus ``GET /v1/models``), so anything that speaks
the OpenAI protocol — chat UIs, SDKs, other tools — talks to your calibrated AI
by swapping one ``base_url``. Three honesty properties make it more than a proxy:

- **What you tested is what you serve.** The system prompt is compiled from the
  spec, and live conversations are transcript-encoded with the SAME function the
  eval harness uses for multi-turn tests (:func:`calibrator.eval.conversation_prompt`).
- **Boot gate.** ``calibrate run`` checks the persisted `ci` verdict first: a
  failing gate refuses to serve (``--force`` to override); a stale or missing
  gate serves with a loud warning. An AI that can't prove it follows its rules
  shouldn't quietly pretend it does.
- **``--guard``** re-runs the spec's deterministic checks on every LIVE answer
  before returning it: a violating answer is retried once, and still-failing
  responses are flagged (``x-calibrate-guard`` header) and logged to
  ``logs/guard.jsonl``. The tests never stop running.

Client ``system`` messages are ignored by design — the calibrated spec is the
authority. No auth: bind localhost only (the CLI warns otherwise). A shared
Host/Origin guard (webguard.py) blocks browser CSRF and DNS rebinding — the
two attacks localhost binding does NOT stop.

NOTE: no ``from __future__ import annotations`` here — FastAPI must resolve the
endpoint annotations (``Request``/``Response``) at runtime, and those are
function-local imports (kept lazy so importing this module never requires the
``api`` extra). Stringified annotations would make FastAPI read them as query
parameters.
"""

import itertools
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from .checks import run_check
from .coerce import as_str, is_str
from .compile import render_system_prompt
from .engines.base import Engine
from .eval import conversation_prompt
from .models import Project
from .store import load_project
from .webguard import install_guard

MAX_CHAT_CHARS = 200_000   # total message content cap — protects the engine
RECENT_COMPLETIONS = 512   # how many completions stay addressable by id for /v1/feedback


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _content_text(content) -> str | None:
    """Message content → text. Accepts plain strings AND the OpenAI
    content-parts form (``[{"type": "text", "text": …}, …]``, which many SDK
    wrappers always send). Returns None for anything non-textual."""
    if is_str(content):
        return content
    if isinstance(content, list):
        parts = [p for p in content if isinstance(p, dict)]
        texts = [p.get("text") for p in parts if p.get("type") == "text" and is_str(p.get("text"))]
        if parts and len(texts) == len(parts) == len(content):
            return "".join(texts)
    return None


def encode_messages(messages: list[dict]) -> str:
    """OpenAI-style messages → the transcript-encoded prompt the engine sees.

    Honesty rules (a wrong answer beats a silently wrong context):
    - ``system`` entries are dropped BY DESIGN — the spec's compiled prompt is
      the authority (documented).
    - tool/function-calling messages are REJECTED, not dropped: this endpoint
      can't execute tools, and losing a tool result from the context would
      corrupt the conversation invisibly.
    - non-text content (images, audio) is rejected; text content-parts are
      accepted and joined. The final message must be from the user."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    turns: list[tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object with role and content")
        role = m.get("role")
        if role == "system":
            continue  # the calibrated spec is the authority
        if role in ("tool", "function") or m.get("tool_calls") or m.get("function_call"):
            raise ValueError("function/tool calling is not supported by this calibrated endpoint — "
                             "send plain user/assistant text messages")
        if role not in ("user", "assistant"):
            raise ValueError(f"unsupported message role {role!r}")
        text = _content_text(m.get("content"))
        if text is None or not text.strip():
            raise ValueError("message content must be non-empty text "
                             "(image/audio content parts are not supported)")
        turns.append((role, text))
    if not turns or turns[-1][0] != "user":
        raise ValueError("the last message must be from the user")
    if sum(len(t) for _, t in turns) > MAX_CHAT_CHARS:
        raise ValueError(f"conversation too large (>{MAX_CHAT_CHARS} characters)")
    history = [f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in turns[:-1]]
    return conversation_prompt(history, turns[-1][1])


def _guard_checks(project: Project) -> list[tuple[str, object]]:
    if project.spec is None:
        return []
    return [(c.id, c.check) for c in project.spec.eval_criteria if c.check is not None]


def _log_guard(project_dir: Path, record: dict) -> None:
    from .store import open_private_append
    try:
        d = project_dir / "logs"
        d.mkdir(exist_ok=True, mode=0o700)
        with open_private_append(d / "guard.jsonl") as fh:  # 0600 — holds live answers
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:  # guarding must never take the endpoint down
        pass


def create_ai_app(project_dir: str | Path, *, engine: Engine | None = None, guard: bool = False,
                  allowed_hosts: list[str] | None = None):
    """Build the FastAPI app serving the calibrated AI at ``/v1``.

    ``engine`` overrides the subject binding (used by tests); the project is
    loaded once — restart to pick up spec changes (the boot gate re-checks)."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Response
        from fastapi.responses import StreamingResponse
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Serving needs the `api` extra:  pip install -e '.[api]'") from exc

    directory = Path(project_dir)
    project = load_project(directory)
    if project.spec is None:
        raise ValueError("Nothing to serve — run `calibrate compile` (or `import`) first.")
    system = render_system_prompt(project.spec)
    if engine is None:
        from .engines import get_engine
        engine = get_engine(project.engines.subject)
    checks = _guard_checks(project) if guard else []

    from .ci import certification_status
    status, detail = certification_status(project, directory)

    app = FastAPI(title=f"{project.name} — calibrated AI", docs_url=None, redoc_url=None)
    # Same Host/Origin guard as `calibrate serve` — localhost binding alone
    # doesn't stop CSRF (a no-preflight cross-origin POST would burn the
    # owner's engine API key) or DNS rebinding. See webguard.py.
    install_guard(app, allowed_hosts)
    # id → {"turns": [user turns], "output": answer}; lets /v1/feedback reference
    # a completion by id (the flywheel's capture point).
    recent: OrderedDict[str, dict] = OrderedDict()
    seq = itertools.count()  # monotonic — unique ids even at same-second timestamps

    def _remember(cid: str, turns: list[str], output: str) -> None:
        recent[cid] = {"turns": turns, "output": output}
        while len(recent) > RECENT_COMPLETIONS:
            recent.popitem(last=False)

    @app.get("/")
    def root():
        """The endpoint self-describes its certification."""
        return {"name": project.name, "goal": project.goal, "engine": engine.name,
                "certification": status, "detail": detail, "guard": guard,
                "openai_base_url": "/v1", "feedback": "POST /v1/feedback"}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": project.name, "object": "model", "created": _now(),
                          "owned_by": "ai-calibrator"}]}

    def _complete(messages: list[dict]) -> tuple[str, dict]:
        from . import rag
        prompt = encode_messages(messages)
        # RAG: augment the system with chunks retrieved for the latest user turn,
        # exactly as run_eval does — so the served AI matches the tested one.
        query = _content_text(messages[-1].get("content")) if messages else None
        eff_system = rag.augment_system(system, directory, query or "")
        content = as_str(engine.complete(prompt, system=eff_system)).strip()
        guard_state: dict = {}
        if checks:
            failed = [cid for cid, chk in checks if not run_check(chk, content)[0]]
            if failed:  # one retry — models are stochastic; then flag, never block
                content = as_str(engine.complete(prompt, system=eff_system)).strip()
                still = [cid for cid, chk in checks if not run_check(chk, content)[0]]
                if still:
                    guard_state = {"guard": "failed", "criteria": still}
                    _log_guard(directory, {
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "failed": still, "output": content})
                else:
                    guard_state = {"guard": "passed-after-retry", "criteria": failed}
            else:
                guard_state = {"guard": "passed", "criteria": []}
        return content, guard_state

    def _guard_header(response: Response, guard_state: dict) -> None:
        if guard_state:
            response.headers["x-calibrate-guard"] = guard_state["guard"] + (
                ":" + ",".join(guard_state["criteria"]) if guard_state["criteria"] else "")

    @app.post("/v1/chat/completions")
    async def chat(request: Request, response: Response):
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "request body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "request body must be a JSON object")
        # Tool/function calling isn't supported by this endpoint — reject it
        # explicitly rather than silently dropping it (an integrator relying on
        # tools would otherwise get a plain completion and never know).
        if body.get("tools") or body.get("functions"):
            raise HTTPException(400, "tool/function calling is not supported by this endpoint")
        try:
            # The engine call is blocking; run it in a threadpool so one in-flight
            # completion (often 30-60s) doesn't freeze every other request on the
            # event loop — the endpoint must stay concurrent.
            content, guard_state = await run_in_threadpool(_complete, body.get("messages"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:  # engine failure → OpenAI-style 502, not a traceback
            raise HTTPException(502, f"engine error: {exc}")

        created = _now()
        # A per-request monotonic counter — NOT len(recent), which plateaus at the
        # ring-buffer cap and would collide for two same-second requests, routing
        # feedback to the wrong conversation.
        cid = f"chatcmpl-{project.name}-{created}-{next(seq)}"
        user_turns = [text for m in body.get("messages", [])
                      if isinstance(m, dict) and m.get("role") == "user"
                      and (text := _content_text(m.get("content"))) is not None]
        _remember(cid, user_turns, content)
        if body.get("stream"):
            # The engine interface is non-streaming; emit valid SSE chunks so
            # streaming clients (most chat UIs) work unchanged.
            def sse():
                first = {"id": cid, "object": "chat.completion.chunk", "created": created,
                         "model": project.name,
                         "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                yield f"data: {json.dumps(first)}\n\n"
                for i in range(0, max(len(content), 1), 120):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": project.name,
                             "choices": [{"index": 0, "delta": {"content": content[i:i + 120]},
                                          "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                last = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": project.name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(last)}\n\n"
                yield "data: [DONE]\n\n"
            resp = StreamingResponse(sse(), media_type="text/event-stream")
            _guard_header(resp, guard_state)
            return resp

        _guard_header(response, guard_state)
        # Approximate prompt size for the usage block — use the real text of each
        # message (content-parts included), not str() of the raw list which would
        # count the JSON structure. Best-effort estimate only.
        prompt_chars = sum(len(_content_text(m.get("content")) or "")
                           for m in (body.get("messages") or []) if isinstance(m, dict))
        return {
            "id": cid, "object": "chat.completion", "created": created, "model": project.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": max(1, prompt_chars // 4),
                      "completion_tokens": max(1, len(content) // 4),
                      "total_tokens": max(2, (prompt_chars + len(content)) // 4)},
        }

    @app.post("/v1/feedback")
    async def feedback(request: Request):
        """The flywheel's capture point: thumbs-up / thumbs-down on a live answer.

        Reference a completion by ``completion_id`` (from /v1/chat/completions),
        or pass ``input``/``turns`` + ``output`` explicitly. ``verdict`` is
        ``up`` | ``down``; a ``correction`` (what the answer SHOULD have been)
        and a ``reason`` are welcome on downs. `calibrate absorb` turns these
        into spec examples + pinned regression tests."""
        from .flywheel import append_feedback

        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "request body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "request body must be a JSON object")
        verdict = body.get("verdict")
        if verdict not in ("up", "down"):
            raise HTTPException(400, "verdict must be 'up' or 'down'")

        cid = body.get("completion_id")
        if is_str(cid) and cid:
            known = recent.get(cid)
            if known is None:
                raise HTTPException(404, "unknown or expired completion_id — "
                                         "pass `input` (or `turns`) and `output` explicitly")
            turns, output = known["turns"], known["output"]
        else:
            output = as_str(body.get("output"))
            raw = body.get("turns") if isinstance(body.get("turns"), list) else [body.get("input")]
            turns = [t for t in raw if is_str(t) and t.strip()]
            if not turns or not output.strip():
                raise HTTPException(400, "pass a completion_id, or `input` (or `turns`) and `output`")

        correction, reason = body.get("correction"), body.get("reason")
        # append_feedback takes the blocking project lock — off the event loop so
        # a concurrent `calibrate absorb` holding the lock can't freeze the server.
        await run_in_threadpool(append_feedback, directory, {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "turns": turns, "output": output, "verdict": verdict,
            "correction": correction if is_str(correction) else None,
            "reason": reason if is_str(reason) else None,
        })
        return {"recorded": True, "next": "run `calibrate absorb` to pin this into the spec + tests"}

    return app
