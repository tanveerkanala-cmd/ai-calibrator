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
authority. No auth: bind localhost only (the CLI warns otherwise).

NOTE: no ``from __future__ import annotations`` here — FastAPI must resolve the
endpoint annotations (``Request``/``Response``) at runtime, and those are
function-local imports (kept lazy so importing this module never requires the
``api`` extra). Stringified annotations would make FastAPI read them as query
parameters.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .checks import run_check
from .coerce import as_str, is_str
from .compile import render_system_prompt
from .engines.base import Engine
from .eval import conversation_prompt
from .models import Project
from .store import load_project

MAX_CHAT_CHARS = 200_000  # total message content cap — protects the engine


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def encode_messages(messages: list[dict]) -> str:
    """OpenAI-style messages → the transcript-encoded prompt the engine sees.

    Client ``system`` entries are dropped (the spec's compiled prompt is the
    authority); the final message must be from the user."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    turns = [m for m in messages
             if isinstance(m, dict) and m.get("role") in ("user", "assistant") and is_str(m.get("content"))]
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("the last message must be from the user")
    if sum(len(m["content"]) for m in turns) > MAX_CHAT_CHARS:
        raise ValueError(f"conversation too large (>{MAX_CHAT_CHARS} characters)")
    history = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in turns[:-1]]
    return conversation_prompt(history, turns[-1]["content"])


def _guard_checks(project: Project) -> list[tuple[str, object]]:
    if project.spec is None:
        return []
    return [(c.id, c.check) for c in project.spec.eval_criteria if c.check is not None]


def _log_guard(project_dir: Path, record: dict) -> None:
    try:
        d = project_dir / "logs"
        d.mkdir(exist_ok=True)
        with (d / "guard.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:  # guarding must never take the endpoint down
        pass


def create_ai_app(project_dir: str | Path, *, engine: Engine | None = None, guard: bool = False):
    """Build the FastAPI app serving the calibrated AI at ``/v1``.

    ``engine`` overrides the subject binding (used by tests); the project is
    loaded once — restart to pick up spec changes (the boot gate re-checks)."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Response
        from fastapi.responses import StreamingResponse
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

    @app.get("/")
    def root():
        """The endpoint self-describes its certification."""
        return {"name": project.name, "goal": project.goal, "engine": engine.name,
                "certification": status, "detail": detail, "guard": guard,
                "openai_base_url": "/v1"}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": project.name, "object": "model", "created": _now(),
                          "owned_by": "ai-calibrator"}]}

    def _complete(messages: list[dict]) -> tuple[str, dict]:
        prompt = encode_messages(messages)
        content = as_str(engine.complete(prompt, system=system)).strip()
        guard_state: dict = {}
        if checks:
            failed = [cid for cid, chk in checks if not run_check(chk, content)[0]]
            if failed:  # one retry — models are stochastic; then flag, never block
                content = as_str(engine.complete(prompt, system=system)).strip()
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
        try:
            content, guard_state = _complete(body.get("messages"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:  # engine failure → OpenAI-style 502, not a traceback
            raise HTTPException(502, f"engine error: {exc}")

        created, cid = _now(), f"chatcmpl-{project.name}-{_now()}"
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
        prompt_chars = sum(len(str(m.get("content", ""))) for m in (body.get("messages") or []) if isinstance(m, dict))
        return {
            "id": cid, "object": "chat.completion", "created": created, "model": project.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": max(1, prompt_chars // 4),
                      "completion_tokens": max(1, len(content) // 4),
                      "total_tokens": max(2, (prompt_chars + len(content)) // 4)},
        }

    return app
