"""Shared HTTP guard for the local servers (`calibrate serve` and `calibrate run`).

Both servers bind localhost by default, but localhost binding alone does not
stop two browser-borne attacks:

- **DNS rebinding** — a malicious page resolves its own domain to 127.0.0.1 and
  becomes same-origin with the local server. Blocked by the Host allowlist.
- **CSRF** — a malicious page fires a no-preflight "simple request" (e.g. a
  ``text/plain`` POST) at the server; the browser sends it cross-origin even
  though the page can't read the reply. Blocked by the Origin check on
  mutating requests.

Fail closed: an absent or unparseable Host is rejected. To expose beyond
localhost, the CLI binds a specific reachable address and adds it to the
allowlist, so BOTH checks still protect you.

NOTE: FastAPI is imported inside :func:`install_guard` so importing this
module never requires the ``api`` extra (same lazy-import rule as runtime.py).
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Ceiling on a single request body, for both servers. Neither is authenticated,
# and `--host` makes either reachable from the LAN, so an unbounded body is a
# one-request memory exhaust. Generous enough for the largest material upload.
MAX_BODY_BYTES = 25 * 1024 * 1024


def _too_large(limit: int):
    """The 413 to raise, built lazily.

    An HTTPException specifically: it is raised from inside the body read, and
    FastAPI turns every OTHER exception raised there into a generic 400 "error
    parsing the body" — which would hide the real reason the request was refused.
    Imported here rather than at module scope so this module keeps importing
    without the ``api`` extra."""
    from fastapi import HTTPException

    return HTTPException(413, _too_large_detail(limit))


def _too_large_detail(limit: int) -> str:
    return f"request body too large (max {limit // 1024 // 1024} MB)"


class _BodyLimit:
    """Spend the size budget as the body arrives, not once it has all landed.

    An endpoint cannot enforce it: FastAPI parses the whole multipart body — a
    file part rolls out of memory into the OS temp dir past 1 MB — *before* the
    endpoint that measures it is entered, so a cap applied there bounds only what
    reaches disk, never what the server accepts. Counting at the ASGI boundary is
    the only place the budget is real, and it covers a chunked body (no
    Content-Length to pre-check) and the JSON routes alike.

    Raising rather than replying lets the app's own error handling answer, so the
    413 comes back through the normal response path."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = 0
        for key, value in scope.get("headers", ()):
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                break
        seen = 0
        limit = self.max_bytes

        if declared > limit:
            # Answer here rather than inside the body read: the budget is only
            # spent by a handler that actually READS the body, so a route with no
            # body parameter (and every GET) would otherwise accept a declared
            # gigabyte without ever consulting the cap.
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body",
                        "body": json.dumps({"detail": _too_large_detail(limit)}).encode()})
            return

        async def _measured_receive():
            nonlocal seen
            if declared > limit:
                raise _too_large(limit)  # refuse an announced oversize body unread
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise _too_large(limit)
            return message

        await self.app(scope, _measured_receive, send)


def install_guard(app, allowed_hosts: list[str] | None = None, *,
                  max_body_bytes: int = MAX_BODY_BYTES) -> None:
    """Attach the body cap + Host-allowlist + anti-CSRF middleware to a FastAPI app.

    One call installs all three. The cap used to be the caller's job, and the
    server that forgot it (`calibrate run`) accepted unbounded bodies for as long
    as it existed — so it belongs to whatever installs the guard, not to whoever
    remembers."""
    from fastapi.responses import JSONResponse

    allowed = set(LOOPBACK_HOSTS) | {h.lower() for h in (allowed_hosts or [])}

    # Added first, so the Host guard below wraps it: a foreign Host is rejected
    # before a single body byte is read, and the 413 still surfaces as a 413
    # (raising inside receive() under the guard's own wrapper degrades it to a
    # misleading 400 — anyio's task-group wrapping defeats FastAPI's re-raise).
    app.add_middleware(_BodyLimit, max_bytes=max_body_bytes)

    @app.middleware("http")
    async def _guard(request, call_next):
        raw = request.headers.get("host") or ""
        host = (raw.split("]")[0].lstrip("[") if raw.startswith("[") else raw.split(":")[0]).lower().rstrip(".")
        if host not in allowed:
            return JSONResponse(status_code=400, content={"detail": "host not allowed"})
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and (urlsplit(origin).hostname or "").lower().rstrip(".") not in allowed:
                return JSONResponse(status_code=403, content={"detail": "cross-origin request blocked"})
        return await call_next(request)
