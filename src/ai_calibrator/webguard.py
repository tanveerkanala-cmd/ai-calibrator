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

from urllib.parse import urlsplit

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "testserver"}


def install_guard(app, allowed_hosts: list[str] | None = None) -> None:
    """Attach the Host-allowlist + anti-CSRF middleware to a FastAPI app."""
    from fastapi.responses import JSONResponse

    allowed = set(LOOPBACK_HOSTS) | {h.lower() for h in (allowed_hosts or [])}

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
