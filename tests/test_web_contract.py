"""Does the shipped web UI agree with the API it calls?

Every other API test drives `api.py` through TestClient, which proves the routes
work but says nothing about whether `web/app.js` calls the routes that exist.
Rename a route or a response field and the CLI stays green, the API stays green,
the whole suite stays green — and the front door, which is what "Guided mode"
means for a non-technical owner, breaks silently.

These are static checks against the shipped assets. They cannot prove the UI
*works* (only a browser can), but they catch the failure that has no other
detector: the two halves drifting apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "src" / "ai_calibrator" / "web"
API_PY = Path(__file__).resolve().parents[1] / "src" / "ai_calibrator" / "api.py"


def _norm(path: str) -> str:
    """Collapse both `{name}` params and JS `${expr}` interpolations to one token."""
    path = re.sub(r"\$\{[^}]*\}", "{p}", path)
    path = re.sub(r"\{[^}]*\}", "{p}", path)
    return path.rstrip("/") or "/"


def _server_routes() -> set[tuple[str, str]]:
    src = API_PY.read_text(encoding="utf-8")
    return {(m.group(1).upper(), _norm(m.group(2)))
            for m in re.finditer(r'@app\.(get|post|put|delete|patch)\("([^"]+)"', src)}


def _ui_calls() -> set[tuple[str, str]]:
    """(method, path) for every request app.js makes.

    Two shapes: the `api(method, path)` helper, which prefixes "/api", and a
    couple of direct `fetch()` calls for uploads and streaming.
    """
    js = (WEB / "app.js").read_text(encoding="utf-8")
    calls: set[tuple[str, str]] = set()

    for m in re.finditer(r'''\bapi\(\s*["'](GET|POST|PUT|DELETE|PATCH)["']\s*,\s*[`'"]([^`'"]+)[`'"]''', js):
        calls.add((m.group(1).upper(), _norm("/api" + m.group(2))))

    for m in re.finditer(r'''\bfetch\(\s*[`'"](/[^`'"]*)[`'"]\s*(?:,\s*\{([^}]*)\})?''', js):
        url, opts = m.group(1), (m.group(2) or "")
        method = re.search(r"method:\s*[\"'](\w+)[\"']", opts)
        calls.add(((method.group(1) if method else "GET").upper(), _norm(url)))

    # The helper's own `fetch("/api" + path)` is not a call site.
    return {c for c in calls if c[1] not in ("/api", "/")}


def test_the_ui_finds_its_call_sites_at_all():
    """A guard on the guard: if app.js is restructured so these patterns stop
    matching, the two tests below would pass by finding nothing."""
    calls = _ui_calls()
    assert len(calls) >= 15, f"only found {len(calls)} UI call sites — the extraction broke"
    assert ("GET", "/api/projects") in calls


def test_every_endpoint_the_ui_calls_exists_on_the_server():
    """The failure with no other detector: a route renamed in api.py, every
    Python test still green, and the web UI 404s on a button nobody clicked in
    CI."""
    missing = sorted(_ui_calls() - _server_routes())
    assert not missing, (
        "web/app.js calls endpoints api.py does not serve:\n"
        + "\n".join(f"  {method} {path}" for method, path in missing))


def test_the_shipped_ui_assets_are_all_present():
    """`create_app` mounts this directory; a missing asset is a blank page."""
    for name in ("index.html", "app.js", "style.css"):
        f = WEB / name
        assert f.is_file() and f.stat().st_size > 0, f"web/{name} missing or empty"


@pytest.mark.parametrize("asset", ["app.js", "style.css"])
def test_index_html_references_the_assets_it_ships_with(asset):
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert asset in html, f"index.html never loads {asset}"


def test_the_ui_is_served_from_the_root_route():
    """`GET /` must return the app, not a JSON listing: it is the URL printed by
    `calibrate serve` and the only one a non-technical owner will ever type."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_calibrator.api import create_app

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        c = TestClient(create_app(Path(d)), base_url="http://localhost")
        r = c.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert "app.js" in r.text


# --- the report is rendered, not dumped -------------------------------------

def test_the_report_is_rendered_not_dumped_as_source():
    """The calibration report is the artifact this tool points at and the one a
    non-technical owner reads or shares. It was written into a `<pre>` as raw
    markdown, so the person the product is FOR saw `## Coverage`, `**67%**` and
    backticks instead of a report."""
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "renderMarkdown(" in js, "no markdown renderer in the UI"
    assert "pre.textContent = d.markdown" not in js, "the report is still dumped as source"


def test_the_markdown_renderer_escapes_before_it_marks_up():
    """Everything in the report is untrusted: the goal, the criteria
    descriptions and the judge's rationales are written by a model from
    ingested documents. Escaping must happen BEFORE any rule adds markup, or a
    rule can emit an element the report's author chose."""
    js = (WEB / "app.js").read_text(encoding="utf-8")
    body = js[js.index("function renderMarkdown"):]
    body = body[:body.index("\n}\n") + 3]

    escape_at = body.index("escapeHtml(")
    for rule in ("<strong>", "<em>", "<code>"):
        assert escape_at < body.index(rule), f"{rule} is produced before escaping"
    # Link syntax would be the one construct that lets untrusted text pick a
    # URL (javascript:). The report has no links; the renderer must not add them.
    assert "<a " not in body and "href" not in body


def test_the_ui_assets_are_served_revalidating():
    """The asset URLs never change — no build step, no content hash — so a
    browser left to its own heuristics can keep running an OLD app.js against a
    NEW API after an upgrade. That is the same halves-drift-apart failure the
    contract test guards, arriving through the cache. `no-cache` means
    revalidate, not "don't store": the etag still answers 304."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_calibrator.api import create_app

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        c = TestClient(create_app(Path(d)), base_url="http://localhost")
        r = c.get("/app.js")
        assert r.status_code == 200
        assert "no-cache" in r.headers.get("cache-control", "")
        etag = r.headers.get("etag")
        assert etag, "no etag — revalidation would resend the whole file every time"
        assert c.get("/app.js", headers={"If-None-Match": etag}).status_code == 304
