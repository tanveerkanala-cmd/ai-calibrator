"""Does the shipped web UI agree with the API it calls?

Every other API test drives `api.py` through TestClient, which proves the routes
work but says nothing about whether `web/app.js` calls the routes that exist.
Rename a route or a response field and the CLI stays green, the API stays green,
the whole suite stays green — and the front door, which is what "Guided mode"
means for a non-technical owner, breaks silently.

Most are static checks against the shipped assets. They cannot prove the UI
*works* (only a browser can), but they catch the failure that has no other
detector: the two halves drifting apart.

The verdicts the panel prints are checked a second way: the real app.js is run
under node against a stubbed API, because "what does the operator actually
read?" is a question no static check can answer. Those skip where node is
absent, so each one is paired with a static check that always runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
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


# --- the panel's verdicts, read the way an operator reads them ---------------

# Loads the shipped app.js in node against a stubbed API, so the assertion is on
# what the panel actually renders rather than on the source that renders it. The
# DOM shim is deliberately thin: only the handful of calls app.js makes.
_HARNESS = r"""
import fs from "node:fs";
import vm from "node:vm";

const [appPath, planPath] = process.argv.slice(2);
const plan = JSON.parse(fs.readFileSync(planPath, "utf8"));
process.on("unhandledRejection", (e) => {
  console.error("unhandled rejection: " + ((e && e.stack) || e));
  process.exit(1);
});

class El {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this._html = "";
    this.style = {};
    this.dataset = {};
    this.value = "";
    this.classList = { add() {}, remove() {} };
  }
  set innerHTML(html) { this._html = String(html); this.children = []; }
  get innerHTML() { return this._html; }
  insertAdjacentHTML(_where, html) { this._html += String(html); }
  appendChild(child) { this.children.push(child); return child; }
  querySelectorAll(tag) {
    const found = [];
    (function walk(el) {
      for (const c of el.children) { if (c.tagName === tag) found.push(c); walk(c); }
    })(this);
    return found;
  }
  get text() { return this._html + this.children.map((c) => c.text).join(""); }
}

const named = {};
const document = {
  createElement: (tag) => new El(tag),
  createTextNode: (t) => { const e = new El("#text"); e._html = String(t); return e; },
  querySelector: (sel) => (named[sel] = named[sel] || new El("div")),
};

const calls = [];
const routes = Object.keys(plan.routes).sort((a, b) => b.length - a.length);
async function fetchStub(url, opts) {
  opts = opts || {};
  calls.push({ url, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : null });
  const hit = routes.find((r) => url === r) || routes.find((r) => url.startsWith(r));
  return { ok: true, status: 200, statusText: "OK", json: async () => (hit ? plan.routes[hit] : []) };
}

const sandbox = { document, fetch: fetchStub, console, FormData: class { append() {} } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(appPath, "utf8"), sandbox, { filename: "app.js" });
const result = await vm.runInContext(`(async () => { ${plan.script} })()`, sandbox, { filename: "plan.js" });
process.stdout.write(JSON.stringify({ result: result === undefined ? null : result, calls }));
"""


def _run_ui(script: str, routes: dict) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed — this one checks app.js by running it")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "harness.mjs").write_text(_HARNESS, encoding="utf-8")
        (d / "plan.json").write_text(json.dumps({"routes": routes, "script": script}), encoding="utf-8")
        # Decode as UTF-8 explicitly: the panel's own text carries non-ASCII
        # (the arrow, the tick), and `text=True` alone decodes with the locale's
        # encoding — ASCII under LC_ALL=C, where reading the harness's own output
        # raises before any assertion runs.
        proc = subprocess.run([node, str(d / "harness.mjs"), str(WEB / "app.js"), str(d / "plan.json")],
                              capture_output=True, text=True, timeout=60,
                              encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _js_function(name: str) -> str:
    """The source of one top-level function in app.js."""
    js = (WEB / "app.js").read_text(encoding="utf-8")
    start = js.index(f"function {name}(")
    return js[start:js.index("\n}\n", start) + 3]


def _drift_payload(baseline: list, candidate: list) -> dict:
    """The real drift payload for two runs — (test_id, passed, input_hash) rows.

    Built by the library, not by hand: a UI test that invents the response it
    wants proves only that the test and the UI agree with each other.
    """
    from ai_calibrator.drift import compare_scorecards, drift_dict
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult

    def card(run_id, rows):
        return Scorecard(run_id=run_id, results=[
            TestResult(test_id=tid, output="o", input_hash=h,
                       criteria=[CriterionResult(criterion_id="c", passed=p)])
            for tid, p, h in rows])

    return drift_dict(compare_scorecards(card("run-0001", baseline), card("run-0002", candidate)))


def test_the_drift_panel_does_not_call_an_impossible_comparison_no_drift():
    """`compile` re-mints t1..tN under the same ids, so a re-run can grade an
    entirely new suite. The rates are then real but describe two different exams
    — and a panel that reads only the rates prints "✓ No drift" beside "Δ -100%",
    a green verdict for a comparison that never happened."""
    payload = _drift_payload(
        [("t1", True, "a" * 16), ("t2", True, "b" * 16), ("t3", True, "c" * 16)],
        [("t1", False, "x" * 16), ("t2", False, "y" * 16), ("t3", False, "z" * 16)])
    assert payload["compared"] == 0 and payload["delta"] is None   # the state under test

    out = _run_ui('await showDrift("p"); return document.querySelector("#panel").text;',
                  {"/api/projects/p/drift": payload})
    rendered = out["result"]

    assert "No drift" not in rendered, f"claims no drift over nothing compared:\n{rendered}"
    assert "-100" not in rendered, f"prints a delta it did not compute:\n{rendered}"
    assert "comparable" in rendered, f"never says the comparison did not happen:\n{rendered}"


def test_the_drift_panel_shows_the_delta_over_the_tests_it_compared():
    """One re-minted probe with the rest holding is the ordinary state after
    answering another interview question. The Δ has to be the difference of the
    two numbers printed beside it, over a population the panel names."""
    held = [(f"t{i}", True, f"{i:016d}") for i in range(1, 10)]
    payload = _drift_payload([*held, ("t10", True, "a" * 16)], [*held, ("t10", False, "b" * 16)])
    assert payload["compared"] == 9 and payload["delta"] == 0.0

    out = _run_ui('await showDrift("p"); return document.querySelector("#panel").text;',
                  {"/api/projects/p/drift": payload})
    rendered = out["result"]

    assert "9 shared" in rendered, f"does not say what the delta covers:\n{rendered}"
    assert "±0%" in rendered and "-10" not in rendered, f"prints a whole-run delta:\n{rendered}"
    assert "1 " in rendered and "not comparable" in rendered, f"hides the excluded probe:\n{rendered}"


def test_saving_the_interview_posts_only_the_answers_the_person_wrote():
    """Every textarea is prefilled with the tool's drafted guess. Posting them
    all records 7 unreviewed engine guesses as the owner's ratified answers —
    the spec compiles from them, and every "unreviewed draft" caveat downstream
    keys on a source flag this route cannot set. The CLI makes the same act an
    explicit `--accept-drafts` choice with a loud warning."""
    state = {
        "name": "p", "goal": "g", "materials": [], "gaps": [], "has_spec": False, "tests": 0,
        "interview": [
            {"id": f"q{i}", "dimension": "Tone", "question": f"question {i}", "rationale": None,
             "answer": None, "draft_answer": f"ENGINE GUESS {i}", "answer_source": None}
            for i in range(1, 11)
        ],
    }
    script = """
      const panel = document.querySelector("#panel");
      renderInterview(panel, STATE, "p");
      const card = panel.children[0];
      const boxes = card.querySelectorAll("textarea");
      boxes[0].value = "USER TYPED 1";
      boxes[2].value = "USER TYPED 3";
      await card.children[card.children.length - 1].onclick();
      return null;
    """.replace("STATE", json.dumps(state))

    out = _run_ui(script, {"/api/projects/p": state})
    posts = [c for c in out["calls"] if c["method"] == "POST" and c["url"].endswith("/answers")]

    assert len(posts) == 1, f"expected one save, got {posts}"
    assert posts[0]["body"]["answers"] == {"q1": "USER TYPED 1", "q3": "USER TYPED 3"}


def test_the_drift_panel_reads_whether_a_comparison_happened_at_all():
    """The static half of the check above, for a machine with no node: the panel
    has to consult the fields that say whether anything was compared."""
    body = _js_function("showDrift")
    for field in ("delta", "compared", "incomparable_tests"):
        assert field in body, f"showDrift never reads {field} — it cannot tell a comparison from none"
    assert "null" in body or "undefined" in body, "showDrift never handles an absent delta"


def test_the_interview_save_is_selective_about_what_it_posts():
    """The static half: the handler that collects textareas must decide per box,
    not sweep them all — an untouched draft is a proposal, not an answer."""
    body = _js_function("renderInterview")
    collect = body[body.index("const answers = {}"):body.index("await api")]
    assert "if" in collect, "every textarea is posted unconditionally, drafts included"


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
