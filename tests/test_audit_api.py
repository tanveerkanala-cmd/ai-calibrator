"""Local-server surface: the HTTP guard, the project lock protocol, and the
input budgets both servers are supposed to enforce."""

import threading
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from ai_calibrator import api
from ai_calibrator.api import create_app, default_projects_root
from ai_calibrator.models import BehaviorSpec, Check, EvalCriterion, Project, Weight
from ai_calibrator.runtime import create_ai_app
from ai_calibrator.store import save_project
from ai_calibrator.webguard import LOOPBACK_HOSTS


class _Engine:
    """Answers whatever it is told to, so a guard check can be made to fail."""

    name = "fake@test"

    def __init__(self, reply="no"):
        self.reply = reply

    def complete(self, prompt, *, system=None, schema=None):
        return self.reply


def _guarded(tmp_path, criterion_id, engine):
    p = Project(name="my-ai", goal="be polite")
    p.spec = BehaviorSpec(goal="be polite", eval_criteria=[
        EvalCriterion(id=criterion_id, description="stays polite", weight=Weight.HIGH,
                      check=Check(kind="contains", value="please"))])
    save_project(p, tmp_path)
    return TestClient(create_ai_app(tmp_path, engine=engine, guard=True))


def test_test_only_hostname_is_not_in_the_shipped_allowlist(tmp_path):
    """`testserver` is the synthetic Host Starlette's TestClient sends. Shipping it
    in the allowlist would put a test-only bypass in the DNS-rebinding and CSRF
    defenses of every `calibrate serve` / `calibrate run` process."""
    assert "testserver" not in LOOPBACK_HOSTS
    c = TestClient(create_app(tmp_path), base_url="http://localhost")
    assert c.get("/api/health", headers={"Host": "testserver"}).status_code == 400
    # The Origin union is the cheaper path: a legitimate loopback Host with an
    # attacker-controlled Origin, which needs no rebinding at all.
    assert c.post("/api/projects", json={"name": "p", "goal": "g"},
                  headers={"Origin": "http://testserver"}).status_code == 403


# The compiler LLM writes criterion ids verbatim into the spec, so a non-English
# project (or a hand-edited project.yaml) can carry an id an HTTP header value
# cannot hold. Flagging must degrade, never take the endpoint down.
@pytest.mark.parametrize("criterion_id", ["敬語_丁寧", "tono_cortés", "a\r\nx-evil: 1"])
@pytest.mark.parametrize("stream", [False, True])
def test_guard_flags_a_failure_whatever_the_criterion_id_contains(tmp_path, criterion_id, stream):
    c = _guarded(tmp_path, criterion_id, _Engine("no"))
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "q"}], "stream": stream})
    assert r.status_code == 200
    flag = r.headers["x-calibrate-guard"]
    assert flag.startswith("failed")
    flag.encode("latin-1")  # header values go on the wire as latin-1
    assert "\r" not in flag and "\n" not in flag


def test_guard_header_is_unchanged_for_an_ordinary_criterion_id(tmp_path):
    c = _guarded(tmp_path, "polite_tone", _Engine("no"))
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.headers["x-calibrate-guard"] == "failed:polite_tone"


@contextmanager
def _held(directory):
    """Hold the project's lock from another actor, as an eval or `calibrate ci`
    run does for minutes at a time."""
    from ai_calibrator.store import project_lock

    acquired, release = threading.Event(), threading.Event()

    def hold():
        with project_lock(directory):
            acquired.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert acquired.wait(5)
    try:
        yield
    finally:
        release.set()
        holder.join(5)


def test_a_busy_project_returns_423_instead_of_holding_the_connection(tmp_path, monkeypatch):
    """Every mutating route waits the same bounded window. A request queued behind
    a multi-minute engine run must answer 423, not pin the connection (and a
    Starlette threadpool slot) until the run finishes."""
    monkeypatch.setattr(api, "_LOCK_WAIT_SECONDS", 0.2)
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200
    (tmp_path / "p" / "materials").mkdir(exist_ok=True)
    (tmp_path / "p" / "materials" / "faq.txt").write_text("hi")
    for source in ("a", "b"):
        proj = Project(name=source, goal="g")
        proj.spec = BehaviorSpec(goal="g")
        save_project(proj, tmp_path / source)

    # Routes that act on the EXISTING busy project.
    on_p = [
        ("DELETE", "/api/projects/p", {}),
        ("DELETE", "/api/projects/p/materials/faq.txt", {}),
        ("POST", "/api/projects/p/log", {"json": {"enabled": True}}),
    ]
    with _held(tmp_path / "p"):
        for method, url, kwargs in on_p:
            started = time.monotonic()
            r = c.request(method, url, **kwargs)
            assert r.status_code == 423, (method, url, r.text)
            assert time.monotonic() - started < 2, (method, url)  # gave up on the deadline

    # The create-shaped routes reach the lock only for a name that does NOT
    # already exist — an existing one is a permanent 409 answered before the
    # wait. Contend on a fresh name so the bounded wait is what is measured.
    on_new = [
        ("POST", "/api/projects", {"json": {"name": "q", "goal": "g"}}),
        ("POST", "/api/import", {"json": {"name": "q", "goal": "g", "prompt": "be nice"}}),
        ("POST", "/api/merge/apply", {"json": {"out": "q", "sources": ["a", "b"]}}),
    ]
    with _held(tmp_path / "q"):
        for method, url, kwargs in on_new:
            started = time.monotonic()
            r = c.request(method, url, **kwargs)
            assert r.status_code == 423, (method, url, r.text)
            assert time.monotonic() - started < 2, (method, url)


def test_concurrent_creates_of_one_name_still_settle_on_200_and_409(tmp_path):
    """The bounded wait must not turn the create race into a 423: one POST wins,
    the other gets a deterministic 409."""
    c = TestClient(create_app(tmp_path))
    results = []
    barrier = threading.Barrier(2)

    def create():
        barrier.wait()
        results.append(c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code)

    threads = [threading.Thread(target=create) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(results) == [200, 409]


def test_oversized_upload_is_refused_before_the_request_is_routed(tmp_path, monkeypatch):
    """The 413 has to land before the body is buffered — FastAPI spools a whole
    multipart upload to the OS temp dir before the endpoint is ever entered, so a
    cap applied there bounds materials/, not what the server accepts. Posting to a
    project that does not exist proves the order: routing would answer 404."""
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4096)
    c = TestClient(create_app(tmp_path))
    r = c.post("/api/projects/nope/materials", files={"file": ("big.txt", b"x" * 8192)})
    assert r.status_code == 413


def test_oversized_body_is_refused_when_no_length_is_declared(tmp_path, monkeypatch):
    """A chunked body carries no Content-Length to pre-check, so the budget has to
    be spent as the bytes arrive — on the JSON routes as much as on uploads."""
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4096)
    c = TestClient(create_app(tmp_path))

    def chunks():
        for _ in range(4):
            yield b"x" * 4096

    r = c.post("/api/projects", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_upload_within_the_cap_still_lands_in_materials(tmp_path):
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200
    r = c.post("/api/projects/p/materials", files={"file": ("faq.txt", b"hello")})
    assert r.status_code == 200 and r.json() == {"uploaded": "faq.txt"}
    assert (tmp_path / "p" / "materials" / "faq.txt").read_bytes() == b"hello"


def test_workbench_feedback_is_capped_like_the_runtime_endpoint(tmp_path):
    """Both endpoints feed the same inbox, and an absorbed record becomes a
    permanent test input re-sent to the subject AND the judge on every later run."""
    from ai_calibrator.flywheel import read_feedback
    from ai_calibrator.runtime import MAX_CHAT_CHARS

    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200
    r = c.post("/api/projects/p/feedback", json={
        "turns": ["q"], "output": "a" * (MAX_CHAT_CHARS + 1), "verdict": "down"})
    assert r.status_code == 400 and "too large" in r.json()["detail"]
    assert read_feedback(tmp_path / "p") == []


def test_workbench_feedback_still_records_an_ordinary_record(tmp_path):
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200
    r = c.post("/api/projects/p/feedback", json={
        "turns": ["can I return after 40 days?"], "output": "sure!", "verdict": "down",
        "correction": "no — the window is 30 days."})
    assert r.status_code == 200 and r.json() == {"recorded": True, "pending": 1}


# --- both servers refuse an oversize body, not just one --------------------

def _served(tmp_path, engine):
    """The `calibrate run` server, seeded with a minimal project."""
    p = Project(name="my-ai", goal="be helpful")
    p.spec = BehaviorSpec(goal="be helpful", eval_criteria=[
        EvalCriterion(id="c1", description="stays helpful", weight=Weight.HIGH)])
    save_project(p, tmp_path)
    return TestClient(create_ai_app(tmp_path, engine=engine, guard=False))


def test_the_served_ai_refuses_an_oversize_body(tmp_path):
    """`calibrate run` is the other server: same guard, same `--host` exposure,
    no authentication. It had no body cap at all, so one request could take the
    process' memory with it. The cap belongs to install_guard, so it cannot be
    installed on one server and forgotten on the other."""
    from ai_calibrator.webguard import MAX_BODY_BYTES

    class _Refuser:
        name = "e@test"

        def complete(self, prompt, *, system=None, schema=None):  # pragma: no cover - never reached
            raise AssertionError("the engine must not be reached for a refused body")

    c = _served(tmp_path, _Refuser())

    oversize = "x" * (MAX_BODY_BYTES + 1)
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": oversize}]})
    assert r.status_code == 413, r.text

    # An announced oversize body is refused without reading it at all.
    r = c.post("/v1/chat/completions",
               headers={"content-length": str(MAX_BODY_BYTES + 1), "content-type": "application/json"},
               content=b'{"messages": []}')
    assert r.status_code == 413, r.text


def test_the_served_ai_still_answers_an_ordinary_request(tmp_path):
    """The cap must not narrow the normal path."""
    class _Answerer:
        name = "e@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "an answer"

    c = _served(tmp_path, _Answerer())
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text


def test_a_name_that_already_exists_answers_409_even_while_the_project_is_busy(tmp_path, monkeypatch):
    """A permanent condition must not be reported as a temporary one. The project
    exists, so the answer is 409 and always will be — waiting out the lock window
    to say "an operation is in progress, retry shortly" is both slow and wrong."""
    import time as _time

    monkeypatch.setattr(api, "_LOCK_WAIT_SECONDS", 5.0)
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200

    with _held(tmp_path / "p"):
        started = _time.monotonic()
        r = c.post("/api/projects", json={"name": "p", "goal": "g"})
        assert r.status_code == 409, r.text
        assert _time.monotonic() - started < 1  # answered, not waited out

        r = c.post("/api/import", json={"name": "p", "goal": "g", "prompt": "be nice"})
        assert r.status_code == 409, r.text


# --- the guard's port handling, and the cap on routes that read no body ----

@pytest.mark.parametrize("host,expected", [
    ("localhost:8765", 200),      # what a real browser sends
    ("127.0.0.1:8765", 200),
    ("[::1]:8765", 200),
    ("localhost", 200),
    ("evil.example:8765", 400),   # a port must not launder a foreign host
    ("evil.example", 400),
])
def test_the_host_allowlist_ignores_the_port_but_not_the_host(tmp_path, host, expected):
    """The suite addresses its clients portlessly, so nothing otherwise exercises
    the guard's port-stripping — a guard that rejected every Host carrying a port
    would pass the whole suite while refusing every real browser request."""
    c = TestClient(create_app(tmp_path))
    assert c.get("/api/health", headers={"Host": host}).status_code == expected


def test_a_declared_oversize_body_is_refused_on_a_route_that_reads_no_body(tmp_path, monkeypatch):
    """The budget is only spent by a handler that READS the body, so a route with
    no body parameter — and every GET — would accept a declared gigabyte without
    the cap ever being consulted."""
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4096)
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200

    for method, url in (("GET", "/api/projects"), ("POST", "/api/projects/p/compile")):
        r = c.request(method, url, headers={"content-length": "1073741824",
                                            "content-type": "application/json"},
                      content=b"{}")
        assert r.status_code == 413, (method, url, r.text)


def test_feedback_answers_423_instead_of_parking_a_worker_thread(tmp_path, monkeypatch):
    """The project lock is held across whole engine runs by `calibrate eval` and
    `calibrate ci`, and each waiting request holds one of the threadpool slots the
    whole process shares. Enough parked feedback POSTs and the server answers
    nothing at all — including, on `calibrate run`, the served AI itself."""
    monkeypatch.setattr(api, "_LOCK_WAIT_SECONDS", 0.2)
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200

    body = {"turns": ["why?"], "output": "because", "verdict": "down"}
    with _held(tmp_path / "p"):
        started = time.monotonic()
        r = c.post("/api/projects/p/feedback", json=body)
        assert r.status_code == 423, r.text
        assert time.monotonic() - started < 2

    # Uncontended, it still records.
    assert c.post("/api/projects/p/feedback", json=body).status_code == 200


def test_the_served_ai_feedback_route_is_bounded_too(tmp_path, monkeypatch):
    from ai_calibrator import runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "FEEDBACK_LOCK_WAIT", 0.2)

    class _Answerer:
        name = "e@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "an answer"

    c = _served(tmp_path, _Answerer())
    body = {"turns": ["why?"], "output": "because", "verdict": "down"}
    with _held(tmp_path):
        started = time.monotonic()
        r = c.post("/v1/feedback", json=body)
        assert r.status_code == 423, r.text
        assert time.monotonic() - started < 2


@pytest.mark.parametrize("field", ["correction", "reason"])
def test_the_feedback_cap_counts_every_field_that_lands_in_the_record(tmp_path, field):
    """An absorbed record becomes a permanent test input sent to both the subject
    and the judge on every future eval, so a field left out of the cap is an
    uncapped permanent cost."""
    from ai_calibrator.runtime import MAX_CHAT_CHARS

    c = TestClient(create_app(tmp_path))
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200

    body = {"turns": ["why?"], "output": "because", "verdict": "down",
            field: "x" * (MAX_CHAT_CHARS + 1)}
    r = c.post("/api/projects/p/feedback", json=body)
    assert r.status_code == 400 and "too large" in r.text


# --- the server reads the directory you are standing in --------------------

def test_the_projects_root_is_the_directory_you_are_in(tmp_path):
    """`calibrate init my-ai` writes ./my-ai and every other command takes a path.
    A server with its own home-directory registry made the documented quickstart
    — init, then serve — end at an empty list."""
    from ai_calibrator.api import default_projects_root

    (tmp_path / "my-ai").mkdir()
    (tmp_path / "my-ai" / "project.yaml").write_text("name: my-ai\ngoal: g\n", encoding="utf-8")

    assert default_projects_root(tmp_path) == tmp_path.resolve()

    c = TestClient(create_app(default_projects_root(tmp_path)))
    assert c.get("/api/projects").json() == ["my-ai"]


def test_standing_inside_a_project_serves_that_project(tmp_path):
    """`cd my-ai && calibrate serve` is the common case, and the listing looks one
    level down — so a directory that is itself a project resolves to its parent."""
    from ai_calibrator.api import default_projects_root

    project = tmp_path / "my-ai"
    project.mkdir()
    (project / "project.yaml").write_text("name: my-ai\ngoal: g\n", encoding="utf-8")

    assert default_projects_root(project) == tmp_path.resolve()

    c = TestClient(create_app(default_projects_root(project)))
    assert c.get("/api/projects").json() == ["my-ai"]


def test_a_project_made_in_the_ui_is_addressable_by_path_from_the_cli(tmp_path):
    """The point of one root: what the UI creates, the CLI can operate on with an
    ordinary relative path — no home-directory detour."""
    from ai_calibrator.store import load_project

    c = TestClient(create_app(default_projects_root(tmp_path)))
    assert c.post("/api/projects", json={"name": "from-ui", "goal": "be helpful"}).status_code == 200

    assert load_project(tmp_path / "from-ui").goal == "be helpful"
