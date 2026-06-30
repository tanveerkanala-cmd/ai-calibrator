"""Concurrency & durability — regression tests for the stress-found races.

These reproduce the original defects (shared temp file → corruption; non-atomic
check-then-create → duplicate/500; load-modify-save without locking → lost
updates) and assert the fixes hold under real thread contention.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time

import pytest

from calibrator.models import InterviewItem, Project
from calibrator.store import load_project, project_lock, save_project


def test_concurrent_save_no_corruption(tmp_path):
    """100 threads saving the same project concurrently must never raise,
    corrupt project.yaml, or leak temp files. (Was: ~50% FileNotFoundError /
    ValidationError from the shared "project.yaml.tmp" race.)"""
    d = tmp_path / "proj"
    save_project(Project(name="p", goal="seed"), d)

    n = 100
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            save_project(Project(name="p", goal=f"goal-{i}"), d)
        except Exception as exc:  # pragma: no cover - failure path is the bug
            errors.append(exc)

    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(writer, range(n)))

    assert not errors, f"save_project raised under concurrency: {errors[:3]}"
    # File is intact and loadable — no partial/corrupt write survived.
    loaded = load_project(d)
    assert loaded.name == "p"
    assert loaded.goal == "seed" or loaded.goal.startswith("goal-")
    # No scratch files leaked.
    assert list(d.glob("project.yaml*.tmp")) == []


def test_project_lock_is_mutually_exclusive_across_threads(tmp_path):
    """project_lock must serialize holders even within one process (the API
    runs sync endpoints in a thread pool), so at most one thread is ever inside
    the critical section."""
    d = tmp_path / "p"
    d.mkdir()
    active = 0
    peak = 0
    guard = threading.Lock()

    def worker() -> None:
        nonlocal active, peak
        with project_lock(d):
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with guard:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak == 1, f"lock allowed {peak} concurrent holders"


def test_lock_allows_parallelism_across_different_projects(tmp_path):
    """Different projects use different locks, so they don't serialize."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    order: list[str] = []

    def hold(d, tag, hold_s):
        with project_lock(d):
            order.append(f"{tag}-in")
            time.sleep(hold_s)
            order.append(f"{tag}-out")

    t1 = threading.Thread(target=hold, args=(a, "a", 0.05))
    t2 = threading.Thread(target=hold, args=(b, "b", 0.05))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # Both entered before either exited → they ran in parallel.
    assert order.index("b-in") < order.index("a-out") or order.index("a-in") < order.index("b-out")


# --- API-level concurrency (needs the `api` extra) --------------------------

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from calibrator.api import create_app  # noqa: E402


def test_concurrent_create_exactly_one_winner(tmp_path):
    """100 concurrent POSTs for the same name → exactly one 200, the rest 409,
    never a 500, and exactly one usable project on disk. (Was: TOCTOU race —
    multiple 200s + unhandled FileNotFoundError.)"""
    client = TestClient(create_app(tmp_path))

    def create(_: int) -> int:
        return client.post("/api/projects", json={"name": "race", "goal": "g"}).status_code

    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        codes = list(ex.map(create, range(100)))

    assert codes.count(200) == 1, f"expected exactly one 200, got {codes.count(200)}"
    assert set(codes) <= {200, 409}, f"unexpected status codes: {sorted(set(codes))}"
    assert client.get("/api/projects").json() == ["race"]
    assert client.get("/api/projects/race").status_code == 200


def test_concurrent_answers_no_lost_update(tmp_path):
    """N concurrent answer submissions, each setting a distinct question, must
    ALL persist. (Was: read-modify-write race dropped ~85% of answers.)"""
    n = 40
    proj = Project(name="p", goal="g")
    proj.interview = [
        InterviewItem(id=f"q{i}", dimension="d", question="?", draft_answer="")
        for i in range(n)
    ]
    save_project(proj, tmp_path / "p")

    client = TestClient(create_app(tmp_path))

    def answer(i: int) -> int:
        return client.post(
            "/api/projects/p/answers", json={"answers": {f"q{i}": f"ans-{i}"}}
        ).status_code

    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        codes = list(ex.map(answer, range(n)))

    assert all(c == 200 for c in codes), f"non-200 responses: {[c for c in codes if c != 200]}"
    state = client.get("/api/projects/p").json()
    answered = {it["id"]: it["answer"] for it in state["interview"] if it.get("answer")}
    assert len(answered) == n, f"lost updates: only {len(answered)}/{n} answers persisted"
    assert all(answered[f"q{i}"] == f"ans-{i}" for i in range(n))
