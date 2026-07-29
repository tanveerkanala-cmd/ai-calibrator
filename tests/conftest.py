"""Shared test scaffolding.

Starlette's ``TestClient`` addresses the app as ``http://testserver`` by default,
so every request carries ``Host: testserver`` — a name the servers' loopback
allowlist (:data:`ai_calibrator.webguard.LOOPBACK_HOSTS`) deliberately does not
accept, because shipping it would bake a test-only bypass into the Host/Origin
guard that protects every ``calibrate serve`` / ``calibrate run`` process.

Default the clients to loopback instead, so the suite exercises the same guard a
real user runs. Tests that probe the guard itself still pass their own
``base_url`` / ``Host``; an explicit argument always wins.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

LOOPBACK_BASE_URL = "http://localhost"


@pytest.fixture(autouse=True)
def _loopback_test_client(monkeypatch):
    original = TestClient.__init__

    def _init(self, app, base_url=LOOPBACK_BASE_URL, *args, **kwargs):
        original(self, app, base_url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "__init__", _init)
