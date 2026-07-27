"""Auth status must not green-check an obvious placeholder key, and the `ant`
probe must fail closed rather than run an unidentified binary."""

import subprocess

import pytest

from ai_calibrator import auth
from ai_calibrator.auth import anthropic_status, openai_status


def test_placeholder_anthropic_key_not_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")   # the docs placeholder
    st = anthropic_status()
    assert st.configured is False and "placeholder" in st.detail


def test_angle_placeholder_not_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "<your-key>")
    assert openai_status().configured is False


def test_real_looking_key_is_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abcdefgh12345678ZZ")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefgh12345678")
    assert anthropic_status().configured is True
    assert openai_status().configured is True




# --- the `ant` probe must fail closed: an unidentified binary is never run ---

class _FakeProc:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture()
def clean_cli_probe():
    """The probe caches its answer for the process, so start and end clean."""
    auth._anthropic_cli.cache_clear()
    yield
    auth._anthropic_cli.cache_clear()


def _fake_ant(monkeypatch, result, path="/usr/local/bin/ant"):
    """Put an `ant` on PATH whose --help returns (or raises) `result`."""
    monkeypatch.setattr(auth.shutil, "which", lambda name: path if name == "ant" else None)

    def fake_run(*args, **kwargs):
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(auth.subprocess, "run", fake_run)
    return path


def test_apache_ant_is_not_accepted(monkeypatch, clean_cli_probe):
    # Apache Ant doesn't understand --help: it complains and exits non-zero.
    _fake_ant(monkeypatch, _FakeProc(1, "", "Unknown argument: --help\nant [options] [target]"))
    assert auth._anthropic_cli() is None


def test_binary_that_says_nothing_is_not_accepted(monkeypatch, clean_cli_probe):
    _fake_ant(monkeypatch, _FakeProc(0, "", ""))
    assert auth._anthropic_cli() is None


def test_unknown_tool_that_exits_cleanly_is_not_accepted(monkeypatch, clean_cli_probe):
    _fake_ant(monkeypatch, _FakeProc(0, "Ant version 1.10.14 compiled on August 16 2023\n"))
    assert auth._anthropic_cli() is None


def test_hanging_or_broken_binary_is_not_accepted(monkeypatch, clean_cli_probe):
    _fake_ant(monkeypatch, subprocess.TimeoutExpired(cmd="ant", timeout=5))
    assert auth._anthropic_cli() is None
    auth._anthropic_cli.cache_clear()
    _fake_ant(monkeypatch, OSError("Exec format error"))
    assert auth._anthropic_cli() is None


def test_real_anthropic_cli_is_accepted(monkeypatch, clean_cli_probe):
    path = _fake_ant(monkeypatch, _FakeProc(0, "ant - the Anthropic CLI\n  auth login   Sign in\n"))
    assert auth._anthropic_cli() == path


def test_status_doesnt_offer_login_for_an_unidentified_ant(monkeypatch, clean_cli_probe):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    _fake_ant(monkeypatch, _FakeProc(1, "", "Unknown argument: --help"))
    st = auth.anthropic_status()
    assert st.configured is False
    assert "isn't the Anthropic CLI" in st.detail and "ANTHROPIC_API_KEY" in st.detail


def test_login_refuses_to_exec_an_unidentified_ant(monkeypatch, clean_cli_probe):
    _fake_ant(monkeypatch, _FakeProc(1, "", "Unknown argument: --help"))

    def boom(*args, **kwargs):
        raise AssertionError(f"refused to run, yet ran {args!r}")

    monkeypatch.setattr(auth.subprocess, "call", boom)
    with pytest.raises(RuntimeError) as exc:
        auth.login_anthropic()
    assert "ANTHROPIC_API_KEY" in str(exc.value)
