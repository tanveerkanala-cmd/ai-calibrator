"""Engine sign-in — how you authenticate to each provider.

- **Claude**: real browser/OAuth login via the official Anthropic CLI
  (`ant auth login`). The `anthropic` SDK resolves that profile automatically,
  so the engine works with **no API key**. (`ANTHROPIC_API_KEY` /
  `ANTHROPIC_AUTH_TOKEN` also work.)
- **OpenAI**: the API is **key-based**. There is no supported "sign in with
  ChatGPT" for third-party tools, so you set `OPENAI_API_KEY`.
- **Local (Ollama)**: no auth at all.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class AuthStatus:
    provider: str
    configured: bool
    detail: str


def _looks_like_placeholder(key: str) -> bool:
    """True if a set key is obviously not a real credential — an ellipsis/angle
    placeholder copied from the docs, or implausibly short — so `auth` doesn't
    green-check it and defer the real failure to the first engine call."""
    k = key.strip()
    return ("..." in k or "<" in k or ">" in k or k.endswith("-")
            or len(k.split("-")[-1]) < 8)


# The binary has to say one of these about itself before we will run it.
_ANTHROPIC_MARKERS = ("anthropic", "claude")


@functools.lru_cache(maxsize=1)
def _anthropic_cli() -> str | None:
    """Path to the Anthropic CLI, or None when it can't be positively identified.

    `ant` is also the name of Apache Ant, an extremely common Java build tool, so
    a bare `shutil.which("ant")` would report a Claude login as available and then
    exec a build tool. Recognizing Apache Ant is the wrong way round, though: the
    binary we know least about is exactly the one we must not hand the user's
    terminal to. So the only accepted binary is one whose own help text names
    Anthropic or Claude; a different tool, a non-zero exit, a hang, no output at
    all — all of it means "no Anthropic CLI here", and the user is pointed at
    ANTHROPIC_API_KEY instead.

    The probe costs a subprocess and `anthropic_status()` runs on both `calibrate
    auth` and `GET /api/auth`, so the answer is cached for the life of the process
    (call `_anthropic_cli.cache_clear()` after installing the CLI)."""
    path = shutil.which("ant")
    if not path:
        return None
    try:
        proc = subprocess.run([path, "--help"], capture_output=True, text=True,
                              errors="replace", timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None  # not executable, or it hung — either way, not usable
    if proc.returncode != 0:
        return None
    blurb = ((proc.stdout or "") + (proc.stderr or "")).lower()
    if not any(marker in blurb for marker in _ANTHROPIC_MARKERS):
        return None
    return path


def anthropic_status() -> AuthStatus:
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        if _looks_like_placeholder(key):
            return AuthStatus("claude", False,
                              "ANTHROPIC_API_KEY is set but doesn't look like a real key "
                              "(placeholder?) — the first call will fail until you set a real one")
        return AuthStatus("claude", True, "ANTHROPIC_API_KEY is set")
    if os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return AuthStatus("claude", True, "ANTHROPIC_AUTH_TOKEN is set (OAuth/login token)")
    if _anthropic_cli() is not None:
        # Presence of the CLI does not mean the user is logged in — report it as
        # not-yet-confirmed rather than a false "configured".
        return AuthStatus(
            "claude", False,
            "Anthropic CLI found, login state unknown — run `calibrate login claude` (or set ANTHROPIC_API_KEY)",
        )
    if shutil.which("ant"):
        # There is an `ant`, but it didn't identify itself as Anthropic's, so
        # browser login is not a route we can offer. Say that plainly.
        return AuthStatus(
            "claude", False,
            "the `ant` on your PATH isn't the Anthropic CLI (Apache Ant shares the name) — "
            "set ANTHROPIC_API_KEY, or install the Anthropic CLI and run `calibrate login claude`",
        )
    return AuthStatus(
        "claude", False,
        "set ANTHROPIC_API_KEY, or install the `ant` CLI and run `calibrate login claude`",
    )


def openai_status() -> AuthStatus:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        if _looks_like_placeholder(key):
            return AuthStatus("openai", False,
                              "OPENAI_API_KEY is set but doesn't look like a real key "
                              "(placeholder?) — the first call will fail until you set a real one")
        return AuthStatus("openai", True, "OPENAI_API_KEY is set")
    return AuthStatus("openai", False, "set OPENAI_API_KEY (platform.openai.com) — key-based, no ChatGPT login")


def ollama_status() -> AuthStatus:
    return AuthStatus("ollama (local)", True, "no auth required")


def all_status() -> list[AuthStatus]:
    return [anthropic_status(), openai_status(), ollama_status()]


def login_anthropic() -> int:
    """Launch the official Anthropic CLI browser login; return its exit code."""
    cli = _anthropic_cli()
    if cli is None:
        raise RuntimeError(
            "Browser login to Claude uses the Anthropic CLI (`ant`), and there is no\n"
            "  confirmed Anthropic CLI on your PATH — nothing safe to launch.\n"
            "  (An `ant` that is Apache Ant, the Java build tool, does not count.)\n"
            "  Install it:  brew install anthropics/tap/ant   (or see docs.claude.com)\n"
            "  …then re-run `calibrate login claude`. Or skip login and set ANTHROPIC_API_KEY."
        )
    # Interactive: inherit stdio so the browser/device flow works.
    try:
        return subprocess.call([cli, "auth", "login"])
    except OSError as exc:  # removed since the probe cached its answer
        raise RuntimeError(
            f"Couldn't run the Anthropic CLI at {cli}: {exc}\n"
            "  Reinstall it, or skip login and set ANTHROPIC_API_KEY."
        ) from exc
