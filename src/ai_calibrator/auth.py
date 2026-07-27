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


def _anthropic_cli() -> str | None:
    """Path to the Anthropic CLI, or None.

    `ant` is also the name of Apache Ant, an extremely common Java build tool, so
    a bare `shutil.which("ant")` would report a Claude login as available and then
    exec a build tool. Verify what the binary actually is before trusting it."""
    path = shutil.which("ant")
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True,
                             timeout=5).stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return None
    if "apache ant" in out:  # the Java build tool, not the Anthropic CLI
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
            "`ant` CLI found, login state unknown — run `calibrate login claude` (or set ANTHROPIC_API_KEY)",
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
            "Browser login to Claude uses the Anthropic CLI (`ant`), which isn't installed.\n"
            "  (If you do have an `ant` on PATH, it is Apache Ant — a different tool.)\n"
            "  Install it:  brew install anthropics/tap/ant   (or see docs.claude.com)\n"
            "  …then re-run `calibrate login claude`. Or skip login and set ANTHROPIC_API_KEY."
        )
    # Interactive: inherit stdio so the browser/device flow works.
    return subprocess.call([cli, "auth", "login"])
