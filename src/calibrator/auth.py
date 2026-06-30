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


def anthropic_status() -> AuthStatus:
    if os.getenv("ANTHROPIC_API_KEY"):
        return AuthStatus("claude", True, "ANTHROPIC_API_KEY is set")
    if os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return AuthStatus("claude", True, "ANTHROPIC_AUTH_TOKEN is set (OAuth/login token)")
    if shutil.which("ant"):
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
    if os.getenv("OPENAI_API_KEY"):
        return AuthStatus("openai", True, "OPENAI_API_KEY is set")
    return AuthStatus("openai", False, "set OPENAI_API_KEY (platform.openai.com) — key-based, no ChatGPT login")


def ollama_status() -> AuthStatus:
    return AuthStatus("ollama (local)", True, "no auth required")


def all_status() -> list[AuthStatus]:
    return [anthropic_status(), openai_status(), ollama_status()]


def login_anthropic() -> int:
    """Launch the official Anthropic CLI browser login; return its exit code."""
    if not shutil.which("ant"):
        raise RuntimeError(
            "Browser login to Claude uses the Anthropic CLI (`ant`), which isn't installed.\n"
            "  Install it:  brew install anthropics/tap/ant   (or see docs.claude.com)\n"
            "  …then re-run `calibrate login claude`. Or skip login and set ANTHROPIC_API_KEY."
        )
    # Interactive: inherit stdio so the browser/device flow works.
    return subprocess.call(["ant", "auth", "login"])
