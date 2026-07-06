"""Coerce untrusted engine-supplied JSON values into safe model field types.

Schema-constrained engines return correctly-typed values, but a non-compliant
model can emit a truthy non-string (e.g. ``"voice": 123``) where a string is
expected. Pydantic v2 does NOT implicitly coerce ``int``/``list``/``dict`` → str,
so such a value raises a ``ValidationError`` deep inside a compile/ingest stage.

These helpers normalize engine output at the parse boundary so junk becomes a
sane default instead of a crash. They are applied ONLY to fresh engine output —
``load_project`` stays strict, so genuinely corrupt project files are still
detected rather than silently coerced.
"""

from __future__ import annotations

import re

# A model id / output-dir token safe to bake into a GENERATED file that later
# runs — train.py / run.py (Python), a Modelfile, README/shell lines. Restricting
# to this charset means the value cannot carry a quote, semicolon, backslash,
# backtick, dollar, space, or newline, so it can't break out of a string literal,
# a comment, or a shell command in any of those templates. Covers real ids:
# "Qwen/Qwen2.5-7B-Instruct", "gemma4:e4b", "mistralai/Mistral-7B-Instruct-v0.3".
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._:/-]+$")


def safe_token(value: str, field: str) -> str:
    """Return ``value`` if it is a plain model/path token, else raise ValueError.

    Guards the code/template generators (finetune, train-engine, export) against
    injection via a hand-edited engine binding or a crafted ``--base``."""
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise ValueError(
            f"{field} must be a plain model/path token (letters, digits, and . _ - : /); "
            f"got {value!r}"
        )
    # The charset permits "." and "/", so also reject traversal-ish shapes: a
    # ".." segment or a leading/trailing "/" has no place in a model id / dir name
    # and could point a generated FROM/output line at the wrong path.
    if ".." in value or value.startswith("/") or value.endswith("/"):
        raise ValueError(f"{field} must not contain '..' or a leading/trailing '/'; got {value!r}")
    return value


def is_str(value: object) -> bool:
    """True only for a non-blank string — used to gate required string fields."""
    return isinstance(value, str) and bool(value.strip())


def as_opt_str(value: object) -> str | None:
    """An optional string field: the string if non-blank, else ``None``."""
    return value if isinstance(value, str) and value.strip() else None


def as_str(value: object, default: str = "") -> str:
    """A required string field: the string if it is one, else ``default``."""
    return value if isinstance(value, str) else default


def as_list(value: object) -> list:
    """A list field, else ``[]``.

    ``dict.get(key, [])`` returns ``None`` when the key is present with value
    ``null`` (the default only applies to *missing* keys), and a non-compliant
    engine can emit an array field as ``null``, a string, or an object. Iterating
    those crashes (``None``) or silently misbehaves (string → chars, dict →
    keys). Wrap every engine-output list access in this to get a real list."""
    return value if isinstance(value, list) else []
