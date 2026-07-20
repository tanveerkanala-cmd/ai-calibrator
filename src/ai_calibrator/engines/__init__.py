"""Pluggable engines — the LLMs that power the tool's intelligent steps."""

from .base import Engine, Role, get_engine, parse_engine_spec

__all__ = ["Engine", "Role", "get_engine", "parse_engine_spec"]
