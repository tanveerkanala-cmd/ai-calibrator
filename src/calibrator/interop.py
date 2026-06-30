"""Eval-format interop — export the spec's tests + rubric to promptfoo.

Anti-lock-in: lets power users run calibrator's generated suite inside promptfoo
(a popular eval harness) instead of only `calibrate eval`. Deterministic — no
engine. The provider-agnostic system prompt becomes a promptfoo prompt, each test
an `input` var, and each expected eval criterion an `llm-rubric` assertion. Edit
the emitted `providers:` line to point at whatever model you want to grade.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .compile import render_system_prompt
from .engines.base import parse_engine_spec
from .models import Project
from .store import atomic_write_text


def _provider_id(spec: str) -> str:
    """Map a ``model@provider`` spec to a promptfoo provider id (best effort)."""
    model, provider = parse_engine_spec(spec)
    return {
        "anthropic": f"anthropic:messages:{model}",
        "openai": f"openai:chat:{model}",
        "ollama": f"ollama:chat:{model}",
    }.get(provider, spec)


def to_promptfoo(project: Project) -> str:
    """Render a promptfoo config (YAML) from the project's spec + tests."""
    spec = project.spec
    if spec is None:
        raise ValueError("No spec — run `calibrate compile` (or `import`) first.")
    crit = {c.id: c.description for c in spec.eval_criteria}

    tests = []
    for t in project.tests:
        targeted = [cid for cid in (t.expects or list(crit)) if cid in crit]
        asserts = [{"type": "llm-rubric", "value": crit[cid]} for cid in targeted]
        if not asserts:  # no criteria → grade against the goal so the test still runs
            asserts = [{"type": "llm-rubric", "value": f"Satisfies the goal: {project.goal}"}]
        tests.append({
            "description": t.id + (f" — {t.notes}" if t.notes else ""),
            "vars": {"input": t.input},
            "assert": asserts,
        })

    config = {
        "description": project.goal,
        "prompts": [render_system_prompt(spec) + "\n\n{{input}}"],
        "providers": [_provider_id(project.engines.subject)],
        "defaultTest": {"options": {"provider": _provider_id(project.engines.judge)}},
        "tests": tests,
    }
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100)


def export_promptfoo(project: Project, *, project_dir: str | Path) -> Path:
    """Write ``<project>/promptfooconfig.yaml`` (atomically)."""
    return atomic_write_text(Path(project_dir) / "promptfooconfig.yaml", to_promptfoo(project))
