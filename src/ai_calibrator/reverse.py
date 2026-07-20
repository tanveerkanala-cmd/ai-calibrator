"""Reverse-calibrate — extract a tested behavior spec from an existing prompt.

Most people already have a system prompt; what they lack is a way to TEST it.
This is the inverse of ``render_system_prompt``: a compiler engine reads an
existing prompt and recovers the behavior spec it implicitly encodes (persona,
standards, never-rules, edge cases, format, refusal policy, and measurable eval
criteria); ``generate_tests`` then writes a probing test suite. The result is a
normal project — immediately ready for ``eval``, ``coverage``, ``redteam``,
``drift``, and ``report``. The fastest on-ramp: bring the prompt you already have
and find out where it actually breaks.
"""

from __future__ import annotations

from pathlib import Path

from .compile import SPEC_SCHEMA, generate_tests, spec_from_dict, write_build_bundle
from .engines.base import Engine, require_object
from .models import BehaviorSpec, EngineBinding, Project, TaskType
from .store import atomic_write_text, save_project

_REVERSE_SYSTEM = (
    "You reverse-engineer a behavior specification from an EXISTING AI system "
    "prompt. Read the prompt and recover what it encodes: the persona/voice, the "
    "standards it states, the hard 'never' rules, edge-case rulings, output "
    "format, and refusal policy. Also define 3-8 eval_criteria — concrete, "
    "independently checkable statements of correct behavior the prompt implies, "
    "each with a short snake_case id and a weight. Capture only what the prompt "
    "actually says or clearly implies; do NOT invent new requirements. Respond "
    "with JSON only, matching the provided schema."
)


def reverse_spec(prompt_text: str, goal: str, task_type: TaskType, engine: Engine) -> BehaviorSpec:
    """Extract the implicit BehaviorSpec encoded by an existing system prompt."""
    prompt = (
        f"GOAL: {goal}\n"
        f"TASK TYPE: {task_type.value}\n\n"
        f'EXISTING SYSTEM PROMPT:\n"""\n{prompt_text}\n"""\n\n'
        "Recover the behavior specification this prompt encodes."
    )
    out = require_object(
        engine.complete(prompt, system=_REVERSE_SYSTEM, schema=SPEC_SCHEMA), "reverse-calibrator")
    return spec_from_dict(out, goal=goal, task_type=task_type)


def reverse_project(
    name: str,
    goal: str,
    prompt_text: str,
    engine: Engine,
    *,
    task_type: TaskType = TaskType.ASSISTANT,
    engine_spec: str | None = None,
    project_dir: str | Path | None = None,
) -> Project:
    """Build a project from an existing prompt: inferred spec + generated tests.

    If ``engine_spec`` is given, the created project's roles all point at it (so
    the same engine that extracted the spec also runs subsequent eval); otherwise
    the default binding is used. Persists the project, build bundle, and the
    original prompt (for provenance) when ``project_dir`` is given.
    """
    spec = reverse_spec(prompt_text, goal, task_type, engine)
    tests = generate_tests(spec, engine)

    bindings = EngineBinding()
    if engine_spec:
        bindings = EngineBinding(extractor=engine_spec, interviewer=engine_spec, predictor=engine_spec,
                                 compiler=engine_spec, judge=engine_spec, subject=engine_spec)

    project = Project(name=name, goal=goal, task_type=task_type, engines=bindings, spec=spec, tests=tests)
    if project_dir is not None:
        save_project(project, project_dir)
        write_build_bundle(spec, tests, project_dir)
        atomic_write_text(Path(project_dir) / "imported_prompt.txt", prompt_text)  # provenance
    return project
