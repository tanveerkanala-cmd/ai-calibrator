"""M3 — Compile: synthesize the behavior spec, then compile the artifact bundle.

The expert's ratified interview answers + extracted facts are synthesized into a
`BehaviorSpec` (the source of truth) by the compiler engine. Everything else is
*compiled from the spec* — the system prompt, RAG config, and eval rubric are
deterministic renders; the test cases are a second engine pass. Output lands in
`<project>/build/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .coerce import as_list, as_opt_str, as_str, is_str
from .engines.base import Engine, require_object
from .store import atomic_write_text
from .models import (
    BehaviorSpec,
    EdgeCase,
    EvalCriterion,
    Example,
    Persona,
    Project,
    TaskType,
    TestCase,
    Weight,
)
from .rag import EMBED_MODEL, TABLE, TOP_K

# --- Structured-output schemas (strict-compatible across providers) ---------

SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "persona": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "voice": {"type": "string"},
                "reading_level": {"type": "string"},
            },
            "required": ["voice", "reading_level"],
        },
        "standards": {"type": "array", "items": {"type": "string"}},
        "do_not": {"type": "array", "items": {"type": "string"}},
        "edge_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "situation": {"type": "string"},
                    "ruling": {"type": "string"},
                },
                "required": ["situation", "ruling"],
            },
        },
        "format": {"type": "string"},
        "refusal_policy": {"type": "string"},
        "eval_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "weight": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["id", "description", "weight"],
            },
        },
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "input": {"type": "string"},
                    "good_output": {"type": "string"},
                    "bad_output": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["input", "good_output", "bad_output", "why"],
            },
        },
    },
    "required": [
        "persona", "standards", "do_not", "edge_cases",
        "format", "refusal_policy", "eval_criteria", "examples",
    ],
}

TESTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "input": {"type": "string"},
                    "expects": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                    "follow_ups": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "input", "expects", "notes", "follow_ups"],
            },
        }
    },
    "required": ["tests"],
}

_SPEC_SYSTEM = (
    "You convert an expert's interview answers into a precise, implementable "
    "behavior specification for an AI. Capture their voice, standards, hard "
    "'never' rules, edge-case rulings, output format, and refusal policy "
    "faithfully — never invent preferences they didn't express. Also define 3-8 "
    "eval_criteria: concrete, independently checkable statements of correct "
    "behavior, each with a short snake_case id and a weight. Give 1-3 good/bad "
    "examples. Respond with JSON only, matching the provided schema."
)

_TESTS_SYSTEM = (
    "You write test inputs that probe whether an AI follows its behavior spec. "
    "Include normal cases and the tricky edge cases the spec calls out. For each "
    "test, list the eval-criterion ids it should satisfy, give it a short stable "
    "id (t1, t2, …), and a one-line note. Where multi-turn behavior matters "
    "(clarification, context carry-over, persistence under pushback), add "
    "'follow_ups': the subsequent user turns of a conversation; leave it [] for "
    "single-turn tests. Respond with JSON only, matching the provided schema."
)


@dataclass
class CompileResult:
    standards: int
    edge_cases: int
    criteria: int
    tests: int
    build_dir: str
    files: list[str]


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _as_weight(value: object) -> Weight:
    """Coerce a model-supplied weight to a valid enum, defaulting to MEDIUM.

    The schema constrains this to low/medium/high, but a non-compliant engine
    could emit anything; an unknown value must degrade gracefully rather than
    raise ``ValueError`` deep in the compile step."""
    try:
        return Weight(str(value).lower())
    except (ValueError, AttributeError):
        return Weight.MEDIUM


def _qa_block(project: Project) -> str:
    parts = []
    for it in project.interview:
        if it.answer:
            parts.append(f"[{it.dimension}] Q: {it.question}\nA: {it.answer}")
    return "\n\n".join(parts)


def spec_from_dict(out: dict, *, goal: str, task_type: TaskType, knowledge_sources=()) -> BehaviorSpec:
    """Map a schema-constrained engine response (SPEC_SCHEMA shape) into a
    BehaviorSpec, coercing/skipping malformed fields. Shared by ``synthesize_spec``
    (compile from interview) and ``reverse_spec`` (import an existing prompt)."""
    persona = out.get("persona")
    if not isinstance(persona, dict):
        persona = {}
    return BehaviorSpec(
        goal=goal,
        task_type=task_type,
        persona=Persona(
            voice=as_opt_str(persona.get("voice")),
            reading_level=as_opt_str(persona.get("reading_level")),
        ),
        standards=[s for s in as_list(out.get("standards")) if is_str(s)],
        do_not=[s for s in as_list(out.get("do_not")) if is_str(s)],
        edge_cases=[
            EdgeCase(situation=e["situation"], ruling=e["ruling"])
            for e in as_list(out.get("edge_cases"))
            if isinstance(e, dict) and is_str(e.get("situation")) and is_str(e.get("ruling"))
        ],
        format=as_opt_str(out.get("format")),
        refusal_policy=as_opt_str(out.get("refusal_policy")),
        knowledge_sources=list(knowledge_sources),
        eval_criteria=[
            EvalCriterion(
                id=c["id"],
                description=as_str(c.get("description")),
                weight=_as_weight(c.get("weight")),
            )
            for c in as_list(out.get("eval_criteria"))
            if isinstance(c, dict) and is_str(c.get("id"))
        ],
        examples=[
            Example(
                input=ex["input"],
                good_output=as_opt_str(ex.get("good_output")),
                bad_output=as_opt_str(ex.get("bad_output")),
                why=as_opt_str(ex.get("why")),
            )
            for ex in as_list(out.get("examples"))
            if isinstance(ex, dict) and is_str(ex.get("input"))
        ],
    )


def synthesize_spec(project: Project, engine: Engine) -> BehaviorSpec:
    """Compiler engine: interview answers + facts → BehaviorSpec."""
    facts = "\n".join(f"- {f}" for f in project.facts) or "(none)"
    prompt = (
        f"GOAL: {project.goal}\n"
        f"TASK TYPE: {project.task_type.value}\n\n"
        f"KNOWN FACTS:\n{facts}\n\n"
        f"EXPERT'S ANSWERS:\n{_qa_block(project)}\n\n"
        "Produce the behavior specification."
    )
    out = require_object(engine.complete(prompt, system=_SPEC_SYSTEM, schema=SPEC_SCHEMA), "compiler")
    return spec_from_dict(out, goal=project.goal, task_type=project.task_type,
                          knowledge_sources=[m.path for m in project.materials])


def generate_tests(spec: BehaviorSpec, engine: Engine) -> list[TestCase]:
    """Compiler engine: spec → test cases that probe the eval criteria."""
    ids = ", ".join(c.id for c in spec.eval_criteria) or "(none)"
    prompt = (
        f"BEHAVIOR SPEC:\n{render_system_prompt(spec)}\n\n"
        f"EVAL CRITERION IDS: {ids}\n\n"
        "Write 6-10 test inputs that probe these criteria, including edge cases."
    )
    out = require_object(engine.complete(prompt, system=_TESTS_SYSTEM, schema=TESTS_SCHEMA), "compiler")
    valid_ids = {c.id for c in spec.eval_criteria}
    tests: list[TestCase] = []
    for i, t in enumerate(as_list(out.get("tests")), start=1):
        if not isinstance(t, dict) or not is_str(t.get("input")):
            continue
        # Drop criterion ids the model invented that aren't in the spec; an
        # empty list then falls back to "grade against all criteria" in run_eval
        # rather than producing an ungradeable test.
        expects = [e for e in as_list(t.get("expects")) if e in valid_ids]
        tests.append(
            TestCase(id=str(t.get("id") or f"t{i}"), input=t["input"], expects=expects,
                     notes=as_opt_str(t.get("notes")),
                     follow_ups=[f for f in as_list(t.get("follow_ups")) if is_str(f)])
        )
    return tests


# --- Deterministic renders (compiled from the spec) -------------------------

def tests_from_examples(spec: BehaviorSpec, existing: list[TestCase] = ()) -> list[TestCase]:
    """Turn the spec's examples into regression tests (§9 golden anchors).

    Each example's input becomes a test graded against all criteria — so the exact
    cases the expert cared about are pinned into the suite. Inputs already covered
    by ``existing`` are skipped — including inputs that appear as a multi-turn
    test's follow-ups (an absorbed conversation's example is its LAST turn while
    the pinned fb test starts at the FIRST; without this, examples-to-tests would
    double-pin the same exchange)."""
    seen = {t.input for t in existing} | {f for t in existing for f in t.follow_ups}
    out: list[TestCase] = []
    for i, ex in enumerate(spec.examples, start=1):
        if is_str(ex.input) and ex.input not in seen:
            seen.add(ex.input)
            out.append(TestCase(id=f"ex_{i}", input=ex.input, expects=[], notes="from spec example"))
    return out


def render_system_prompt(spec: BehaviorSpec) -> str:
    lines = [f"You are an AI for the following goal:\n{spec.goal}", ""]
    if spec.persona and (spec.persona.voice or spec.persona.reading_level):
        voice = spec.persona.voice or ""
        rl = f" (reading level: {spec.persona.reading_level})" if spec.persona.reading_level else ""
        lines += [f"VOICE: {(voice + rl).strip()}", ""]  # .strip() avoids a double space when voice is empty
    if spec.standards:
        lines.append("STANDARDS:")
        lines += [f"- {s}" for s in spec.standards]
        lines.append("")
    if spec.do_not:
        lines.append("NEVER:")
        lines += [f"- {d}" for d in spec.do_not]
        lines.append("")
    if spec.edge_cases:
        lines.append("EDGE CASES:")
        lines += [f"- When {e.situation}: {e.ruling}" for e in spec.edge_cases]
        lines.append("")
    if spec.format:
        lines += [f"FORMAT: {spec.format}", ""]
    if spec.refusal_policy:
        lines += [f"REFUSALS: {spec.refusal_policy}", ""]
    if spec.knowledge_sources:
        lines += [
            "Ground answers in the provided knowledge base; do not invent facts "
            "it does not support.",
            "",
        ]
    return "\n".join(lines).strip() + "\n"


def rubric(spec: BehaviorSpec) -> dict:
    return {
        "criteria": [
            {"id": c.id, "description": c.description, "weight": c.weight.value}
            for c in spec.eval_criteria
        ]
    }


def rag_config(spec: BehaviorSpec) -> dict:
    return {
        "knowledge_sources": list(spec.knowledge_sources),
        "index": "knowledge.lancedb",
        "table": TABLE,
        "embedder": EMBED_MODEL,
        "top_k": TOP_K,
    }


def write_build_bundle(spec: BehaviorSpec, tests: list[TestCase], project_dir: str | Path) -> list[str]:
    """Deterministically (re)write the build/ artifacts from the spec — no engine
    calls. Used by compile_project and to refresh build/ after the refine loop
    mutates the spec, so build/ never goes stale."""
    build = Path(project_dir) / "build"
    build.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def _write(name: str, content: str) -> None:
        atomic_write_text(build / name, content)
        files.append(f"build/{name}")

    _write("spec.yaml", yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    _write("system_prompt.txt", render_system_prompt(spec))
    _write("rubric.yaml", yaml.safe_dump(rubric(spec), sort_keys=False, allow_unicode=True))
    _write("rag.config.yaml", yaml.safe_dump(rag_config(spec), sort_keys=False, allow_unicode=True))
    _write(
        "tests.jsonl",
        "".join(json.dumps(t.model_dump(mode="json")) + "\n" for t in tests),
    )
    return files


# Test ids that are PINNED regressions, not regenerable synthesis output: fb_*
# are absorbed live-feedback cases (flywheel), rt_* are promoted red-team
# catches. A recompile regenerates the `t*` synthesis tests but must carry these
# forward, or the product's "a flagged case can never silently regress" promise
# is a lie. See compile_project.
_PINNED_TEST_PREFIXES = ("fb_", "rt_")


def _merge_prior_spec(prior: BehaviorSpec, spec: BehaviorSpec) -> None:
    """Carry forward spec content the fresh synthesis can't reproduce (mutates
    ``spec``): accumulated ``edge_cases``, and ``eval_criteria`` — critically any
    with a deterministic ``check`` (from ``calibrate add-check``) and any
    prior-only criterion (red-team ``rt_*``, custom). Fresh criteria win by id,
    but never at the cost of a preserved check."""
    have_ec = {(ec.situation, ec.ruling) for ec in spec.edge_cases}
    spec.edge_cases = list(spec.edge_cases) + [
        ec for ec in prior.edge_cases if (ec.situation, ec.ruling) not in have_ec
    ]
    merged: dict[str, EvalCriterion] = {c.id: c for c in spec.eval_criteria}
    for c in prior.eval_criteria:
        existing = merged.get(c.id)
        if existing is None or (c.check is not None and existing.check is None):
            merged[c.id] = c
    spec.eval_criteria = list(merged.values())


def _pin_prior_tests(prior_tests: list[TestCase], tests: list[TestCase]) -> list[TestCase]:
    """Re-pin ``fb_*`` (flywheel) / ``rt_*`` (red-team) regression tests that the
    fresh synthesis doesn't regenerate, appended to the fresh ``t*`` tests."""
    fresh_ids = {t.id for t in tests}
    pinned = [
        t for t in prior_tests
        if t.id.startswith(_PINNED_TEST_PREFIXES) and t.id not in fresh_ids
    ]
    return tests + pinned


def compile_project(project: Project, engine: Engine, *, project_dir: str | Path) -> CompileResult:
    """Synthesize the spec + tests, write the build bundle, update the project.

    Never loses what was already captured. Recompiling carries the prior spec's
    standards / never-rules / examples / edge-cases / eval-criteria (including
    deterministic ``add-check`` checks and red-team criteria) into the new spec,
    and re-pins ``fb_*`` (flywheel) and ``rt_*`` (red-team) regression tests —
    the synthesis only regenerates the ``t*`` probe tests. With no interview
    answers to synthesize from, the existing spec is preserved as-is rather than
    overwritten by a spec synthesized from an empty interview."""
    prior = project.spec
    prior_tests = list(project.tests)
    if any(it.answer for it in project.interview):
        spec = synthesize_spec(project, engine)
        if prior is not None:
            # Carry forward previously-captured rules so a recompile can't drop them.
            spec.standards = _dedup(list(spec.standards) + list(prior.standards))
            spec.do_not = _dedup(list(spec.do_not) + list(prior.do_not))
            have = {ex.input for ex in spec.examples}
            spec.examples = list(spec.examples) + [ex for ex in prior.examples if ex.input not in have]
    else:
        spec = prior or BehaviorSpec(goal=project.goal, task_type=project.task_type)
    # Merge criteria/edge-cases BEFORE generating tests, so fresh tests see the
    # preserved (e.g. red-team) criteria as valid `expects` targets.
    if prior is not None and spec is not prior:
        _merge_prior_spec(prior, spec)
    tests = _pin_prior_tests(prior_tests, generate_tests(spec, engine))
    project.spec = spec
    project.tests = tests

    files = write_build_bundle(spec, tests, project_dir)

    return CompileResult(
        standards=len(spec.standards),
        edge_cases=len(spec.edge_cases),
        criteria=len(spec.eval_criteria),
        tests=len(tests),
        build_dir=str(Path(project_dir) / "build"),
        files=files,
    )
