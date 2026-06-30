"""M1 — Ingest: parse materials, extract facts + the gap list, build the index.

The headline output is the **gap list** — the dimensions a reliable AI for this
goal needs settled that the user's materials don't answer. Those gaps drive the
interview (M2). The Extractor role (an engine) does the analysis; everything
engine-dependent lives in `extract_gaps`, which is unit-tested with a mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import rag
from .coerce import as_opt_str, is_str
from .engines.base import Engine, require_object
from .models import Gap, Material, Project
from .parsing import chunk_text, read_document

# Cap the text fed to the Extractor so a large corpus can't blow the context.
MAX_EXTRACT_CHARS = 50_000

# Structured-output schema (strict-compatible: additionalProperties:false +
# every property required), so it works across Anthropic / OpenAI / Ollama.
GAP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dimension": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
                "required": ["dimension", "why_it_matters"],
            },
        },
    },
    "required": ["facts", "gaps"],
}

_EXTRACT_SYSTEM = (
    "You analyze a user's materials to prepare for building an AI that meets "
    "their goal. Do two things: (1) extract concrete facts, rules, and standards "
    "the materials state; (2) identify GAPS — dimensions a reliable AI for this "
    "task needs settled that the materials do NOT answer (e.g. tone/voice, "
    "quality standards, do/don't rules, edge-case handling, output format, "
    "refusal & escalation policy). Only list gaps genuinely left open by the "
    "materials. Respond with JSON only, matching the provided schema."
)


@dataclass
class IngestResult:
    materials: int
    chunks: int
    facts: int
    gaps: int
    indexed: int | None  # None = vector index skipped (rag extra absent / disabled)


def _excerpt(text: str, n: int = 240) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


def parse_materials(source_dir: str | Path) -> list[tuple[Path, str]]:
    """Read every (non-hidden) file under `source_dir` into text."""
    base = Path(source_dir)
    if not base.exists():
        return []
    docs: list[tuple[Path, str]] = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            text = read_document(p)
            if text.strip():
                docs.append((p, text))
    return docs


def _join_capped(docs: list[tuple[Path, str]], cap: int) -> str:
    parts: list[str] = []
    total = 0
    for p, text in docs:
        header = f"\n\n=== {Path(p).name} ===\n"
        budget = cap - total
        if budget <= 0:
            break
        body = text[:budget]
        parts.append(header + body)
        total += len(header) + len(body)
    return "".join(parts)


def extract_gaps(
    goal: str,
    task_type: str,
    docs: list[tuple[Path, str]],
    engine: Engine,
) -> tuple[list[str], list[Gap]]:
    """Run the Extractor engine over the materials → (facts, gaps)."""
    corpus = _join_capped(docs, MAX_EXTRACT_CHARS)
    prompt = (
        f"GOAL: {goal}\n"
        f"TASK TYPE: {task_type}\n\n"
        f"MATERIALS:\n{corpus or '(none provided)'}\n\n"
        "Extract the facts and identify the gaps."
    )
    result = require_object(engine.complete(prompt, system=_EXTRACT_SYSTEM, schema=GAP_SCHEMA), "extractor")
    facts = [str(f) for f in result.get("facts", [])]
    gaps = [
        Gap(dimension=g["dimension"], why_it_matters=as_opt_str(g.get("why_it_matters")))
        for g in result.get("gaps", [])
        if isinstance(g, dict) and is_str(g.get("dimension"))
    ]
    return facts, gaps


def ingest_project(
    project: Project,
    source_dir: str | Path,
    engine: Engine,
    *,
    project_dir: str | Path | None = None,
    build_index: bool = True,
) -> IngestResult:
    """Parse materials, extract facts + gaps, optionally index — and update the
    project in place."""
    base = Path(source_dir)
    docs = parse_materials(base)

    materials: list[Material] = []
    chunks: list[dict] = []
    for p, text in docs:
        rel = str(p.relative_to(base))
        materials.append(
            Material(path=rel, kind=(p.suffix.lstrip(".") or "text"), summary=_excerpt(text))
        )
        for i, c in enumerate(chunk_text(text)):
            chunks.append({"id": f"{rel}#{i}", "text": c, "source": rel})

    facts: list[str] = []
    gaps: list[Gap] = []
    if docs:  # don't call the engine on an empty corpus
        facts, gaps = extract_gaps(project.goal, project.task_type.value, docs, engine)

    indexed: int | None = None
    if build_index and project_dir is not None and chunks:
        indexed = rag.build_index(project_dir, chunks)

    project.materials = materials
    project.facts = facts
    project.gaps = gaps
    return IngestResult(
        materials=len(materials),
        chunks=len(chunks),
        facts=len(facts),
        gaps=len(gaps),
        indexed=indexed,
    )
