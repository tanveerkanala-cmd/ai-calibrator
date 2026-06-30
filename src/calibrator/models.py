"""Data contracts — the spine of the Calibration Core.

Every pipeline stage reads and writes these typed models, so stages stay
decoupled and a project is just serializable data on disk. `BehaviorSpec` is the
source of truth; the system prompt, RAG config, rubric, and tests all compile
from it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    ASSISTANT = "assistant"
    SUPPORT_ASSISTANT = "support_assistant"
    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    WRITER = "writer"
    AGENT = "agent"
    OTHER = "other"


class Weight(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --- Inputs gathered from the user -----------------------------------------

class Material(BaseModel):
    """A source file the user uploaded."""
    path: str
    kind: str = "document"
    summary: str | None = None


class Gap(BaseModel):
    """A dimension the materials don't settle — a candidate interview topic."""
    dimension: str
    why_it_matters: str | None = None


class InterviewItem(BaseModel):
    """One adaptive question, its drafted answer, and the user's ratified answer."""
    id: str
    dimension: str
    question: str
    draft_answer: str | None = None      # propose-and-ratify: the tool's guess
    answer: str | None = None            # the user's confirmed/corrected answer
    rationale: str | None = None         # why the tool asked (teach-while-scaffolding)


# --- The compiled behavior spec (source of truth) --------------------------

class Persona(BaseModel):
    voice: str | None = None
    reading_level: str | None = None


class EdgeCase(BaseModel):
    situation: str
    ruling: str


class Example(BaseModel):
    input: str
    good_output: str | None = None
    bad_output: str | None = None
    why: str | None = None


class EvalCriterion(BaseModel):
    id: str
    description: str
    weight: Weight = Weight.MEDIUM


class BehaviorSpec(BaseModel):
    goal: str
    task_type: TaskType = TaskType.ASSISTANT
    persona: Persona = Field(default_factory=Persona)
    standards: list[str] = Field(default_factory=list)
    do_not: list[str] = Field(default_factory=list)
    edge_cases: list[EdgeCase] = Field(default_factory=list)
    format: str | None = None
    refusal_policy: str | None = None
    knowledge_sources: list[str] = Field(default_factory=list)
    eval_criteria: list[EvalCriterion] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)


# --- Evaluation ------------------------------------------------------------

class TestCase(BaseModel):
    id: str
    input: str
    expects: list[str] = Field(default_factory=list)   # EvalCriterion ids
    notes: str | None = None


class CriterionResult(BaseModel):
    criterion_id: str
    passed: bool
    score: float = 0.0
    rationale: str | None = None


class TestResult(BaseModel):
    test_id: str
    output: str
    criteria: list[CriterionResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)


class Scorecard(BaseModel):
    run_id: str
    results: list[TestResult] = Field(default_factory=list)

    @field_validator("run_id")
    @classmethod
    def _safe_run_id(cls, v: str) -> str:
        # run_id becomes a directory name under evals/ — keep it a safe, non-empty
        # path component (no separators / traversal / NUL).
        if not isinstance(v, str) or not v.strip():
            raise ValueError("run_id must be a non-empty string")
        if "/" in v or "\\" in v or ".." in v or "\x00" in v:
            raise ValueError("run_id must not contain path separators or '..'")
        return v

    @property
    def pass_rate(self) -> float:
        # Only count gradeable tests (those with criteria). A test with no
        # criteria was never actually graded, so it must not be folded into the
        # denominator as a silent failure.
        graded = [r for r in self.results if r.criteria]
        if not graded:
            return 0.0
        return sum(1 for r in graded if r.passed) / len(graded)


# --- Engine bindings (role -> "model@provider") ----------------------------

# Cloud (Claude) is the default for quality. Reasoning roles use Opus; the
# high-volume judge uses cheap/fast Haiku (see ARCHITECTURE §5.1). Point any
# role at "<model>@ollama" to run locally with no key. The repo ships no keys —
# the user supplies their own ANTHROPIC_API_KEY.
_REASONING = "claude-opus-4-8@anthropic"
_JUDGE = "claude-haiku-4-5@anthropic"
_SUBJECT = "claude-sonnet-4-6@anthropic"


class EngineBinding(BaseModel):
    """Which engine powers each role. Cloud (Claude) default, BYO key via
    ANTHROPIC_API_KEY; point any role at "<model>@ollama" to run locally."""
    extractor: str = _REASONING
    interviewer: str = _REASONING
    predictor: str = _REASONING
    compiler: str = _REASONING
    judge: str = _JUDGE
    # the model the *configured AI* runs on (the "subject" being evaluated) —
    # distinct from the engine roles above
    subject: str = _SUBJECT


# --- The project (everything, serializable) --------------------------------

class Project(BaseModel):
    name: str
    goal: str
    task_type: TaskType = TaskType.ASSISTANT

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("project name must be a non-empty string")
        return v
    materials: list[Material] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    interview: list[InterviewItem] = Field(default_factory=list)
    spec: BehaviorSpec | None = None
    tests: list[TestCase] = Field(default_factory=list)
    engines: EngineBinding = Field(default_factory=EngineBinding)
    # Opt-in: log this project's engine decisions locally so the Engine-Trainer
    # can later localize a role onto your own model. Off by default; stays on disk.
    log_interactions: bool = False
