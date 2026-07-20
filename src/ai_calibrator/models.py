"""Data contracts — the spine of the Calibration Core.

Every pipeline stage reads and writes these typed models, so stages stay
decoupled and a project is just serializable data on disk. `BehaviorSpec` is the
source of truth; the system prompt, RAG config, rubric, and tests all compile
from it.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# YAML re-interprets these Unicode line separators as line breaks, so a value
# containing one does NOT survive a project.yaml round-trip (save→load changes
# it). They can arrive from an ingested document (PDF/DOCX line separators).
# Normalizing them to "\n" at construction is semantic-preserving and makes every
# persisted field round-trip stably. (Hypothesis property test found this.)
_YAML_LINE_SEPARATORS = str.maketrans({"\u0085": "\n", "\u2028": "\n", "\u2029": "\n"})


def _normalize_yaml_text(v: object) -> object:
    if isinstance(v, str):
        return v.translate(_YAML_LINE_SEPARATORS)
    if isinstance(v, list):
        return [_normalize_yaml_text(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize_yaml_text(x) for k, x in v.items()}
    return v


class PreservingModel(BaseModel):
    """Base for every model persisted to disk (project.yaml / scorecard.json).

    ``extra="allow"`` keeps unknown fields — a hand-edit typo, or a field written
    by a NEWER calibrator version — through load→save instead of silently
    destroying them (pydantic's default ``ignore`` drops extras on the next
    save). The data is inert to this version but survives; ``calibrate lint``
    flags it so a typo doesn't hide forever."""
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _strip_yaml_hostile_separators(cls, data: object) -> object:
        # Only the exotic U+0085/U+2028/U+2029 separators, normalized to "\n" so
        # every stored string survives the YAML round-trip unchanged.
        return _normalize_yaml_text(data) if isinstance(data, dict) else data


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

    @property
    def numeric(self) -> int:
        """Scoring weight for the weighted score: high=3, medium=2, low=1."""
        return {"low": 1, "medium": 2, "high": 3}[self.value]


# --- Inputs gathered from the user -----------------------------------------

class Material(PreservingModel):
    """A source file the user uploaded."""
    path: str
    kind: str = "document"
    summary: str | None = None


class Gap(PreservingModel):
    """A dimension the materials don't settle — a candidate interview topic."""
    dimension: str
    why_it_matters: str | None = None


class InterviewItem(PreservingModel):
    """One adaptive question, its drafted answer, and the user's ratified answer."""
    id: str
    dimension: str
    question: str
    draft_answer: str | None = None      # propose-and-ratify: the tool's guess
    answer: str | None = None            # the user's confirmed/corrected answer
    rationale: str | None = None         # why the tool asked (teach-while-scaffolding)


# --- The compiled behavior spec (source of truth) --------------------------

class Persona(PreservingModel):
    voice: str | None = None
    reading_level: str | None = None


class EdgeCase(PreservingModel):
    situation: str
    ruling: str


class Example(PreservingModel):
    input: str
    good_output: str | None = None
    bad_output: str | None = None
    why: str | None = None


class Check(PreservingModel):
    """A deterministic (code-graded) check for a criterion — exact, no LLM (§9)."""
    kind: Literal["contains", "not_contains", "regex", "max_chars", "min_chars", "non_empty"]
    value: str = ""


class EvalCriterion(PreservingModel):
    id: str
    description: str
    weight: Weight = Weight.MEDIUM
    # If set, the criterion is graded deterministically by this check instead of
    # by the LLM judge — cheaper and exact for objectively-verifiable behavior.
    check: Check | None = None


class BehaviorSpec(PreservingModel):
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

class TestCase(PreservingModel):
    id: str
    input: str                                          # the first (or only) user turn
    expects: list[str] = Field(default_factory=list)    # EvalCriterion ids
    notes: str | None = None
    # Subsequent user turns for a multi-turn conversation test. Empty = single-turn.
    follow_ups: list[str] = Field(default_factory=list)


class CriterionResult(PreservingModel):
    criterion_id: str
    passed: bool
    score: float = 0.0
    rationale: str | None = None
    # Judge agreement when graded with multiple passes (self-consistency): the
    # fraction of passes that agreed with the majority verdict. None = single pass.
    confidence: float | None = None
    # The criterion's weight AT GRADING TIME — recorded so the scorecard stays
    # self-contained (honest) even if the spec's weights change later.
    # None = graded by a version that didn't record weights (treated as medium).
    weight: Weight | None = None


class TestResult(PreservingModel):
    test_id: str
    output: str
    criteria: list[CriterionResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

    @property
    def weighted_score(self) -> float:
        """Weight-honest score in [0,1]: Σ(weight·score)/Σ(weight) over the
        test's criteria (high=3, medium=2, low=1; unrecorded → medium).

        Pass/fail stays binary — ANY failing criterion fails the test — but this
        says HOW it failed: 0.85 means only low-weight criteria missed; 0.25
        means the important ones did."""
        if not self.criteria:
            return 0.0
        weights = [(c.weight or Weight.MEDIUM).numeric for c in self.criteria]
        return sum(w * c.score for w, c in zip(weights, self.criteria, strict=True)) / sum(weights)


class Scorecard(PreservingModel):
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

    @property
    def weighted_score(self) -> float:
        """Mean weighted_score over graded tests — the weight-honest companion to
        pass_rate (which stays the binary headline number)."""
        graded = [r for r in self.results if r.criteria]
        if not graded:
            return 0.0
        return sum(r.weighted_score for r in graded) / len(graded)


# --- Engine bindings (role -> "model@provider") ----------------------------

# Cloud (Claude) is the default for quality. Reasoning roles use Opus; the
# high-volume judge uses cheap/fast Haiku (see ARCHITECTURE §5.1). Point any
# role at "<model>@ollama" to run locally with no key. The repo ships no keys —
# the user supplies their own ANTHROPIC_API_KEY.
_REASONING = "claude-opus-4-8@anthropic"
_JUDGE = "claude-haiku-4-5@anthropic"
_SUBJECT = "claude-sonnet-4-6@anthropic"


class EngineBinding(PreservingModel):
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

# Reserved on Windows — a folder can't be named any of these (case-insensitive,
# with or without an extension), so a project name that is one would break there.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


class Project(PreservingModel):
    name: str
    goal: str
    task_type: TaskType = TaskType.ASSISTANT

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("project name must be a non-empty string")
        v = v.strip()
        if len(v) > 120:
            # the name becomes a directory name; filesystems cap components at
            # ~255 bytes — fail here with a clear message, not an OSError later
            raise ValueError("project name too long (max 120 characters)")
        # The name becomes a directory component. Reject path separators, the
        # Windows-reserved characters (\ / : * ? " < > |), control chars, and the
        # . / .. specials — so a name valid on POSIX can't create an invalid path
        # (or traverse) on Windows, and vice versa. Fail here, cross-platform,
        # rather than as a confusing OSError at mkdir time.
        bad = set('/\\:*?"<>|') & set(v)
        if bad or any(ord(c) < 32 for c in v):
            raise ValueError(
                "project name may not contain path separators or any of \\ / : * ? \" < > | "
                "(it becomes a folder name); use letters, digits, spaces, - or _"
            )
        if v in (".", ".."):
            raise ValueError("project name cannot be '.' or '..'")
        if v[-1] in " .":  # Windows silently strips a trailing space/dot from a path component
            raise ValueError("project name may not end with a space or '.'")
        # Windows reserved DEVICE names (case-insensitive, with or without an
        # extension) can't be a folder there — reject cross-platform.
        if v.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"project name {v!r} is a reserved device name on Windows — pick another")
        return v  # normalized: stripped, <= 120, filesystem-safe on every platform
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
