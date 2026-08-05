"""Spec-lint — catch quality problems in a behavior spec *before* you eval.

Evaluation tells you whether the AI meets the spec; linting tells you whether the
*spec itself* is any good. Cheap, mostly deterministic checks surface the issues
that quietly waste an eval run: no measurable criteria, criteria nothing tests,
vague/unfalsifiable standards, duplicates, a missing refusal policy. The optional
``--deep`` pass reuses the multi-stakeholder conflict detector on the spec's own
rules to find **self-contradictions** ("be concise" vs "always explain in depth").

CI-friendly: errors mean the spec isn't ready to calibrate against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from .coverage import analyze_coverage
from .engines.base import Engine
from .models import BehaviorSpec, Project, TestCase

# Vague, unfalsifiable words that make a standard hard to test objectively.
_WEASEL_WORDS = {"good", "bad", "nice", "appropriate", "appropriately", "helpful",
                 "proper", "properly", "reasonable", "reasonably", "relevant", "etc"}
_WEASEL_PHRASES = ("as needed", "as appropriate", "when necessary", "and so on", "etc.")
_MIN_LEN = 12  # a standard/criterion shorter than this is almost certainly untestable


@dataclass
class LintIssue:
    code: str
    severity: str   # "error" | "warn" | "info"
    message: str
    where: str | None = None


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _weasel(text: str) -> str | None:
    low = text.lower()
    for phrase in _WEASEL_PHRASES:
        if phrase in low:
            return phrase
    words = {w.strip(".,;:!?\"'()") for w in low.split()}
    hit = words & _WEASEL_WORDS
    return next(iter(hit)) if hit else None


def lint_spec(spec: BehaviorSpec, tests: list[TestCase]) -> LintReport:
    """Deterministic quality checks on a spec + its tests (no engine)."""
    issues: list[LintIssue] = []

    if not spec.standards and not spec.do_not and not spec.edge_cases:
        issues.append(LintIssue("no_rules", "warn", "Spec states no standards, never-rules, or edge cases."))
    if not spec.eval_criteria:
        issues.append(LintIssue("no_criteria", "error", "No eval criteria — there is nothing to grade against."))
    if not tests:
        issues.append(LintIssue("no_tests", "warn", "No tests — run `calibrate compile` (or `import`) to generate them."))

    # Duplicate ids / rules.
    seen_ids: set[str] = set()
    for c in spec.eval_criteria:
        if c.id in seen_ids:
            issues.append(LintIssue("duplicate_criterion", "error", f"Duplicate criterion id {c.id!r}.", c.id))
        seen_ids.add(c.id)
    for label, items in (("standard", spec.standards), ("never-rule", spec.do_not)):
        for x in {i for i in items if items.count(i) > 1}:
            issues.append(LintIssue("duplicate_rule", "warn", f"Duplicate {label}: {x[:50]!r}."))

    # Vague / unfalsifiable standards + criteria.
    for s in spec.standards:
        if len(s.strip()) < _MIN_LEN:
            issues.append(LintIssue("vague_standard", "warn", f"Standard is too short to test: {s!r}.", s))
        elif (w := _weasel(s)):
            issues.append(LintIssue("vague_standard", "info", f"Standard leans on vague word {w!r}: {s[:60]!r}.", s))
    for c in spec.eval_criteria:
        if len(c.description.strip()) < _MIN_LEN:
            issues.append(LintIssue("vague_criterion", "warn",
                                    f"Criterion {c.id!r} description is too short to grade reliably.", c.id))


    # Duplicate TEST ids. Every surface that compares a run to the suite — or to
    # another run — collapses results into a dict keyed by test_id (see
    # `identity`), so a duplicate makes one of the two invisible to regression
    # and golden-output checking: a pinned anchor that can no longer fail the
    # gate. The content check in `identity` narrows which results may be
    # compared; it does not make two tests sharing an id distinguishable. An
    # error, so `ci` refuses to certify the suite.
    seen_tests: set[str] = set()
    dupe_tests: set[str] = set()
    for t in tests:
        (dupe_tests if t.id in seen_tests else seen_tests).add(t.id)
    for tid in sorted(dupe_tests):
        issues.append(LintIssue(
            "duplicate_test_id", "error",
            f"Test id {tid!r} appears more than once — drift and snapshot key results by id, "
            "so only one of them is ever compared. Give each test a unique id.", tid))

    # Untested criteria + orphan expectations (reuse the coverage analysis).
    cov = analyze_coverage(spec, tests)
    for uncovered in cov.uncovered_criteria:
        issues.append(LintIssue("untested_criterion", "warn",
                                f"Criterion {uncovered.id!r} has no targeted test.", uncovered.id))
    # An `expects` naming a criterion the spec doesn't have grades against nothing:
    # the test still runs (and still costs an engine call) but contributes no
    # verdict, so it silently leaves the pass rate's denominator. That is a green
    # gate over untested behavior — an error, so `ci` refuses to certify it.
    for cid in sorted(set(cov.orphan_expectations)):
        who = [t.id for t in tests if cid in (t.expects or [])]
        issues.append(LintIssue(
            "orphan_expectation", "error",
            f"Test(s) {', '.join(who[:5])} expect criterion {cid!r}, which is not in the spec — "
            "they run but are never graded. Add the criterion, or retarget the tests.", cid))

    # A 'never'-heavy spec with no refusal policy usually wants one.
    if spec.do_not and not spec.refusal_policy:
        issues.append(LintIssue("no_refusal_policy", "info",
                                "Spec has never-rules but no refusal policy — define how it should decline."))
    return LintReport(issues)


def _walk_extras(value: object, path: str, out: list[tuple[str, str]]) -> None:
    if isinstance(value, BaseModel):
        for key in (value.model_extra or {}):
            out.append((path, key))
        for name in type(value).model_fields:
            _walk_extras(getattr(value, name), f"{path}.{name}", out)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _walk_extras(item, f"{path}[{i}]", out)


def lint_unknown_fields(project: Project) -> list[LintIssue]:
    """Flag fields in project.yaml this version doesn't recognize.

    They're preserved through save (see models.PreservingModel) but ignored by
    every computation — usually a hand-edit typo, sometimes a file written by a
    newer calibrator. Either way the owner should know."""
    found: list[tuple[str, str]] = []
    _walk_extras(project, "project", found)
    return [
        LintIssue("unknown_field", "warn",
                  f"Unrecognized field {key!r} at {path} — kept in the file but ignored "
                  "by this version (a typo, or written by a newer calibrator?).",
                  f"{path}.{key}")
        for path, key in found
    ]


def lint_schema_version(project: Project) -> list[LintIssue]:
    """Flag a project.yaml written by a NEWER calibrator than this one.

    `PreservingModel` keeps fields this build doesn't recognize, and
    `lint_unknown_fields` names them — but only fields it has never heard of.
    A change in what an EXISTING field means reads as ordinary data, so the
    version marker is the only thing that can say "this file is from the
    future". Loading continues: refusing outright would strand a user whose
    file is fine in every way that matters to them.
    """
    from .models import SCHEMA_VERSION

    version = getattr(project, "schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or version <= SCHEMA_VERSION:
        return []
    return [LintIssue(
        "future_schema_version", "warn",
        f"project.yaml declares schema_version {version}, but this calibrator understands "
        f"{SCHEMA_VERSION}. It was written by a newer version: fields this build does not "
        "know are preserved, but anything whose MEANING changed is being read the old way. "
        "Upgrade with `pip install -U ai-calibrator`.",
        "schema_version")]


def lint_engine_roles(project: Project) -> list[LintIssue]:
    """Flag a judge that is the same model as the subject it grades.

    The product's claim is that the AI is TESTED. When one model both answers
    and grades, the pass rate measures that model's opinion of itself: its
    blind spots are shared by construction, so the failures it cannot see are
    exactly the ones it cannot report. A shared idiosyncrasy reads as agreement
    and the certificate says 100%.

    This is the most attackable number the tool produces, and it is easy to
    reach by accident — `calibrate engines <project> --all <model>` points every
    role at one model in a single command, and the local-model quickstart in the
    README does precisely that. A warning, not an error: it is a real and
    reasonable way to work when no second model is available (and for a
    deterministic `check` the judge is not consulted at all), so it must not
    block the gate. It must, however, be said out loud rather than discovered by
    someone reading the config.
    """
    engines = project.engines
    subject, judge = (engines.subject or "").strip(), (engines.judge or "").strip()
    if not subject or not judge or subject.casefold() != judge.casefold():
        return []
    graded_by_judge = 0
    if project.spec is not None:
        graded_by_judge = sum(1 for c in project.spec.eval_criteria if c.check is None)
    if not graded_by_judge:
        # Every criterion is checked deterministically: no judge call is made,
        # so the models being identical decides nothing.
        return []
    return [LintIssue(
        "judge_is_subject", "warn",
        f"The judge and the subject are the same model ({subject}) — it grades its own "
        f"answers on {graded_by_judge} criterion/criteria, so the pass rate is that "
        "model's opinion of itself and shares its blind spots. Bind a different model "
        "with `calibrate engines <project> judge <model@provider>`, or convert the "
        "criteria to deterministic checks with `calibrate add-check`.",
        "engines.judge")]


def lint_contradictions(spec: BehaviorSpec, engine: Engine) -> list[LintIssue]:
    """Engine pass: find rules within the spec that contradict each other.

    Reuses the multi-stakeholder conflict detector, treating the single spec as
    one 'stakeholder' — a self-contradictory spec can never be fully satisfied."""
    from .stakeholders import detect_conflicts, gather
    statements = gather({"spec": spec})
    return [
        LintIssue("self_contradiction", "error",
                  f'Contradiction: "{c.a.text}" vs "{c.b.text}" — {c.explanation}')
        for c in detect_conflicts(statements, engine)
    ]


def lint_dict(report: LintReport) -> dict:
    return {
        "ok": report.ok,
        "errors": len(report.errors),
        "warnings": len(report.warnings),
        "issues": [{"code": i.code, "severity": i.severity, "message": i.message, "where": i.where}
                   for i in report.issues],
    }
