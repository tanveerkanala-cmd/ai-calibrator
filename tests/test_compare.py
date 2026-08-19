"""`compare` — the thesis experiment: the same suite, specialist vs baseline.

The command's whole value is that the comparison is fair and the report is
honest, so that is what these tests pin down: the baseline really is stripped
of the specialization (prompt and retrieval), the judge grades both sides
under identical context, the estimate matches the calls actually made, a
losing specialist is reported plainly, and no side run ever pollutes the
project's real scorecard history (which `drift` and `ci` baseline against).
"""

import json
import re

from ai_calibrator.compare import (
    baseline_system,
    compare,
    compare_call_estimate,
    save_compare,
    summary_lines,
)
from ai_calibrator.eval import latest_run_id
from ai_calibrator.models import BehaviorSpec, Check, EvalCriterion, Project
from ai_calibrator.models import TestCase as Case

MARKER = "SPEC-MARKER: never guess a price"


class RecordingSubject:
    """Answers GOOD iff its system prompt carries the spec, and records systems."""

    name = "subject@test"

    def __init__(self, good_when_marker: bool = True):
        self.good_when_marker = good_when_marker
        self.systems: list = []

    def complete(self, prompt, *, system=None, schema=None):
        self.systems.append(system)
        has_marker = bool(system) and MARKER in system
        good = has_marker if self.good_when_marker else not has_marker
        return "GOOD answer" if good else "BAD answer"


class RecordingJudge:
    """Passes every listed criterion iff the output says GOOD; records systems."""

    name = "judge@test"

    def __init__(self):
        self.systems: list = []
        self.calls = 0

    def complete(self, prompt, *, system=None, schema=None):
        self.calls += 1
        self.systems.append(system)
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        good = "GOOD" in prompt
        return {"results": [
            {"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0, "rationale": ""}
            for i in ids
        ]}


def _project(n_tests: int = 2) -> Project:
    p = Project(name="p", goal="Answer store questions politely.")
    p.spec = BehaviorSpec(
        goal=p.goal,
        standards=[MARKER],
        eval_criteria=[EvalCriterion(id="c1", description="on-policy")],
    )
    p.tests = [Case(id=f"t{i}", input=f"q{i}", expects=["c1"]) for i in range(1, n_tests + 1)]
    return p


# --- what each side is actually given -------------------------------------


def test_goal_baseline_strips_the_specialization():
    subject = RecordingSubject()
    report = compare(_project(), subject, RecordingJudge(), vs="goal")
    specialist, baseline = subject.systems[:2], subject.systems[2:]
    assert all(s and MARKER in s for s in specialist)  # compiled prompt, verbatim path
    assert all(s == "Answer store questions politely." for s in baseline)  # goal ONLY
    assert report.vs == "goal"


def test_bare_baseline_sends_no_system_at_all():
    subject = RecordingSubject()
    compare(_project(), subject, RecordingJudge(), vs="bare")
    assert subject.systems[2:] == [None, None]


def test_multi_turn_baseline_is_stripped_too():
    p = _project(n_tests=1)
    p.tests[0].follow_ups = ["and then?"]
    subject = RecordingSubject()
    compare(p, subject, RecordingJudge(), vs="goal")
    assert all(s == p.goal for s in subject.systems[2:])  # every baseline turn


def test_judge_context_is_identical_on_both_sides():
    judge = RecordingJudge()
    compare(_project(), RecordingSubject(), judge, vs="bare")
    specialist, baseline = judge.systems[:2], judge.systems[2:]
    assert specialist == baseline  # same standard for both, or the delta means nothing
    assert all(MARKER in s for s in baseline)  # and it IS the user's spec


def test_baseline_never_retrieves(tmp_path, monkeypatch):
    import ai_calibrator.rag as rag
    seen_dirs = []
    real = rag.augment_system

    def spy(system, project_dir, query, top_k=rag.TOP_K):
        seen_dirs.append(project_dir)
        return real(system, project_dir, query, top_k)

    monkeypatch.setattr(rag, "augment_system", spy)
    compare(_project(), RecordingSubject(), RecordingJudge(), vs="goal", project_dir=tmp_path)
    assert seen_dirs[:2] == [tmp_path, tmp_path]  # specialist: RAG as deployed
    assert seen_dirs[2:] == [None, None]  # baseline: never


def test_vs_must_be_goal_or_bare():
    try:
        compare(_project(), RecordingSubject(), RecordingJudge(), vs="nope")
    except ValueError as exc:
        assert "goal" in str(exc) and "bare" in str(exc)
    else:
        raise AssertionError("unknown vs accepted")


def test_compile_required():
    p = Project(name="p", goal="g")
    try:
        compare(p, RecordingSubject(), RecordingJudge())
    except ValueError as exc:
        assert "compile" in str(exc)
    else:
        raise AssertionError("ran without a spec")


def test_baseline_system_variants():
    p = _project()
    assert baseline_system(p, "goal") == p.goal
    assert baseline_system(p, "bare") is None


# --- the measurement ------------------------------------------------------


def test_report_measures_the_delta():
    report = compare(_project(), RecordingSubject(), RecordingJudge())
    assert report.specialist.pass_rate == 1.0
    assert report.baseline.pass_rate == 0.0
    assert report.delta == 1.0
    assert report.n_tests == 2
    assert report.subject == "subject@test" and report.judge == "judge@test"
    assert not report.partial


def test_deterministic_criteria_split_from_judged():
    p = _project(n_tests=1)
    p.spec.eval_criteria = [
        EvalCriterion(id="c1", description="on-policy"),
        EvalCriterion(id="c2", description="says GOOD",
                      check=Check(kind="contains", value="GOOD")),
    ]
    p.tests = [Case(id="t1", input="q", expects=["c1", "c2"])]
    judge = RecordingJudge()
    report = compare(p, RecordingSubject(), judge)
    kinds = {c.id: c.kind for c in report.per_criterion}
    assert kinds == {"c1": "judged", "c2": "check"}
    by_id = {c.id: c for c in report.per_criterion}
    assert by_id["c2"].specialist_passed == 1 and by_id["c2"].baseline_passed == 0
    assert judge.calls == 2  # one per side; the check never reached the judge


def test_estimate_matches_calls_actually_made():
    subject, judge = RecordingSubject(), RecordingJudge()
    p = _project(n_tests=3)
    compare(p, subject, judge, judge_passes=2)
    estimated = compare_call_estimate(3, judge_passes=2)
    assert len(subject.systems) + judge.calls == estimated


# --- honest reporting -----------------------------------------------------


def test_a_losing_specialist_is_reported_plainly():
    report = compare(_project(), RecordingSubject(good_when_marker=False), RecordingJudge())
    assert report.delta < 0
    text = "\n".join(summary_lines(report))
    assert "baseline outperformed" in text.lower()


def test_a_tie_is_not_dressed_up_as_a_win():
    subject = RecordingSubject()
    subject.complete = lambda prompt, *, system=None, schema=None: "GOOD answer"  # both sides pass
    report = compare(_project(), subject, RecordingJudge())
    assert report.delta == 0.0
    text = "\n".join(summary_lines(report))
    assert "no measurable difference" in text.lower()


def test_smoke_runs_are_stamped_partial():
    report = compare(_project(n_tests=3), RecordingSubject(), RecordingJudge(), max_tests=1)
    assert report.partial
    assert "partial" in "\n".join(summary_lines(report)).lower()


def test_summary_discloses_the_conditions():
    report = compare(_project(), RecordingSubject(), RecordingJudge(), vs="goal")
    text = "\n".join(summary_lines(report))
    assert "subject@test" in text and "judge@test" in text
    assert "goal" in text  # which baseline this delta is against


# --- state safety ---------------------------------------------------------


def test_never_pollutes_the_run_history(tmp_path):
    report = compare(_project(), RecordingSubject(), RecordingJudge(), project_dir=tmp_path)
    save_compare(tmp_path, report)
    assert latest_run_id(tmp_path) is None  # drift/ci baselines untouched
    saved = json.loads((tmp_path / "evals" / "compare.json").read_text())
    assert saved["specialist"]["pass_rate"] == 1.0
    assert saved["vs"] == "goal"


# --- CLI ------------------------------------------------------------------


def test_cli_compare_end_to_end(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from ai_calibrator import engines
    from ai_calibrator.cli import app
    from ai_calibrator.store import save_project

    class DualEngine:
        """Subject on plain calls, all-pass judge on schema calls."""
        name = "m@test"

        def complete(self, prompt, *, system=None, schema=None):
            if schema is not None:
                ids = re.findall(r"^- (\S+):", prompt, re.M)
                good = "GOOD" in prompt
                return {"results": [{"criterion_id": i, "passed": good,
                                     "score": 1.0, "rationale": ""} for i in ids]}
            return "GOOD answer" if (system and MARKER in system) else "BAD answer"

    p = _project()
    p.engines.subject = "m@ollama"
    p.engines.judge = "m@ollama"
    save_project(p, tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: DualEngine())

    result = CliRunner().invoke(app, ["compare", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "engine call(s)" in result.output  # spend before spending
    assert "grades its own" in result.output  # judge == subject warning
    assert (tmp_path / "evals" / "compare.json").exists()


def test_cli_rejects_unknown_vs(tmp_path):
    from typer.testing import CliRunner

    from ai_calibrator.cli import app
    from ai_calibrator.store import save_project

    save_project(_project(), tmp_path)
    result = CliRunner().invoke(app, ["compare", str(tmp_path), "--vs", "chaos"])
    assert result.exit_code != 0
