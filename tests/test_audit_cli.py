"""CLI + interview invariants: never destroy work, never report a number nobody measured.

Every test here pins a user-visible contract the CLI already documents — ratified
answers survive a failed regeneration, a mistyped path changes nothing, and a
comparison that could not be made is refused instead of scored.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ai_calibrator.cli import app

runner = CliRunner()


def _has_no_traceback(result) -> bool:
    """See the twin in ``test_cli_robustness``: asked of the RESULT, because
    CliRunner stores an unhandled exception on ``result.exception`` and writes
    nothing to the output, which made the old output-string check unfalsifiable."""
    exc = result.exception
    return exc is None or isinstance(exc, SystemExit)


# --- interview: ratified answers survive a mid-run engine failure ----------

def _answered_project(tmp_path):
    from ai_calibrator.models import Gap, InterviewItem, Project
    from ai_calibrator.store import save_project

    project = Project(name="p", goal="g")
    project.gaps = [Gap(dimension=d) for d in ("escalation", "pricing", "tone", "format")]
    project.interview = [
        InterviewItem(id="q1", dimension="escalation", question="?", answer="page the on-call"),
        InterviewItem(id="q2", dimension="pricing", question="?"),
        InterviewItem(id="q3", dimension="tone", question="?", answer="warm, never apologetic"),
        InterviewItem(id="q4", dimension="format", question="?", answer="three bullets, then a TL;DR"),
    ]
    save_project(project, tmp_path)
    return project


class _BoomEngine:
    """Drafts nothing: the first gap that needs the engine fails."""
    name = "boom@test"

    def complete(self, prompt, *, system=None, schema=None):
        raise RuntimeError("engine timed out")


class _DraftEngine:
    name = "drafty@test"

    def complete(self, prompt, *, system=None, schema=None):
        return {"question": "q?", "draft_answer": "a model guess", "rationale": "why"}


def test_progress_snapshot_carries_every_answer(tmp_path):
    """Each incremental snapshot is persisted verbatim, so a snapshot that has not
    folded in the later answers yet would write them out of the project."""
    from ai_calibrator.interview import generate_questions

    project = _answered_project(tmp_path)
    originals = {it.answer for it in project.interview if it.answer}
    seen: list[set] = []
    generate_questions(
        project, _DraftEngine(),
        on_progress=lambda items, done, total: seen.append({i.answer for i in items if i.answer}),
    )
    assert len(seen) == 4
    for snapshot in seen:
        assert snapshot >= originals


def test_regenerate_keeps_answers_when_the_engine_fails(tmp_path, monkeypatch):
    import ai_calibrator.engines as engines
    from ai_calibrator.store import load_project

    _answered_project(tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _BoomEngine())

    result = runner.invoke(app, ["interview", str(tmp_path), "--regenerate"])
    assert result.exit_code == 1, result.output
    assert _has_no_traceback(result)

    after = load_project(tmp_path)
    assert {it.answer for it in after.interview if it.answer} == {
        "page the on-call", "warm, never apologetic", "three bullets, then a TL;DR"
    }


def test_regenerate_after_a_failure_does_not_overwrite_answers_with_drafts(tmp_path, monkeypatch):
    """The advertised recovery re-run must not quietly replace ratified answers
    with model drafts — that is a worse outcome than the visible failure."""
    import ai_calibrator.engines as engines
    from ai_calibrator.store import load_project

    _answered_project(tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _BoomEngine())
    runner.invoke(app, ["interview", str(tmp_path), "--regenerate"])

    monkeypatch.setattr(engines, "get_engine", lambda spec: _DraftEngine())
    result = runner.invoke(app, ["interview", str(tmp_path), "--regenerate", "--accept-drafts"])
    assert result.exit_code == 0, result.output

    answers = {it.dimension: it.answer for it in load_project(tmp_path).interview}
    assert answers["escalation"] == "page the on-call"
    assert answers["tone"] == "warm, never apologetic"
    assert answers["format"] == "three bullets, then a TL;DR"
    assert answers["pricing"] == "a model guess"  # the only gap that was ever unanswered


# --- ingest: a mistyped --source is a typo, not "delete everything" --------

class _ExtractEngine:
    name = "fake@test"

    def complete(self, prompt, *, system=None, schema=None):
        return {"facts": [], "gaps": []}


def _ingested_project(tmp_path):
    from ai_calibrator.models import Gap, Material, Project
    from ai_calibrator.store import save_project

    project = Project(name="p", goal="g")
    project.materials = [Material(path="policy.md", kind="md", summary="returns policy")]
    project.facts = ["Returns are accepted for 30 days."]
    project.gaps = [Gap(dimension="tone")]
    save_project(project, tmp_path)
    (tmp_path / "materials" / "policy.md").write_text("Returns are accepted for 30 days.")
    (tmp_path / "knowledge.lancedb").mkdir()  # index built from those files
    return project


def test_ingest_refuses_a_source_directory_that_does_not_exist(tmp_path, monkeypatch):
    """A path that isn't there is a typo (or the wrong working directory), not the
    deliberate "I deleted every material" that clears the corpus."""
    import ai_calibrator.engines as engines
    from ai_calibrator.store import load_project

    _ingested_project(tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _ExtractEngine())

    result = runner.invoke(app, ["ingest", str(tmp_path), "--source", str(tmp_path / "matrials")])
    assert result.exit_code == 1, result.output
    assert "does not exist" in result.output
    assert _has_no_traceback(result)

    after = load_project(tmp_path)
    assert len(after.materials) == 1 and after.facts and after.gaps
    assert (tmp_path / "knowledge.lancedb").exists()


def test_ingest_of_an_empty_source_directory_still_clears_the_corpus(tmp_path, monkeypatch):
    import ai_calibrator.engines as engines
    from ai_calibrator.store import load_project

    _ingested_project(tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _ExtractEngine())
    empty = tmp_path / "elsewhere"
    empty.mkdir()

    result = runner.invoke(app, ["ingest", str(tmp_path), "--source", str(empty)])
    assert result.exit_code == 0, result.output
    assert "is empty — clearing" in result.output

    after = load_project(tmp_path)
    assert after.materials == [] and after.facts == [] and after.gaps == []


def test_ingest_no_index_says_an_earlier_index_is_still_being_queried(tmp_path, monkeypatch):
    """--no-index skips the rebuild, not the retrieval: whatever was indexed before
    keeps feeding eval and run, so silence here reads as "retrieval is off"."""
    import ai_calibrator.engines as engines

    _ingested_project(tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _ExtractEngine())

    result = runner.invoke(app, ["ingest", str(tmp_path), "--no-index"])
    assert result.exit_code == 0, result.output
    assert "skipped (--no-index)" in result.output
    assert "still in place and still feeds every eval" in result.output
    assert _has_no_traceback(result)


def test_ingest_no_index_stays_quiet_when_no_index_was_ever_built(tmp_path, monkeypatch):
    import ai_calibrator.engines as engines

    _ingested_project(tmp_path)
    (tmp_path / "knowledge.lancedb").rmdir()
    monkeypatch.setattr(engines, "get_engine", lambda spec: _ExtractEngine())

    result = runner.invoke(app, ["ingest", str(tmp_path), "--no-index"])
    assert result.exit_code == 0, result.output
    assert "skipped (--no-index)" in result.output
    assert "still in place" not in result.output


# --- eval: the tests that were never graded are part of the story ---------

def test_eval_reports_the_tests_it_could_not_grade(tmp_path, monkeypatch):
    """A test whose `expects` names no criterion in the spec is dropped from the
    rate's denominator — honest, but only if the count that was dropped is said."""
    import re

    import ai_calibrator.engines as engines
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project
    from ai_calibrator.models import TestCase as CaseModel
    from ai_calibrator.store import save_project

    class _SubjectAndJudge:
        """Answers as the subject; passes every criterion as the judge."""
        name = "fake@test"

        def complete(self, prompt, *, system=None, schema=None):
            if schema is None:
                return "the documented policy"
            return {"results": [{"criterion_id": cid, "passed": True, "score": 1.0, "rationale": "r"}
                                for cid in re.findall(r"^- (\S+):", prompt, re.M)]}

    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(goal="g", standards=["Answer with the documented policy."],
                                eval_criteria=[EvalCriterion(id="c1", description="d")])
    project.tests = ([CaseModel(id=f"t{i}", input="q", expects=["c1"]) for i in range(1, 6)]
                     + [CaseModel(id=f"u{i}", input="q", expects=["c_gone"]) for i in range(1, 4)])
    save_project(project, tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _SubjectAndJudge())

    result = runner.invoke(app, ["eval", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "5/5" in result.output
    assert "3 of 8" in result.output and "not graded" in result.output
    assert _has_no_traceback(result)


# --- merge: a relative destination is the CLI's own idiom for "here" -------

@pytest.mark.parametrize("dest, from_sub", [(".", False), ("..", True)])
def test_merge_into_a_relative_destination_keeps_the_rulings(tmp_path, monkeypatch, dest, from_sub):
    """The merged project's name comes from the destination's basename, which is
    empty for `.` — and it is read AFTER the reconciliation loop, so a rejection
    there throws away every conflict ruling the user typed."""
    import os

    import yaml

    import ai_calibrator.stakeholders as stake
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import load_project, save_project

    class _Eng:
        name = "fake@test"

    a, b = tmp_path / "a", tmp_path / "b"
    save_project(Project(name="a", goal="g", spec=BehaviorSpec(goal="g", standards=["Always apologize first."])), a)
    save_project(Project(name="b", goal="g", spec=BehaviorSpec(goal="g", standards=["Never apologize."])), b)
    monkeypatch.setattr("ai_calibrator.engines.get_engine", lambda spec: _Eng())
    monkeypatch.setattr(stake, "detect_conflicts", lambda statements, engine: [stake.Conflict(
        id="C1", a=statements[0], b=statements[1],
        explanation="one demands an apology, the other forbids it", severity="high")])

    out = tmp_path / "merged"
    cwd = out / "sub" if from_sub else out
    cwd.mkdir(parents=True)
    here = os.getcwd()
    try:
        os.chdir(cwd)
        r = runner.invoke(app, ["merge", dest, "--from", str(a), "--from", str(b)],
                          input="b\nlegal overruled support\n")
    finally:
        os.chdir(here)

    assert r.exit_code == 0, r.output
    assert _has_no_traceback(r)
    assert load_project(out).name == "merged"
    audit = yaml.safe_load((out / "reconciliation.yaml").read_text(encoding="utf-8"))
    assert audit["conflicts"][0]["rationale"] == "legal overruled support"


def test_merge_rejects_an_unusable_destination_before_reconciling(tmp_path, monkeypatch):
    """A destination whose basename can't be a project name must fail up front —
    finding out after the reconciliation loop costs the user every ruling."""
    import ai_calibrator.stakeholders as stake
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import save_project

    class _Eng:
        name = "fake@test"

    a, b = tmp_path / "a", tmp_path / "b"
    save_project(Project(name="a", goal="g", spec=BehaviorSpec(goal="g", standards=["x"])), a)
    save_project(Project(name="b", goal="g", spec=BehaviorSpec(goal="g", standards=["y"])), b)
    monkeypatch.setattr("ai_calibrator.engines.get_engine", lambda spec: _Eng())
    monkeypatch.setattr(stake, "detect_conflicts", lambda statements, engine: [])

    r = runner.invoke(app, ["merge", str(tmp_path / "CON"),  # reserved device name on Windows
                            "--from", str(a), "--from", str(b)])
    assert r.exit_code == 1, r.output
    assert "reserved device name" in r.output
    assert "Analyzing" not in r.output  # bailed before any reconciliation work
    assert _has_no_traceback(r)


# --- finetune --gate: a comparison that never happened is not a win -------

def _gate_project(tmp_path, baseline_ids):
    """A project whose baseline run graded ``baseline_ids``; the candidate ran the
    whole suite (the two memorized ex_* plus the later-added rt_*)."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import (BehaviorSpec, CriterionResult, EvalCriterion, Example,
                                      Project, Scorecard, TestResult)
    from ai_calibrator.models import TestCase as CaseModel
    from ai_calibrator.store import save_project

    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(
        goal="g",
        eval_criteria=[EvalCriterion(id="c1", description="d")],
        examples=[Example(input="memorized 1", good_output="a", source="human"),
                  Example(input="memorized 2", good_output="a", source="human")],
    )
    project.tests = [CaseModel(id="ex_1", input="memorized 1", expects=["c1"]),
                     CaseModel(id="ex_2", input="memorized 2", expects=["c1"]),
                     CaseModel(id="rt_1", input="unseen 1", expects=["c1"]),
                     CaseModel(id="rt_2", input="unseen 2", expects=["c1"])]
    save_project(project, tmp_path)

    def _card(run_id, outcomes):
        return Scorecard(run_id=run_id, subject="subject@test", judge="judge@test", results=[
            TestResult(test_id=tid, output="o",
                       criteria=[CriterionResult(criterion_id="c1", passed=ok,
                                                 score=1.0 if ok else 0.0)])
            for tid, ok in outcomes])

    save_scorecard(tmp_path, _card("run-0001", [(tid, True) for tid in baseline_ids]))
    save_scorecard(tmp_path, _card("run-0002", [("ex_1", True), ("ex_2", True),
                                                ("rt_1", True), ("rt_2", False)]))


def test_gate_refuses_a_baseline_that_never_ran_the_held_out_tests(tmp_path):
    """The baseline scored 100% on every test it graded — but none of them are
    held out, so there is no rate to compare and none may be invented."""
    _gate_project(tmp_path, ["ex_1", "ex_2"])

    result = runner.invoke(app, ["finetune", str(tmp_path), "--gate",
                                 "--baseline", "run-0001", "--candidate", "run-0002"])
    assert result.exit_code == 2, result.output
    assert "CANNOT JUDGE" in result.output
    assert "ACCEPT" not in result.output
    assert "baseline 0%" not in result.output
    assert _has_no_traceback(result)


def test_gate_still_rejects_a_candidate_that_loses_on_the_held_out_tests(tmp_path):
    """The refusal above must not swallow the ordinary comparison: when both runs
    graded the held-out tests, the verdict is still earned on the numbers."""
    _gate_project(tmp_path, ["ex_1", "ex_2", "rt_1", "rt_2"])

    result = runner.invoke(app, ["finetune", str(tmp_path), "--gate",
                                 "--baseline", "run-0001", "--candidate", "run-0002"])
    assert result.exit_code == 2, result.output
    assert "REJECT" in result.output
    assert "baseline 100% → candidate 50%" in result.output
    assert _has_no_traceback(result)


def _gate_project_pair(tmp_path, baseline_outcomes, candidate_outcomes):
    """Same shape as ``_gate_project``, but both runs' graded sets are explicit."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import (BehaviorSpec, CriterionResult, EvalCriterion, Example,
                                      Project, Scorecard, TestResult)
    from ai_calibrator.models import TestCase as CaseModel
    from ai_calibrator.store import save_project

    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(
        goal="g",
        eval_criteria=[EvalCriterion(id="c1", description="d")],
        examples=[Example(input="memorized 1", good_output="a", source="human")],
    )
    project.tests = [CaseModel(id="ex_1", input="memorized 1", expects=["c1"]),
                     CaseModel(id="rt_1", input="unseen 1", expects=["c1"]),
                     CaseModel(id="rt_2", input="unseen 2", expects=["c1"])]
    save_project(project, tmp_path)

    def _card(run_id, outcomes):
        return Scorecard(run_id=run_id, subject="subject@test", judge="judge@test", results=[
            TestResult(test_id=tid, output="o",
                       criteria=[CriterionResult(criterion_id="c1", passed=ok,
                                                 score=1.0 if ok else 0.0)])
            for tid, ok in outcomes])

    save_scorecard(tmp_path, _card("run-0001", baseline_outcomes))
    save_scorecard(tmp_path, _card("run-0002", candidate_outcomes))


def test_gate_refuses_two_runs_whose_held_out_tests_do_not_overlap(tmp_path):
    """Equal counts, different tests. The baseline graded rt_1 (failed) and the
    candidate graded rt_2 (passed) — one held-out test each, so a check on the
    COUNT sees a matched pair and says nothing, and the gate reads
    "baseline 0% → candidate 100%" as a win off two different exams."""
    _gate_project_pair(tmp_path,
                       baseline_outcomes=[("ex_1", True), ("rt_1", False)],
                       candidate_outcomes=[("ex_1", True), ("rt_2", True)])

    result = runner.invoke(app, ["finetune", str(tmp_path), "--gate",
                                 "--baseline", "run-0001", "--candidate", "run-0002"])
    assert result.exit_code == 2, result.output
    assert "CANNOT JUDGE" in result.output
    assert "ACCEPT" not in result.output
    assert _has_no_traceback(result)


def test_gate_scores_only_the_held_out_tests_both_runs_graded(tmp_path):
    """A partial overlap is still judgeable — on the shared tests alone. Baseline
    and candidate share rt_1 (baseline failed it, candidate passed it); the
    candidate additionally graded rt_2, which the baseline never saw and which
    therefore cannot count toward either rate."""
    _gate_project_pair(tmp_path,
                       baseline_outcomes=[("ex_1", True), ("rt_1", False)],
                       candidate_outcomes=[("ex_1", True), ("rt_1", True), ("rt_2", False)])

    result = runner.invoke(app, ["finetune", str(tmp_path), "--gate",
                                 "--baseline", "run-0001", "--candidate", "run-0002"])
    assert "gating on the 1 test(s) both runs graded and held out of training" in result.output
    assert "baseline 0% → candidate 100%" in result.output
    assert "graded different tests" in result.output   # and says so
    assert _has_no_traceback(result)


# --- an ingest that read nothing is not a successful ingest ---------------

def test_ingest_fails_when_no_file_in_a_populated_source_could_be_read(tmp_path, monkeypatch):
    """The corpus, facts, gaps and index are all replaced with emptiness. Green
    ✓ and exit 0 would report that as work done."""
    import ai_calibrator.engines as engines
    from ai_calibrator.models import Project
    from ai_calibrator.store import save_project

    class _Engine:
        name = "fake@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"facts": [], "gaps": []}

    monkeypatch.setattr(engines, "get_engine", lambda spec: _Engine())

    project = Project(name="p", goal="g")
    save_project(project, tmp_path)
    mats = tmp_path / "materials"
    (mats / "prices.xlsx").write_bytes(b"PK\x03\x04" + bytes(range(256)) * 10)
    (mats / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8)

    result = runner.invoke(app, ["ingest", str(tmp_path), "--no-index"])

    assert result.exit_code == 1, result.output
    assert "✓ Ingested" not in result.output
    assert "prices.xlsx" in result.output and "logo.png" in result.output
    assert _has_no_traceback(result)


def test_import_refuses_a_destination_that_cannot_name_a_project(tmp_path):
    """`merge` checks this before its interactive loop; `import` derives the name
    the same way and used to find out only after a billed engine call."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a support agent. Be brief.", encoding="utf-8")
    # A reserved device name: rejected on every platform, and unlike "..." it
    # survives resolve() on Windows instead of collapsing to the parent.
    target = tmp_path / "CON"

    result = runner.invoke(app, ["import", str(target), "--goal", "g",
                                 "--prompt", str(prompt_file)])

    assert result.exit_code == 1, result.output
    assert "Can't name a project after" in result.output
    assert _has_no_traceback(result)


def test_gate_refuses_same_ids_asking_different_questions(tmp_path):
    """The last comparability hole: matching ids.

    Every existing guard here is an id-set or count check — both runs graded
    something, they share tests, the judges match, neither is partial. All of
    them pass when `compile` re-mints t1..tN between the two evals, because the
    IDS are identical and only the questions changed. The gate then reads
    "baseline 0% → candidate 100%" off two different exams and says ACCEPT.
    """
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import (BehaviorSpec, CriterionResult, EvalCriterion, Project,
                                      Scorecard, TestResult)
    from ai_calibrator.models import TestCase as CaseModel
    from ai_calibrator.store import save_project

    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d")],
                                examples=[])          # nothing memorized: no overlap path
    project.tests = [CaseModel(id="t1", input="a question", expects=["c1"]),
                     CaseModel(id="t2", input="another question", expects=["c1"])]
    save_project(project, tmp_path)

    def _card(run_id, outcomes):
        return Scorecard(run_id=run_id, subject="subject@test", judge="judge@test", results=[
            TestResult(test_id=tid, output="o", input_hash=h,
                       criteria=[CriterionResult(criterion_id="c1", passed=ok,
                                                 score=1.0 if ok else 0.0)])
            for tid, ok, h in outcomes])

    # Same ids, same count, same judge — different questions behind them.
    save_scorecard(tmp_path, _card("run-0001", [("t1", False, "aaaa000000000000"),
                                                ("t2", False, "cccc222222222222")]))
    save_scorecard(tmp_path, _card("run-0002", [("t1", True, "bbbb111111111111"),
                                                ("t2", True, "dddd333333333333")]))

    result = runner.invoke(app, ["finetune", str(tmp_path), "--gate",
                                 "--baseline", "run-0001", "--candidate", "run-0002"])

    assert result.exit_code == 2, result.output
    assert "CANNOT JUDGE" in result.output
    assert "re-minted" in result.output
    assert "ACCEPT" not in result.output          # a 0% → 100% "win" off two exams
    assert _has_no_traceback(result)


def test_gate_still_judges_two_runs_that_asked_the_same_questions(tmp_path):
    """The refusal must not swallow the ordinary case: identical questions in
    both runs still gate, and back-compat scorecards (no recorded hash) still
    compare by id exactly as they did before the field existed."""
    _gate_project_pair(tmp_path,
                       baseline_outcomes=[("ex_1", True), ("rt_1", False)],
                       candidate_outcomes=[("ex_1", True), ("rt_1", True)])

    result = runner.invoke(app, ["finetune", str(tmp_path), "--gate",
                                 "--baseline", "run-0001", "--candidate", "run-0002"])

    assert "CANNOT JUDGE" not in result.output
    assert "re-minted" not in result.output
    assert _has_no_traceback(result)


def test_gate_refuses_non_comparable_runs_even_with_no_training_overlap(tmp_path):
    """Comparability is not an overlap question. With nothing memorized, the gate
    used to take an unguarded path and could accept off two runs that graded
    entirely different tests — the exact failure the overlap path refuses."""
    _gate_project_pair(tmp_path,
                       baseline_outcomes=[("rt_1", False)],
                       candidate_outcomes=[("rt_2", True)])
    # Remove the examples so nothing is a training prompt: no overlap at all.
    from ai_calibrator.store import load_project, save_project
    project = load_project(tmp_path)
    project.spec.examples = []
    save_project(project, tmp_path)

    result = runner.invoke(app, ["finetune", str(tmp_path), "--gate",
                                 "--baseline", "run-0001", "--candidate", "run-0002"])

    assert result.exit_code == 2, result.output
    assert "CANNOT JUDGE" in result.output and "ACCEPT" not in result.output
    assert _has_no_traceback(result)


# --- an answer the tool wrote is not an answer a person gave ---------------

def _interview_project(tmp_path):
    from ai_calibrator.models import Gap, InterviewItem, Project
    from ai_calibrator.store import save_project

    p = Project(name="p", goal="g")
    p.gaps = [Gap(dimension="shipping_cost"), Gap(dimension="tone")]
    p.interview = [
        InterviewItem(id="q1", dimension="shipping_cost", question="What is it?",
                      draft_answer="It is $4.95."),
        InterviewItem(id="q2", dimension="tone", question="What voice?",
                      draft_answer="Warm and plain."),
    ]
    save_project(p, tmp_path)
    return p


def test_accepted_drafts_are_recorded_as_the_tools_own_answers(tmp_path):
    """`--accept-drafts` takes the tool's guess unreviewed. It cannot know what the
    materials leave unstated, so a draft can assert policy nobody wrote — and
    `compile` turns answers into standards, criteria and graded tests. The spec
    must stay able to tell those from a human's decision."""
    from ai_calibrator.store import load_project

    _interview_project(tmp_path)
    result = runner.invoke(app, ["interview", str(tmp_path), "--accept-drafts"])
    assert result.exit_code == 0, result.output

    items = load_project(tmp_path).interview
    assert [it.answer for it in items] == ["It is $4.95.", "Warm and plain."]
    assert all(it.answer_source == "engine" for it in items)
    assert all(it.unratified for it in items)
    assert "accepted without review" in result.output


def test_a_typed_answer_and_an_accepted_draft_are_both_human_decisions(tmp_path):
    """Reading a draft and pressing Enter IS ratification; typing your own is
    authorship. Neither is the tool answering itself."""
    from ai_calibrator.store import load_project

    _interview_project(tmp_path)
    # q1: type a correction. q2: Enter to accept the draft.
    result = runner.invoke(app, ["interview", str(tmp_path)], input="Not stated for orders under $35.\n\n")
    assert result.exit_code == 0, result.output

    by_id = {it.id: it for it in load_project(tmp_path).interview}
    assert by_id["q1"].answer == "Not stated for orders under $35."
    assert by_id["q1"].answer_source == "human"
    assert by_id["q2"].answer == "Warm and plain."
    assert by_id["q2"].answer_source == "human_ratified"
    assert not any(it.unratified for it in by_id.values())
    assert "accepted without review" not in result.output


def test_an_answer_from_before_the_field_existed_is_not_called_engine_written(tmp_path):
    """Existing projects must not be retroactively distrusted: no recorded source
    means unknown, which is never reported as the tool's own guess."""
    from ai_calibrator.models import InterviewItem

    item = InterviewItem(id="q1", dimension="tone", question="?", answer="Warm.")
    assert item.answer_source is None and not item.unratified


@pytest.mark.parametrize("seconds,expected", [
    (0.4, "1s"), (12.0, "12s"), (59.4, "59s"), (75.0, "a minute"), (400.0, "7 min"),
])
def test_a_progress_estimate_is_never_more_precise_than_it_deserves(seconds, expected):
    """The estimate is extrapolated from a handful of calls, so it rounds — a
    to-the-second countdown would imply a confidence it does not have."""
    from ai_calibrator.cli import _duration

    assert _duration(seconds) == expected
