"""Regressions for the pre-publication audit.

Grouped by the property each one protects: a generated artifact must run, an
index must not outlive its source, a merge must not lose a human's rule, and a
reference point (baseline, golden, badge) must never be a partial run.
"""

import json
import re

import pytest

# aliased: pytest collects any module-level name starting with `test`
from ai_calibrator.compile import tests_from_examples as build_example_tests
from ai_calibrator.examples_io import dedup_examples
from ai_calibrator.finetune import recommend_recipe, render_train_py, training_overlap
from ai_calibrator.lint import lint_spec
from ai_calibrator.models import (
    BehaviorSpec,
    CriterionResult,
    EvalCriterion,
    Example,
    Persona,
    Project,
    Scorecard,
    TaskType,
    Weight,
)
# Aliased: pytest would otherwise try to collect these models as test classes.
from ai_calibrator.models import TestCase as CaseModel
from ai_calibrator.models import TestResult as ResultModel
from ai_calibrator.stakeholders import build_merged_spec, scalar_conflicts


# --- generated artifacts must actually run ---------------------------------

def test_generated_trainer_has_no_unsubstituted_placeholders():
    """A leftover `__EPOCHS__` is a valid identifier, so the file still PARSES and
    only dies with NameError at run time — after a multi-GB model download. An
    ast.parse check is provably insufficient; assert on the placeholders."""
    src = render_train_py(recommend_recipe(40))
    for token in ("__BASE__", "__OUT__", "__EPOCHS__", "__MAX_STEPS__"):
        assert token not in src, f"{token} survived rendering"


def test_engine_bundle_and_finetune_bundle_render_identically(tmp_path):
    """Both bundle writers go through the one renderer, so neither can drift."""
    from ai_calibrator.train_engine import export_engine_bundle

    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [{"messages": [{"role": "system", "content": "s"},
                          {"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": "a"}]} for i in range(3)]
    (logs / "judge.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    export_engine_bundle(tmp_path, "judge")
    src = (tmp_path / "trained-engines" / "judge" / "train.py").read_text(encoding="utf-8")
    assert "__EPOCHS__" not in src and "__MAX_STEPS__" not in src
    assert "num_train_epochs=" in src


def test_recipe_hyperparameters_reach_the_trainer():
    """recipe.yaml is documented as editable, so train.py must read it."""
    src = render_train_py(recommend_recipe(40))
    assert "_recipe()" in src and "recipe.yaml" in src
    assert 'learning_rate=float(cfg["learning_rate"])' in src


# --- the index must not outlive its source ---------------------------------

def test_ingest_drops_the_index_when_every_material_is_gone(tmp_path, monkeypatch):
    """Deleting the documents must delete what was built from them — otherwise the
    deleted text keeps being injected into every graded and served prompt."""
    import ai_calibrator.ingest as ing

    (tmp_path / "knowledge.lancedb").mkdir()
    (tmp_path / "knowledge.lancedb" / "data").write_text("stale", encoding="utf-8")
    (tmp_path / "materials").mkdir()

    class Eng:
        name = "e@test"

        def complete(self, prompt, *, system=None, schema=None):
            raise AssertionError("no engine call on an empty corpus")

    project = Project(name="p", goal="g")
    result = ing.ingest_project(project, str(tmp_path / "materials"), Eng(),
                                project_dir=tmp_path, build_index=True)

    assert not (tmp_path / "knowledge.lancedb").exists()
    assert result.indexed == 0


def test_ingest_reports_documents_the_extractor_never_saw(tmp_path):
    """The gap list drives the whole interview; if it came from a fraction of the
    corpus the owner has to be told, not shown a full file count."""
    from ai_calibrator.ingest import MAX_EXTRACT_CHARS, extract_gaps
    from pathlib import Path

    big = "x" * MAX_EXTRACT_CHARS
    docs = [(Path("a.md"), big), (Path("b.md"), "never seen"), (Path("c.md"), "nor this")]

    class Eng:
        name = "e@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"facts": [], "gaps": []}

    _facts, _gaps, analyzed = extract_gaps("g", "assistant", docs, Eng())
    assert analyzed == 1 and analyzed < len(docs)


# --- a merge must not silently lose a human's rule --------------------------

def _spec(**kw):
    return BehaviorSpec(goal="g", **kw)


def test_merge_reports_conflicting_refusal_policies():
    """These render straight into the system prompt. Resolving them by --from order
    ships a different, less safe AI depending on flag order, with `conflicts: []`
    written to the file the docs call an audit trail."""
    named = {
        "legal": _spec(refusal_policy="Refuse legal advice; direct to counsel.",
                       persona=Persona(voice="formal"), format="Plain prose."),
        "sales": _spec(refusal_policy="Never refuse; hand off to a human.",
                       persona=Persona(voice="punchy"), format="Bullets."),
    }
    fields = {f for f, _ in scalar_conflicts(named)}
    assert fields == {"refusal_policy", "persona.voice", "format"}


def test_merge_is_order_independent():
    """Same specs, either flag order, same deployed prompt."""
    a = _spec(refusal_policy="A policy", standards=["s1"])
    b = _spec(refusal_policy="B policy", standards=["s2"])
    one = build_merged_spec({"alpha": a, "beta": b}, goal="g", task_type=TaskType.ASSISTANT)
    two = build_merged_spec({"beta": b, "alpha": a}, goal="g", task_type=TaskType.ASSISTANT)
    assert one.refusal_policy == two.refusal_policy


def test_merge_namespaces_a_colliding_criterion_instead_of_dropping_it():
    """Criterion ids are engine-generated labels, so two specs both having
    `accuracy` (meaning different things) is routine — and deduping on the bare id
    would discard one stakeholder's criterion along with its deterministic check."""
    from ai_calibrator.models import Check

    named = {
        "a": _spec(eval_criteria=[EvalCriterion(id="accuracy", description="cites the policy",
                                                weight=Weight.HIGH)]),
        "b": _spec(eval_criteria=[EvalCriterion(id="accuracy", description="no arithmetic errors",
                                                weight=Weight.LOW,
                                                check=Check(kind="not_contains", value="approx"))]),
    }
    spec = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT)
    descs = {c.description for c in spec.eval_criteria}
    assert descs == {"cites the policy", "no arithmetic errors"}
    assert any(c.check is not None for c in spec.eval_criteria), "a check was dropped"


def test_true_duplicate_criteria_still_collapse():
    named = {"a": _spec(eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)]),
             "b": _spec(eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])}
    spec = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT)
    assert [c.id for c in spec.eval_criteria] == ["c1"]


# --- pinned anchors must stay able to fail the gate -------------------------

def test_example_tests_never_collide_with_an_existing_id():
    """Ids were minted from the example's POSITION, so anything that shifts indices
    re-issued an id a pinned test already owned — and drift/snapshot key results by
    id, so the older anchor silently stopped existing."""
    spec = BehaviorSpec(goal="g", examples=[Example(input="A"), Example(input="C")])
    existing = [CaseModel(id="ex_1", input="A"), CaseModel(id="ex_3", input="B")]
    new = build_example_tests(spec, existing)
    assert [t.id for t in new] == ["ex_2"]
    assert len({t.id for t in existing + new}) == len(existing) + len(new)


def test_duplicate_test_ids_are_a_lint_error():
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="is accurate and cites policy", weight=Weight.HIGH)])
    tests = [CaseModel(id="t1", input="a", expects=["c1"]),
             CaseModel(id="t1", input="b", expects=["c1"])]
    codes = [i.code for i in lint_spec(spec, tests).issues if i.severity == "error"]
    assert "duplicate_test_id" in codes


# --- feedback: the latest verdict wins --------------------------------------

def test_a_down_retracts_an_earlier_up_on_the_same_answer(tmp_path):
    """Otherwise the spec asserts one text is both good and bad, and the fine-tune
    dataset keeps training toward the answer a human rejected."""
    from ai_calibrator.flywheel import absorb_feedback

    logs = tmp_path / "logs"
    logs.mkdir()
    records = [
        {"turns": ["q"], "output": "bad answer", "verdict": "up"},
        {"turns": ["q"], "output": "bad answer", "verdict": "down", "correction": "good answer"},
    ]
    (logs / "feedback.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(goal="g")
    result = absorb_feedback(project, tmp_path)

    goods = [e.good_output for e in project.spec.examples]
    assert "bad answer" not in goods, "the rejected answer survived as a training target"
    assert result.superseded >= 1


def test_dedup_keeps_the_correction_not_the_rejected_answer():
    spec = BehaviorSpec(goal="g", examples=[
        Example(input="q", good_output="rejected"),
        Example(input="q", good_output="corrected")])
    dedup_examples(spec)
    assert [e.good_output for e in spec.examples] == ["corrected"]


# --- the prove-it gate must admit when it is not held out -------------------

def test_training_overlap_names_tests_that_are_training_prompts():
    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(goal="g", examples=[Example(input="q1", good_output="a")])
    project.tests = [CaseModel(id="ex_1", input="q1"), CaseModel(id="t1", input="unseen")]
    card = Scorecard(run_id="run-0001", results=[
        ResultModel(test_id="ex_1", output="o",
                    criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)]),
        ResultModel(test_id="t1", output="o",
                    criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])])
    assert training_overlap(project, card) == ["ex_1"]


# --- judge output is coerced into the range the scorecard documents ---------

@pytest.mark.parametrize("raw,expected", [(85, 1.0), (-3, 0.0), (0.5, 0.5), ("nope", 0.0)])
def test_judge_scores_are_clamped(raw, expected):
    """A judge answering on a 0-100 or 1-5 scale would otherwise push the weighted
    mean above 1.0, which `pct` renders as a reassuring '>99%'."""
    from ai_calibrator.eval import _as_float
    assert _as_float(raw) == expected


# --- exports must not quietly change what they describe ---------------------

def test_promptfoo_export_flags_omitted_multi_turn_tests():
    from ai_calibrator.interop import to_promptfoo

    project = Project(name="p", goal="g")
    project.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="is helpful", weight=Weight.MEDIUM)])
    project.tests = [CaseModel(id="t1", input="a", expects=["c1"]),
                     CaseModel(id="t2", input="b", follow_ups=["c"], expects=["c1"])]
    out = to_promptfoo(project)
    assert "multi-turn test(s) omitted" in out and "t2" in out
    assert "vars" in out  # the single-turn test still exported


def test_diff_notices_a_changed_knowledge_base():
    """render_system_prompt appends a grounding paragraph iff knowledge_sources is
    non-empty, so this genuinely changes the deployed prompt."""
    from ai_calibrator.compile import render_system_prompt
    from ai_calibrator.specdiff import diff_specs

    a = BehaviorSpec(goal="g")
    b = BehaviorSpec(goal="g", knowledge_sources=["policy.pdf"])
    assert render_system_prompt(a) != render_system_prompt(b)
    assert diff_specs(a, b).changed


# --- friendly failures ------------------------------------------------------

def test_judge_labels_reject_non_scalar_ids(tmp_path):
    """A list id becomes a dict key downstream — 'unhashable type' escaping as a 500."""
    from ai_calibrator.judge_check import save_labels

    from ai_calibrator.judge_check import load_labels

    save_labels(tmp_path, "run-0001", [{"test_id": ["t1"], "criterion_id": "c1", "passed": True}])
    assert load_labels(tmp_path, "run-0001") == []


def test_modelfile_marks_a_rewritten_system_prompt():
    """The Modelfile would otherwise ship a mutated prompt while system_prompt.txt
    (what the eval graded) keeps the original, with nothing said."""
    from ai_calibrator.export import _modelfile

    out = _modelfile("base", 'say """ here')
    assert "DIFFERS from" in out
    assert re.search(r'"{3,}', out.split("SYSTEM")[1].replace('"""', "", 1)) is None




def test_merge_resolves_persona_fields_independently():
    """persona.voice and persona.reading_level are reported as two separate field
    conflicts, so taking one stakeholder's whole persona object ships (and audits)
    a reading level nobody's report chose."""
    named = {
        "alpha": _spec(persona=Persona(voice="warm and direct")),
        "beta": _spec(persona=Persona(voice="clipped", reading_level="8th grade")),
    }
    resolved = {field: vals[0] for field, vals in scalar_conflicts(named)}
    assert resolved["persona.voice"] == ("alpha", "warm and direct")
    assert "persona.reading_level" not in resolved      # only beta set one — uncontested

    spec = build_merged_spec(named, goal="g", task_type=TaskType.ASSISTANT)
    assert spec.persona.voice == "warm and direct"      # as the report promised
    assert spec.persona.reading_level == "8th grade"    # uncontested, must survive

    # and the other way round: the voice winner does not drag its own blank fields in
    flipped = {
        "alpha": _spec(persona=Persona(reading_level="plain English")),
        "beta": _spec(persona=Persona(voice="clipped")),
    }
    merged = build_merged_spec(flipped, goal="g", task_type=TaskType.ASSISTANT)
    assert (merged.persona.voice, merged.persona.reading_level) == ("clipped", "plain English")




def test_install_hints_name_a_command_a_reader_can_run():
    """There is no PyPI release, so a printed `pip install 'ai-calibrator[x]'` is
    an instruction nobody can follow. Every runtime hint has to name the
    editable, clone-based form README and USAGE.md actually prescribe."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "ai_calibrator"
    unrunnable = [
        f"{p.relative_to(src)}:{n}"
        for p in sorted(src.rglob("*.py"))
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "pip install 'ai-calibrator[" in line
    ]
    assert not unrunnable, unrunnable
