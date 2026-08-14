"""Spec-lint — proactive quality checks on a behavior spec."""

from ai_calibrator.lint import lint_contradictions, lint_schema_version, lint_spec
from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from ai_calibrator.models import TestCase as Case


def test_clean_spec_has_no_errors():
    spec = BehaviorSpec(
        goal="g", standards=["Always cite the 30-day return window."],
        do_not=["Never promise refunds we don't offer."], refusal_policy="Decline politely and redirect.",
        eval_criteria=[EvalCriterion(id="cite", description="cites the policy window", weight=Weight.HIGH)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["cite"])])
    assert r.ok and not r.errors


def test_no_criteria_is_error():
    r = lint_spec(BehaviorSpec(goal="g", standards=["Be concise and clear always."]), [])
    assert not r.ok and any(i.code == "no_criteria" for i in r.errors)


def test_untested_criterion_warns():
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="something objectively testable", weight=Weight.HIGH)])
    r = lint_spec(spec, [])  # nothing targets c1
    assert any(i.code == "untested_criterion" and i.where == "c1" for i in r.issues)


def test_vague_and_short_standards_flagged():
    spec = BehaviorSpec(goal="g", standards=["ok", "Be helpful and appropriate as needed"],
                        eval_criteria=[EvalCriterion(id="c1", description="x is clearly satisfied", weight=Weight.HIGH)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["c1"])])
    assert "vague_standard" in {i.code for i in r.issues}  # "ok" too short + weasel words


def test_duplicate_criterion_is_error():
    spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="dup", description="first definition here", weight=Weight.HIGH),
        EvalCriterion(id="dup", description="second definition here", weight=Weight.LOW)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["dup"])])
    assert any(i.code == "duplicate_criterion" for i in r.errors)


def test_never_rules_without_refusal_policy_is_info():
    spec = BehaviorSpec(goal="g", do_not=["Never give medical advice."],
                        eval_criteria=[EvalCriterion(id="c1", description="gives no medical advice", weight=Weight.HIGH)])
    r = lint_spec(spec, [Case(id="t1", input="q", expects=["c1"])])
    assert any(i.code == "no_refusal_policy" for i in r.issues)


def test_lint_contradictions_reuses_conflict_detector():
    class ConflictEngine:
        name = "ce@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"conflicts": [{"a": 1, "b": 2, "explanation": "cannot both hold", "severity": "high"}]}

    spec = BehaviorSpec(goal="g", standards=["Always be brief.", "Always explain in great detail."])
    issues = lint_contradictions(spec, ConflictEngine())
    assert len(issues) == 1 and issues[0].code == "self_contradiction" and issues[0].severity == "error"


def test_lint_flags_unknown_fields():
    """Preserved-but-unrecognized fields (typo / newer version) surface as warnings."""
    from ai_calibrator.lint import lint_unknown_fields
    from ai_calibrator.models import Project

    project = Project.model_validate({
        "name": "p", "goal": "g",
        "standrds_typo": ["oops"],
        "spec": {"goal": "g", "eval_criteria": [{"id": "c1", "description": "d", "wieght": "high"}]},
    })
    issues = lint_unknown_fields(project)
    wheres = {i.where for i in issues}
    assert {"project.standrds_typo", "project.spec.eval_criteria[0].wieght"} <= wheres
    assert all(i.code == "unknown_field" and i.severity == "warn" for i in issues)
    # and a clean project yields none
    assert lint_unknown_fields(Project(name="p", goal="g")) == []


# --- a model grading its own answers ---------------------------------------

def _roles_project(subject, judge, *, judged_criteria=1, checked_criteria=0):
    from ai_calibrator.models import Check, EngineBinding

    crits = [EvalCriterion(id=f"j{i}", description="is judged by the model", weight=Weight.HIGH)
             for i in range(judged_criteria)]
    crits += [EvalCriterion(id=f"c{i}", description="is checked exactly", weight=Weight.HIGH,
                            check=Check(kind="contains", value="please"))
              for i in range(checked_criteria)]
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=crits)
    p.engines = EngineBinding(subject=subject, judge=judge)
    return p


def test_lint_warns_when_the_judge_is_the_subject():
    """`calibrate engines <project> --all <model>` points every role at one
    model in a single command, and the README's local quickstart says to do
    exactly that. The pass rate is then the model's opinion of itself."""
    from ai_calibrator.lint import lint_engine_roles

    issues = lint_engine_roles(_roles_project("qwen2.5:7b@ollama", "qwen2.5:7b@ollama"))

    assert len(issues) == 1
    assert issues[0].code == "judge_is_subject"
    assert issues[0].severity == "warn"        # a real way to work — must not block the gate
    assert "qwen2.5:7b@ollama" in issues[0].message


def test_lint_warns_however_the_same_engine_is_spelled():
    """`parse_engine_spec` defaults the provider to ollama and strips both
    halves, so several strings build the identical engine — and the CLI's own
    error text tells the user to type the short one ("or just `model` for local
    Ollama"). Comparing the raw strings meant rebinding one role with a different
    spelling retired the warning while the model still graded its own answers."""
    from ai_calibrator.lint import lint_engine_roles

    for judge in ("qwen2.5:7b", " qwen2.5:7b ", "qwen2.5:7b @ ollama", "qwen2.5:7b@OLLAMA"):
        issues = lint_engine_roles(_roles_project("qwen2.5:7b@ollama", judge))
        assert [i.code for i in issues] == ["judge_is_subject"], judge


def test_lint_is_quiet_when_the_judge_is_a_different_model():
    from ai_calibrator.lint import lint_engine_roles

    assert lint_engine_roles(_roles_project("qwen2.5:7b@ollama", "claude-haiku-4-5@anthropic")) == []


def test_lint_ignores_self_grading_when_no_criterion_reaches_the_judge():
    """Every criterion graded by a deterministic `check` means no judge call is
    made at all, so the two bindings being identical decides nothing."""
    from ai_calibrator.lint import lint_engine_roles

    p = _roles_project("m@ollama", "m@ollama", judged_criteria=0, checked_criteria=2)
    assert lint_engine_roles(p) == []


def test_ci_lint_stage_surfaces_the_self_grading_warning_without_failing(tmp_path):
    """It has to reach the gate report — a warning nobody sees is not a warning
    — but it must not turn a legitimate single-model setup into a failure."""
    from ai_calibrator.ci import run_ci
    from ai_calibrator.models import TestCase as CaseModel
    from ai_calibrator.store import save_project

    class _E:
        name = "m@ollama"

        def complete(self, prompt, *, system=None, schema=None):
            import re
            ids = re.findall(r"^- (\S+):", prompt, re.M)
            if ids:
                return {"results": [{"criterion_id": i, "passed": True, "score": 1.0,
                                     "rationale": "r"} for i in ids]}
            return "an answer"

    p = _roles_project("m@ollama", "m@ollama")
    p.tests = [CaseModel(id="t1", input="q", expects=["j0"])]
    save_project(p, tmp_path)

    def _warnings(project):
        result = run_ci(project, _E(), _E(), project_dir=tmp_path)
        stage = next(s for s in result.stages if s.name == "lint")
        assert stage.status == "pass"           # a warning, never an error
        return int(stage.detail.split("error(s), ")[1].split(" ")[0])

    same = _warnings(p)

    # The same project with a distinct judge carries exactly one warning fewer.
    # That difference is what proves THIS rule reached the gate, rather than
    # some other lint warning happening to account for the count.
    p.engines.judge = "claude-haiku-4-5@anthropic"
    save_project(p, tmp_path)
    assert _warnings(p) == same - 1


# --- the on-disk format's version marker -----------------------------------

def test_project_yaml_declares_its_schema_version_first(tmp_path):
    """The day this ships, project.yaml is a compatibility contract with
    strangers. A version marker cannot be added retroactively to files already
    written, and one nobody can find is not a marker — so it leads the file."""
    import yaml

    from ai_calibrator.models import SCHEMA_VERSION
    from ai_calibrator.store import save_project

    save_project(Project(name="p", goal="g"), tmp_path)
    text = (tmp_path / "project.yaml").read_text(encoding="utf-8")

    assert text.splitlines()[0] == f"schema_version: {SCHEMA_VERSION}"
    assert yaml.safe_load(text)["schema_version"] == SCHEMA_VERSION


def test_a_file_written_before_the_field_existed_loads_as_version_one(tmp_path):
    import yaml

    from ai_calibrator.store import load_project, save_project

    save_project(Project(name="p", goal="g"), tmp_path)
    raw = yaml.safe_load((tmp_path / "project.yaml").read_text(encoding="utf-8"))
    raw.pop("schema_version")
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    assert load_project(tmp_path).schema_version == 1
    assert lint_schema_version(load_project(tmp_path)) == []   # not "from the future"


def test_stamping_the_version_does_not_stale_an_existing_certification(tmp_path):
    """`config_hash` decides whether a gate still certifies the current config.
    If a new persisted field moved it, everyone's existing certification would
    read as stale the moment they upgraded, for no behavior change at all."""
    import yaml

    from ai_calibrator.ci import config_hash
    from ai_calibrator.models import TestCase as CaseModel
    from ai_calibrator.store import load_project, save_project

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="cites the policy", weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(p, tmp_path)
    current = config_hash(load_project(tmp_path), tmp_path)

    raw = yaml.safe_load((tmp_path / "project.yaml").read_text(encoding="utf-8"))
    raw.pop("schema_version")                       # a project.yaml from before the upgrade
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    assert config_hash(load_project(tmp_path), tmp_path) == current


def test_lint_warns_about_a_project_from_a_newer_calibrator():
    from ai_calibrator.models import SCHEMA_VERSION

    p = Project(name="p", goal="g")
    p.schema_version = SCHEMA_VERSION + 1

    issues = lint_schema_version(p)

    assert len(issues) == 1
    assert issues[0].code == "future_schema_version" and issues[0].severity == "warn"


def test_scorecard_also_carries_the_version(tmp_path):
    import json

    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import SCHEMA_VERSION, Scorecard

    save_scorecard(tmp_path, Scorecard(run_id="run-0001"))
    written = json.loads((tmp_path / "evals" / "run-0001" / "scorecard.json").read_text(encoding="utf-8"))
    assert written["schema_version"] == SCHEMA_VERSION
