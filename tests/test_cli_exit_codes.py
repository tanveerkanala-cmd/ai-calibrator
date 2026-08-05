"""The exit-code contracts every CI pipeline depends on.

`ci`, `run`, `lint`, `drift` and `snapshot --check` all document a non-zero exit
on failure — that promise is the entire integration surface of this tool, and a
gate that exits 0 when it failed is worse than no gate. None of these paths was
reached by any test: each could be deleted or inverted with the suite still
green. These pin the contract, not the wording.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ai_calibrator.cli import app
from ai_calibrator.models import (
    BehaviorSpec,
    Check,
    CriterionResult,
    EvalCriterion,
    Project,
    Scorecard,
    Weight,
)
from ai_calibrator.models import TestCase as CaseModel
from ai_calibrator.models import TestResult as ResultRow
from ai_calibrator.store import save_project

runner = CliRunner()


def _no_crash(result) -> bool:
    exc = result.exception
    return exc is None or isinstance(exc, SystemExit)


def _project(tmp_path, *, check_value="please"):
    """A compiled project whose single criterion is graded deterministically, so
    every stage below runs without an engine."""
    p = Project(name="p", goal="be polite")
    p.spec = BehaviorSpec(goal="be polite", eval_criteria=[
        EvalCriterion(id="c1", description="stays polite", weight=Weight.HIGH,
                      check=Check(kind="contains", value=check_value))])
    p.tests = [CaseModel(id="t1", input="a question", expects=["c1"])]
    save_project(p, tmp_path)
    return p


def _stub_uvicorn(monkeypatch):
    """Stop `run` before it actually binds a port — when uvicorn is installed at
    all. It ships in the `api` extra, so a plain `.[dev]` venv has none and the
    command exits 1 at its own ImportError guard instead. The gate decision
    below it is what these tests pin, and that happens either way."""
    try:
        import uvicorn
    except ImportError:
        return None
    calls: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app_obj, **kw: calls.update(kw) or None)
    return calls


class _Engine:
    """Answers a fixed string, so the deterministic check decides the verdict."""

    name = "fake@test"

    def __init__(self, reply="please"):
        self.reply = reply

    def complete(self, prompt, *, system=None, schema=None):
        return self.reply


def _stub_engines(monkeypatch, reply="please"):
    import ai_calibrator.engines as engines
    monkeypatch.setattr(engines, "get_engine", lambda spec: _Engine(reply))


def _gate(tmp_path, project, *, ok: bool):
    """Write a last-gate record that certifies the CURRENT config, so
    `certification_status` returns pass/fail rather than stale."""
    from ai_calibrator.ci import config_hash, save_gate

    class _Result:
        run_id = "run-0001"
        pass_rate = 1.0 if ok else 0.0
        stages = [type("S", (), {"name": "eval", "status": "pass" if ok else "fail",
                                 "detail": "d"})()]

    from ai_calibrator.ci import CiResult, CiStage
    result = CiResult(run_id="run-0001", stages=[
        CiStage("eval", "pass" if ok else "fail", "detail")])
    save_gate(project, result, tmp_path)
    assert (tmp_path / "evals" / "last-gate.json").exists()
    # Sanity: the record must match the current config or the status is "stale".
    saved = json.loads((tmp_path / "evals" / "last-gate.json").read_text())
    assert saved["config_hash"] == config_hash(project, tmp_path)
    return saved


# --- `calibrate run`: the boot gate ----------------------------------------

def test_run_refuses_to_serve_a_failed_gate(tmp_path):
    """The product's marquee safety claim: an AI that cannot prove it follows
    its rules does not serve. Deleting this branch left all tests green."""
    p = _project(tmp_path)
    _gate(tmp_path, p, ok=False)

    r = runner.invoke(app, ["run", str(tmp_path)])

    assert r.exit_code == 2, r.output
    assert "REFUSING TO SERVE" in r.output
    assert _no_crash(r)


def test_run_force_serves_a_failed_gate_but_says_so(tmp_path, monkeypatch):
    """--force is the documented override. It must not print the refusal — and
    must not print a clean certification either."""
    booted = _stub_uvicorn(monkeypatch)
    p = _project(tmp_path)
    _gate(tmp_path, p, ok=False)

    r = runner.invoke(app, ["run", str(tmp_path), "--force"])

    assert r.exit_code != 2, r.output          # the refusal did NOT fire
    assert "REFUSING TO SERVE" not in r.output
    assert "UNCERTIFIED" in r.output           # the override is loud, not silent
    assert _no_crash(r)
    if booted is not None:
        assert r.exit_code == 0 and booted, "uvicorn.run was never reached"


def test_run_serves_a_passing_gate(tmp_path, monkeypatch):
    booted = _stub_uvicorn(monkeypatch)
    p = _project(tmp_path)
    _gate(tmp_path, p, ok=True)

    r = runner.invoke(app, ["run", str(tmp_path)])

    assert r.exit_code != 2, r.output
    assert "Certified" in r.output
    assert _no_crash(r)
    if booted is not None:
        assert r.exit_code == 0 and booted


def test_run_without_a_spec_exits_one_not_two(tmp_path):
    """1 means "couldn't gate", 2 means "the AI failed it" — a pipeline reads
    the difference."""
    save_project(Project(name="p", goal="g"), tmp_path)
    r = runner.invoke(app, ["run", str(tmp_path)])
    assert r.exit_code == 1 and "Nothing to serve" in r.output and _no_crash(r)


# --- `calibrate ci`: 0 / 1 / 2 ---------------------------------------------

def test_ci_exits_zero_when_the_gate_passes(tmp_path, monkeypatch):
    _project(tmp_path)                     # the check wants "please" in the output
    _stub_engines(monkeypatch, reply="please, here is the policy")

    r = runner.invoke(app, ["ci", str(tmp_path)])

    assert r.exit_code == 0, r.output
    assert _no_crash(r)


def test_ci_exits_two_when_the_ai_fails_the_gate(tmp_path, monkeypatch):
    """2 is the contract for "the AI failed", and it is the code a deploy
    pipeline branches on. Exiting 0 here would ship an AI that failed its own
    gate; exiting 1 would read as a tooling problem and get retried."""
    _project(tmp_path)
    _stub_engines(monkeypatch, reply="no.")      # misses the required "please"

    r = runner.invoke(app, ["ci", str(tmp_path)])

    assert r.exit_code == 2, r.output
    assert _no_crash(r)


def test_ci_json_reports_the_failure_it_exited_two_for(tmp_path, monkeypatch):
    _project(tmp_path)
    _stub_engines(monkeypatch, reply="no.")

    r = runner.invoke(app, ["ci", str(tmp_path), "--json"])

    assert r.exit_code == 2
    payload = json.loads(r.output[r.output.index("{"):])   # pretty-printed, not one line
    assert payload["ok"] is False
    assert any(s["status"] == "fail" for s in payload["stages"])


def test_ci_exits_one_when_it_cannot_gate(tmp_path):
    """No spec: the gate could not run at all. Not the same as failing it."""
    save_project(Project(name="p", goal="g"), tmp_path)
    r = runner.invoke(app, ["ci", str(tmp_path)])
    assert r.exit_code == 1, r.output
    assert "Nothing to gate" in r.output
    assert _no_crash(r)


def test_ci_json_emits_structured_reason_on_every_error_exit(tmp_path):
    """`ci --json` piped into a parser must never get a coloured sentence."""
    save_project(Project(name="p", goal="g"), tmp_path)
    r = runner.invoke(app, ["ci", str(tmp_path), "--json"])
    assert r.exit_code == 1
    payload = json.loads(r.output.strip().splitlines()[-1])
    assert payload["ok"] is False and payload["gate"] == "error" and payload["reason"]


def test_ci_validation_errors_exit_one(tmp_path):
    _project(tmp_path)
    for args, needle in (
        (["--threshold", "1.5"], "--threshold"),
        (["--tolerance", "-1"], "--tolerance"),
        (["--judge-passes", "0"], "--judge-passes"),
    ):
        r = runner.invoke(app, ["ci", str(tmp_path), *args])
        assert r.exit_code == 1, (args, r.output)
        assert needle in r.output
        assert _no_crash(r)


# --- `calibrate lint` / `drift` / `snapshot --check` ------------------------

def test_lint_exits_nonzero_on_an_error_level_issue(tmp_path):
    """Duplicate test ids are a lint ERROR precisely because every comparison
    surface collapses results into a dict keyed by id."""
    p = _project(tmp_path)
    p.tests = [CaseModel(id="t1", input="a", expects=["c1"]),
               CaseModel(id="t1", input="b", expects=["c1"])]   # duplicate id
    save_project(p, tmp_path)

    r = runner.invoke(app, ["lint", str(tmp_path)])

    assert r.exit_code == 1, r.output
    assert "t1" in r.output
    assert _no_crash(r)


def test_lint_exits_zero_on_a_clean_spec(tmp_path):
    _project(tmp_path)
    r = runner.invoke(app, ["lint", str(tmp_path)])
    assert r.exit_code == 0, r.output


def test_snapshot_check_exits_two_on_drift(tmp_path):
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.snapshot import save_golden

    _project(tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[
        ResultRow(test_id="t1", output="NEW answer",
                  criteria=[CriterionResult(criterion_id="c1", passed=True)])]))
    save_golden(tmp_path, {"t1": "OLD answer"})

    r = runner.invoke(app, ["snapshot", str(tmp_path), "--check"])

    assert r.exit_code == 2, r.output
    assert "output changed" in r.output
    assert _no_crash(r)


def test_snapshot_check_exits_zero_when_outputs_match(tmp_path):
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.snapshot import save_golden

    _project(tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[
        ResultRow(test_id="t1", output="same",
                  criteria=[CriterionResult(criterion_id="c1", passed=True)])]))
    save_golden(tmp_path, {"t1": "same"})

    r = runner.invoke(app, ["snapshot", str(tmp_path), "--check"])
    assert r.exit_code == 0, r.output
    assert "match the golden" in r.output


def test_snapshot_refuses_to_pin_a_partial_run(tmp_path):
    """The CLI twin of a guard the API side already had tested. Pinning from an
    interrupted run replaces a complete golden with a strict subset, silently
    narrowing every future --check."""
    from ai_calibrator.eval import save_scorecard

    _project(tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", partial=True, results=[
        ResultRow(test_id="t1", output="o",
                  criteria=[CriterionResult(criterion_id="c1", passed=True)])]))

    r = runner.invoke(app, ["snapshot", str(tmp_path)])

    assert r.exit_code == 1, r.output
    assert "PARTIAL" in r.output
    assert not (tmp_path / "golden.json").exists()   # nothing written
    assert _no_crash(r)


def test_coverage_reports_a_real_number(tmp_path):
    """The percentage a user reads to decide whether their spec is adequately
    tested was rendered in a command body no test invoked — it could be made to
    print 100% unconditionally with the suite green."""
    p = _project(tmp_path)
    p.spec.eval_criteria.append(
        EvalCriterion(id="c2", description="never invents policy", weight=Weight.HIGH))
    save_project(p, tmp_path)   # t1 expects c1 only, so c2 is uncovered

    r = runner.invoke(app, ["coverage", str(tmp_path)])

    assert _no_crash(r)
    assert "c2" in r.output          # names the uncovered criterion
    assert "100%" not in r.output    # and does not claim full coverage


def test_drift_exits_two_on_a_regression(tmp_path, monkeypatch):
    """`drift` documents a CI-friendly exit 2 on regression. The path was never
    reached, so the signal a scheduled model-bump check depends on could be
    removed with the suite green."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import test_input_hash

    p = _project(tmp_path)
    # Baseline: the suite passed. Same question, so the two runs are comparable.
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[
        ResultRow(test_id="t1", output="please", input_hash=test_input_hash(p.tests[0]),
                  criteria=[CriterionResult(criterion_id="c1", passed=True)])]))
    _stub_engines(monkeypatch, reply="no.")      # the model now misses "please"

    r = runner.invoke(app, ["drift", str(tmp_path)])

    assert r.exit_code == 2, r.output
    assert "t1" in r.output
    assert _no_crash(r)


def test_drift_exits_zero_when_behavior_holds(tmp_path, monkeypatch):
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import test_input_hash

    p = _project(tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[
        ResultRow(test_id="t1", output="please", input_hash=test_input_hash(p.tests[0]),
                  criteria=[CriterionResult(criterion_id="c1", passed=True)])]))
    _stub_engines(monkeypatch, reply="please, of course")

    r = runner.invoke(app, ["drift", str(tmp_path)])
    assert r.exit_code == 0, r.output


# --- the one function that rmdir()s a user's directory ---------------------

def test_cleanup_removes_only_a_directory_we_created(tmp_path):
    from ai_calibrator.cli import _cleanup_empty_project_dir

    theirs = tmp_path / "theirs"
    theirs.mkdir()
    _cleanup_empty_project_dir(theirs, we_created=False)
    assert theirs.is_dir(), "removed a directory it did not create"


def test_cleanup_never_removes_a_directory_holding_anything(tmp_path):
    from ai_calibrator.cli import _cleanup_empty_project_dir

    d = tmp_path / "ours"
    d.mkdir()
    (d / ".lock").write_text("")          # a FILE — locking.py os.open()s it
    (d / "notes.txt").write_text("a file the user put here", encoding="utf-8")

    _cleanup_empty_project_dir(d, we_created=True)

    assert d.is_dir() and (d / "notes.txt").exists()


def test_cleanup_removes_our_own_lock_only_litter(tmp_path):
    from ai_calibrator.cli import _cleanup_empty_project_dir

    d = tmp_path / "ours"
    d.mkdir()
    (d / ".lock").write_text("")          # a FILE — locking.py os.open()s it

    _cleanup_empty_project_dir(d, we_created=True)

    assert not d.exists()


# --- every command, one smoke pass -----------------------------------------

def _command_names():
    return sorted(c.name or c.callback.__name__.rstrip("_").replace("_", "-")
                  for c in app.registered_commands)


# Commands that do not take a project directory as their first positional, and
# so are not meaningfully "pointed at an empty directory".
_NOT_PROJECT_SCOPED = {"init", "auth", "login", "serve", "merge", "import"}


@pytest.mark.parametrize("name", _command_names())
def test_every_command_documents_itself(name):
    """`--help` must render for every command. Rich wraps and colorizes, so this
    checks that it parsed and exited cleanly, not the wording."""
    r = runner.invoke(app, [name, "--help"])
    assert r.exit_code == 0, r.output
    assert _no_crash(r)


@pytest.mark.parametrize("name", [n for n in _command_names() if n not in _NOT_PROJECT_SCOPED])
def test_every_project_command_fails_cleanly_on_an_empty_directory(tmp_path, name):
    """Pointed at a directory with no project, every command must say so and
    exit non-zero — never crash, and never exit 0 having done nothing.

    This is the net: `cli.py` is 2,670 lines and the whole user-facing surface,
    so a stranger's PR that breaks one command's early validation should fail
    here rather than in someone's pipeline.
    """
    r = runner.invoke(app, [name, str(tmp_path)])
    assert _no_crash(r), f"{name} crashed: {r.exception!r}"
    assert r.exit_code != 0, f"{name} exited 0 on a directory with no project"


@pytest.mark.parametrize("name", [n for n in _command_names() if n not in _NOT_PROJECT_SCOPED])
def test_every_project_command_survives_a_corrupt_project(tmp_path, name):
    """The friendly-error path lives in the shared `_load`, but each command has
    to reach it. A raw YAML error escaping any one of them is a traceback in a
    user's terminal."""
    (tmp_path / "project.yaml").write_text("{ invalid: yaml: [", encoding="utf-8")
    r = runner.invoke(app, [name, str(tmp_path)])
    assert _no_crash(r), f"{name} crashed on a corrupt project: {r.exception!r}"
    assert r.exit_code != 0, f"{name} exited 0 on a corrupt project"


# --- the number of billed calls, before they are billed ---------------------

class _CountingEngine:
    """Counts every call, so an estimate can be checked against reality."""

    name = "counter@test"
    calls = 0

    def complete(self, prompt, *, system=None, schema=None):
        type(self).calls += 1
        import re
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        if ids:
            return {"results": [{"criterion_id": i, "passed": True, "score": 1.0,
                                 "rationale": "r"} for i in ids]}
        return "please, here is the answer"


def test_eval_states_its_call_count_and_the_estimate_is_right(tmp_path, monkeypatch):
    """A first user pointing this at a folder of documents has no way to guess
    what a command will spend. The estimate is worth nothing if it is wrong, so
    this checks it against the calls actually made."""
    import ai_calibrator.engines as engines

    p = _project(tmp_path)
    p.tests = [CaseModel(id=f"t{i}", input=f"q{i}", expects=["c1"]) for i in range(3)]
    save_project(p, tmp_path)

    _CountingEngine.calls = 0
    monkeypatch.setattr(engines, "get_engine", lambda spec: _CountingEngine())

    r = runner.invoke(app, ["eval", str(tmp_path)])
    assert r.exit_code == 0, r.output

    # The notice is printed BEFORE the work, and names a number.
    import re
    m = re.search(r"~(\d+) engine call\(s\)", r.output)
    assert m, r.output
    estimated = int(m.group(1))

    # 3 tests, answered once each. The criterion is a deterministic `check`, so
    # the judge is never called — the estimate is an upper bound, and must not
    # be an under-count.
    assert estimated >= _CountingEngine.calls, (estimated, _CountingEngine.calls)
    assert estimated == 6      # 3 tests × (1 answer + 1 judge pass)


def test_rightsize_states_the_cost_of_the_whole_matrix(tmp_path, monkeypatch):
    import ai_calibrator.engines as engines

    p = _project(tmp_path)
    p.tests = [CaseModel(id=f"t{i}", input=f"q{i}", expects=["c1"]) for i in range(4)]
    save_project(p, tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _CountingEngine())

    r = runner.invoke(app, ["rightsize", str(tmp_path), "--models", "a@ollama,b@ollama"])

    import re
    m = re.search(r"~(\d+) engine call\(s\)", r.output)
    assert m, r.output
    assert int(m.group(1)) == 4 * 2 * 2      # tests × models × (answer + grade)
    assert "2 model(s)" in r.output


def test_interview_states_one_call_per_gap(tmp_path, monkeypatch):
    import ai_calibrator.engines as engines
    from ai_calibrator.models import Gap

    p = _project(tmp_path)
    p.gaps = [Gap(dimension=d) for d in ("scope", "tone", "escalation")]
    save_project(p, tmp_path)

    class _Q:
        name = "q@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"question": "?", "draft_answer": "d", "rationale": "r"}

    monkeypatch.setattr(engines, "get_engine", lambda spec: _Q())

    r = runner.invoke(app, ["interview", str(tmp_path)], input="")

    import re
    m = re.search(r"~(\d+) engine call\(s\)", r.output)
    assert m, r.output
    assert int(m.group(1)) == 3          # one per gap
