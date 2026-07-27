"""CLI robustness — friendly errors instead of raw tracebacks, port validation.

Regression tests: unhandled YAMLError / ValidationError in _load across every
project command, and unvalidated --port crashing uvicorn with an OverflowError
after printing a green "success" line.
"""

from __future__ import annotations

from typer.testing import CliRunner

from ai_calibrator.cli import app

runner = CliRunner()


def _has_no_traceback(output: str) -> bool:
    return "Traceback (most recent call last)" not in output


def test_status_on_malformed_yaml_is_friendly(tmp_path):
    (tmp_path / "project.yaml").write_text("{ invalid: yaml: [")
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid or corrupted" in result.output
    assert _has_no_traceback(result.output)


def test_status_on_incomplete_project_is_friendly(tmp_path):
    # Valid YAML, but missing the required `goal` field → pydantic ValidationError.
    (tmp_path / "project.yaml").write_text("name: only-a-name\n")
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid or corrupted" in result.output
    assert _has_no_traceback(result.output)


def test_status_on_empty_project_file_is_friendly(tmp_path):
    # yaml.safe_load("") is None → Project.model_validate(None) raises.
    (tmp_path / "project.yaml").write_text("")
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid or corrupted" in result.output
    assert _has_no_traceback(result.output)


def test_compile_on_corrupt_project_is_friendly(tmp_path):
    # The friendly handling lives in the shared _load, so every command benefits.
    (tmp_path / "project.yaml").write_text(": : :\n  - broken")
    result = runner.invoke(app, ["compile", str(tmp_path)])
    assert result.exit_code == 1
    assert _has_no_traceback(result.output)


def test_serve_rejects_out_of_range_port(tmp_path):
    for bad in ("-1", "0", "70000", "99999", "65536"):   # 0 now rejected (would print a wrong URL)
        result = runner.invoke(app, ["serve", f"--port={bad}"])
        assert result.exit_code == 1, f"port {bad} should be rejected"
        assert "between 1 and 65535" in result.output
        assert _has_no_traceback(result.output)


def test_serve_help_works():
    # Sanity: the command still parses and the option is documented. Rich colorizes
    # + wraps help (and CI has no TTY), so strip ANSI and collapse whitespace before
    # checking — otherwise `--port` can be split or styled and the substring misses.
    import re
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    compact = re.sub(r"\s+", "", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert "--port" in compact


def test_redteam_rejects_out_of_range_max_probes(tmp_path):
    # Validation happens before any project load / engine init (CLI/API parity).
    for bad in ("0", "51", "100"):
        result = runner.invoke(app, ["redteam", str(tmp_path), f"--max-probes={bad}"])
        assert result.exit_code == 1
        assert "between 1 and 50" in result.output
        assert _has_no_traceback(result.output)


def test_teach_rejects_out_of_range_n(tmp_path):
    for bad in ("0", "21", "100"):
        result = runner.invoke(app, ["teach", str(tmp_path), f"--n={bad}"])
        assert result.exit_code == 1
        assert "between 1 and 20" in result.output
        assert _has_no_traceback(result.output)


def test_init_rejects_path_like_name():
    # The positional is a NAME (folder name); separators/traversal → friendly error.
    for bad in ["../evil", "a/b", "..", "/abs"]:
        r = runner.invoke(app, ["init", bad, "--goal", "g"])
        assert r.exit_code == 1 and "simple folder name" in r.output and _has_no_traceback(r.output)


def test_init_rejects_empty_name():
    r = runner.invoke(app, ["init", "", "--goal", "g"])
    assert r.exit_code == 1 and "must not be empty" in r.output and _has_no_traceback(r.output)


def test_init_accepts_plain_name_and_explicit_path(tmp_path):
    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert runner.invoke(app, ["init", "myproj", "--goal", "g"]).exit_code == 0
    finally:
        os.chdir(cwd)
    # --path stays free-form for explicit locations (like `git init <path>`)
    assert runner.invoke(app, ["init", "anyname", "--goal", "g", "--path", str(tmp_path / "loc")]).exit_code == 0


def test_init_writes_gitignore_and_honest_engines_line(tmp_path):
    """init must write a .gitignore so a future `git init` can't commit secrets or
    eval runs, and must not claim one engine covers 'all roles' when judge/subject
    differ."""
    r = runner.invoke(app, ["init", "proj", "--goal", "g", "--path", str(tmp_path / "proj")])
    assert r.exit_code == 0
    gi = (tmp_path / "proj" / ".gitignore").read_text(encoding="utf-8")
    assert "evals/" in gi and ".env" in gi and "*.key" in gi
    assert "all roles" not in r.output          # the misleading phrasing is gone
    assert "(judge)" in r.output and "(subject)" in r.output


def test_init_rejects_overlong_name():
    r = runner.invoke(app, ["init", "a" * 1000, "--goal", "g"])
    assert r.exit_code == 1 and "too long" in r.output and _has_no_traceback(r.output)


def test_help_survives_ascii_and_cp1252_terminals():
    """Regression: `calibrate --help` crashed with UnicodeEncodeError under limited
    encodings (Rich rendering the → glyph). Glyphs must degrade, never crash."""
    import os
    import subprocess
    import sys

    for enc in ("ascii", "cp1252"):
        env = {**os.environ, "PYTHONIOENCODING": enc}
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.argv=['calibrate','--help']; from ai_calibrator.cli import main; main()"],
            env=env, capture_output=True, timeout=60)  # bytes: the child writes in `enc`
        stderr = r.stderr.decode(enc, errors="replace")
        assert r.returncode == 0, f"{enc}: {stderr[-300:]}"
        assert "UnicodeEncodeError" not in stderr


def test_snapshot_on_corrupt_scorecard_is_friendly(tmp_path):
    """A corrupt scorecard must produce a friendly error, not a raw traceback."""
    d = tmp_path / "p"
    d.mkdir()
    (d / "project.yaml").write_text("name: p\ngoal: g\n")
    run = d / "evals" / "run-0001"
    run.mkdir(parents=True)
    (run / "scorecard.json").write_text("{ truncated json")
    r = runner.invoke(app, ["snapshot", str(d)])
    assert r.exit_code == 1 and "Could not read scorecard" in r.output and _has_no_traceback(r.output)


def test_eval_rounds_bounds_validated_before_project(tmp_path):
    """Parity with API EvalBody (ge=1 le=100): a bad --rounds is a validation
    error, not masked by a 'nothing to evaluate' state message."""
    d = tmp_path / "p"; d.mkdir()
    (d / "project.yaml").write_text("name: p\ngoal: g\n")   # no spec/tests
    for bad in ("0", "101"):
        r = runner.invoke(app, ["eval", str(d), f"--rounds={bad}"])
        assert r.exit_code == 1 and "between 1 and 100" in r.output and _has_no_traceback(r.output)


def test_train_requires_a_compiled_spec(tmp_path):
    (tmp_path / "project.yaml").write_text("name: p\ngoal: g\n")   # no spec
    r = runner.invoke(app, ["train", str(tmp_path)])
    assert r.exit_code == 1 and "compile" in r.output and _has_no_traceback(r.output)


def test_train_requires_examples(tmp_path):
    # a compiled spec but no examples → the Advanced tier has nothing to learn from
    import yaml
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "p", "goal": "g",
        "spec": {"goal": "g", "eval_criteria": [{"id": "c1", "description": "d", "weight": "high"}]},
    }))
    r = runner.invoke(app, ["train", str(tmp_path)])
    assert r.exit_code == 1 and "training examples" in r.output.lower() and _has_no_traceback(r.output)


def test_train_offers_deps_and_respects_decline(tmp_path, monkeypatch):
    """If the training stack is missing, it OFFERS to install and does nothing on
    'no' — never a silent pip install."""
    import importlib.util
    import yaml
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "p", "goal": "g",
        "spec": {"goal": "g", "examples": [{"input": "hi", "good_output": "hello there"}],
                 "eval_criteria": [{"id": "c1", "description": "d", "weight": "high"}]},
    }))
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda m: None if m == "torch" else real(m))
    r = runner.invoke(app, ["train", str(tmp_path)], input="n\n")   # decline the install prompt
    assert r.exit_code == 1
    assert "torch" in r.output and "-e '.[train]'" in r.output   # guided, not silent
    assert _has_no_traceback(r.output)


def test_examples_requires_spec_then_imports(tmp_path):
    import yaml
    (tmp_path / "project.yaml").write_text("name: p\ngoal: g\n")   # no spec
    r = runner.invoke(app, ["examples", str(tmp_path)])
    assert r.exit_code == 1 and "compile" in r.output and _has_no_traceback(r.output)
    # give it a spec, then import a CSV through the command
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({"name": "p", "goal": "g", "spec": {"goal": "g"}}))
    csv = tmp_path / "qa.csv"; csv.write_text("question,answer\nHi?,Hello!\nBye?,See ya!\n")
    r = runner.invoke(app, ["examples", str(tmp_path), "--import", str(csv)])
    assert r.exit_code == 0 and "Imported 2" in r.output and "48 more" in r.output
    from ai_calibrator.store import load_project
    assert len(load_project(tmp_path).spec.examples) == 2


def test_init_reserved_name_is_friendly_not_traceback(tmp_path, monkeypatch):
    """Names the model validator rejects (reserved device names, trailing dot,
    control chars) must produce a friendly message + exit 1, not a raw pydantic
    traceback."""
    from typer.testing import CliRunner

    from ai_calibrator.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    for bad in ("CON", "myproj.", "a\tb"):
        result = runner.invoke(app, ["init", bad, "--goal", "g"])
        assert result.exit_code == 1, (bad, result.output)
        assert "Traceback" not in result.output
        assert "Invalid project name" in result.output


def test_version_flag_prints_version_and_exits():
    """`--version` is the first thing a user or bug reporter runs; it must work
    without a project, print the installed version, and exit 0."""
    from ai_calibrator import __version__

    for flag in ("--version", "-V"):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0, flag
        assert __version__ in result.output, flag
        assert _has_no_traceback(result.output)


def test_init_rejects_empty_goal(tmp_path):
    r = runner.invoke(app, ["init", "p", "--goal", "  ", "--path", str(tmp_path / "p")])
    assert r.exit_code == 1 and "goal" in r.output.lower() and _has_no_traceback(r.output)


def test_init_onto_existing_file_is_friendly(tmp_path):
    (tmp_path / "taken").write_text("i am a file")
    r = runner.invoke(app, ["init", "x", "--goal", "g", "--path", str(tmp_path / "taken")])
    assert r.exit_code == 1 and "already exists" in r.output and _has_no_traceback(r.output)


def test_command_on_missing_project_leaves_no_junk_dir(tmp_path):
    # eval on a typo'd/nonexistent project must NOT create a directory or .lock
    target = tmp_path / "typo"
    r = runner.invoke(app, ["eval", str(target)])
    assert r.exit_code == 1 and "No calibrator project" in r.output
    assert not target.exists()  # nothing littered


def test_ci_json_emits_json_on_cannot_gate(tmp_path):
    import json as _json
    d = tmp_path / "p"
    d.mkdir()
    (d / "project.yaml").write_text("name: p\ngoal: g\n")  # no spec/tests
    r = runner.invoke(app, ["ci", str(d), "--json"])
    assert r.exit_code == 1
    payload = _json.loads(r.output.strip())  # must be valid JSON on the cannot-gate path
    assert payload["ok"] is False and payload["gate"] == "error"


def test_help_has_no_internal_milestone_jargon():
    """User-facing help must not leak internal milestone markers (M1/M3+) or
    section-sign references — they mean nothing to a user."""
    import re

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert not re.search(r"\(M\d+\+?\)", result.output)
    assert "§" not in result.output




def test_teach_records_each_judgment_exactly_once(tmp_path, monkeypatch):
    """Two-phase save: phase 1 checkpoints the judgments, phase 2 folds in the
    inferred standards. Recording the judgments in BOTH doubles the examples every
    session and prints the pre-duplication count."""
    import ai_calibrator.teach as teach_mod
    from ai_calibrator.models import Project
    from ai_calibrator.store import load_project, save_project

    class _Eng:
        name = "fake@test"

    save_project(Project(name="p", goal="g"), tmp_path)
    monkeypatch.setattr("ai_calibrator.engines.get_engine", lambda spec: _Eng())
    monkeypatch.setattr(teach_mod, "propose_candidates",
                        lambda *a, **k: [teach_mod.Candidate(id="ex1", input="q", output="a")])
    monkeypatch.setattr(teach_mod, "infer_standards",
                        lambda *a, **k: {"standards": ["Always cite the policy."], "do_not": []})

    r = runner.invoke(app, ["teach", str(tmp_path), "--n", "1"], input="y\n\n")
    assert r.exit_code == 0, r.output
    spec = load_project(tmp_path).spec
    assert len(spec.examples) == 1                  # one judgment → exactly one example
    assert "recorded 1 example" in r.output         # and the printed count is the truth




def test_merge_audit_records_the_resolution_that_shipped(tmp_path, monkeypatch):
    """persona is merged as a WHOLE object (first stakeholder by name with any
    persona field), so the reconciliation file must not claim a per-field winner
    the merge never picked."""
    import yaml

    import ai_calibrator.stakeholders as stake
    from ai_calibrator.models import BehaviorSpec, Persona, Project
    from ai_calibrator.store import load_project, save_project

    class _Eng:
        name = "fake@test"

    dirs = {}
    for nm, persona in (("alpha", Persona(reading_level="grade 5")),
                        ("beta", Persona(voice="terse")),
                        ("gamma", Persona(voice="chatty"))):
        dirs[nm] = tmp_path / nm
        save_project(Project(name=nm, goal="g", spec=BehaviorSpec(goal="g", persona=persona)), dirs[nm])
    monkeypatch.setattr("ai_calibrator.engines.get_engine", lambda spec: _Eng())
    monkeypatch.setattr(stake, "detect_conflicts", lambda statements, engine: [])

    out = tmp_path / "merged"
    r = runner.invoke(app, ["merge", str(out), "--from", str(dirs["alpha"]),
                            "--from", str(dirs["beta"]), "--from", str(dirs["gamma"])])
    assert r.exit_code == 0, r.output
    audit = yaml.safe_load((out / "reconciliation.yaml").read_text(encoding="utf-8"))
    voice = next(f for f in audit["field_conflicts"] if f["field"] == "persona.voice")
    shipped = load_project(out).spec.persona.voice
    # Per-field resolution: alpha supplies no voice, so beta's ships.
    assert shipped == "terse"
    assert voice["resolved_to"]["value"] == shipped         # audit says what shipped




def test_merge_report_only_works_when_the_destination_exists(tmp_path, monkeypatch):
    """--report-only writes nothing, so an already-merged destination must not
    block a fresh read-only conflict report."""
    import ai_calibrator.stakeholders as stake
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import save_project

    class _Eng:
        name = "fake@test"

    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "merged"
    save_project(Project(name="a", goal="g", spec=BehaviorSpec(goal="g", standards=["x"])), a)
    save_project(Project(name="b", goal="g", spec=BehaviorSpec(goal="g", standards=["y"])), b)
    save_project(Project(name="merged", goal="g", spec=BehaviorSpec(goal="g")), out)
    monkeypatch.setattr("ai_calibrator.engines.get_engine", lambda spec: _Eng())
    monkeypatch.setattr(stake, "detect_conflicts", lambda statements, engine: [])

    r = runner.invoke(app, ["merge", str(out), "--from", str(a), "--from", str(b), "--report-only"])
    assert r.exit_code == 0, r.output
    assert "report only" in r.output and _has_no_traceback(r.output)




def test_train_dependency_check_catches_installed_but_too_old():
    """find_spec only proves importability; the version floors are what the
    generated trainer actually needs, so an old trl must count as needed."""
    from ai_calibrator import cli

    assert cli._dep_satisfied("pytest", "pytest>=1.0") is True
    assert cli._dep_satisfied("pytest", "pytest>=999.0") is False
    assert cli._dep_satisfied("no_such_module_xyz", "no_such_module_xyz>=1.0") is False




def test_diff_prints_a_knowledge_only_change(tmp_path):
    """Knowledge sources change the deployed system prompt, so a knowledge-only
    diff must show the change instead of printing an empty report."""
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import save_project

    a, b = tmp_path / "a", tmp_path / "b"
    save_project(Project(name="a", goal="g", spec=BehaviorSpec(goal="g")), a)
    save_project(Project(name="b", goal="g",
                         spec=BehaviorSpec(goal="g", knowledge_sources=["refund-policy.pdf"])), b)
    r = runner.invoke(app, ["diff", str(a), str(b)])
    assert r.exit_code == 0, r.output
    assert "refund-policy.pdf" in r.output and "No behavior change" not in r.output




def test_retrieval_off_reason_separates_missing_extra_from_broken_index(tmp_path, monkeypatch):
    """Two failures with two different fixes — the message must not send an owner
    hunting for a broken index they never built."""
    from ai_calibrator import cli, rag

    monkeypatch.setattr(rag, "index_available", lambda: False)
    msg = cli._retrieval_off_reason(tmp_path)
    assert "not installed" in msg and "unusable" not in msg

    monkeypatch.setattr(rag, "index_available", lambda: True)
    monkeypatch.setattr(rag, "probe", lambda d: "no index")
    assert cli._retrieval_off_reason(tmp_path) == "your documents are NOT in play"
    monkeypatch.setattr(rag, "probe", lambda d: "RuntimeError: table missing")
    assert "index present but unusable" in cli._retrieval_off_reason(tmp_path)




def test_add_check_rejects_an_empty_needle(tmp_path):
    """`contains` with the value omitted can never fail — a criterion that would
    report PASS on output nothing graded."""
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from ai_calibrator.store import load_project, save_project

    save_project(Project(name="p", goal="g", spec=BehaviorSpec(
        goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])), tmp_path)
    for kind in ("contains", "not_contains", "regex"):
        r = runner.invoke(app, ["add-check", str(tmp_path), "c1", kind])
        assert r.exit_code == 1, (kind, r.output)
        assert "needs a value" in r.output and _has_no_traceback(r.output)
    assert load_project(tmp_path).spec.eval_criteria[0].check is None
    # kinds that take no value, and a real needle, still work
    assert runner.invoke(app, ["add-check", str(tmp_path), "c1", "non_empty"]).exit_code == 0
    assert runner.invoke(app, ["add-check", str(tmp_path), "c1", "contains", "30-day"]).exit_code == 0




def test_ingest_of_an_emptied_materials_dir_clears_the_old_corpus(tmp_path, monkeypatch):
    """Deleting every material and re-ingesting must PURGE facts, gaps and index.

    The command refused an empty materials folder outright, so the corpus built
    from files the owner had deleted survived and kept feeding every graded and
    served prompt."""
    import ai_calibrator.engines as engines
    from ai_calibrator.models import Gap, Material, Project
    from ai_calibrator.store import load_project, save_project

    class _Engine:
        name = "fake@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"facts": [], "gaps": []}

    monkeypatch.setattr(engines, "get_engine", lambda spec: _Engine())

    proj = Project(name="p", goal="g")
    proj.materials = [Material(path="faq.md", kind="md", summary="old policy")]
    proj.facts = ["Returns are accepted for 30 days."]
    proj.gaps = [Gap(dimension="tone")]
    save_project(proj, tmp_path)
    (tmp_path / "materials").mkdir(exist_ok=True)  # every file deleted
    (tmp_path / "knowledge.lancedb").mkdir()  # index built from those files

    result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "clearing" in result.output
    assert _has_no_traceback(result.output)

    after = load_project(tmp_path)
    assert after.materials == [] and after.facts == [] and after.gaps == []
    assert not (tmp_path / "knowledge.lancedb").exists()


def test_ingest_still_guides_a_project_that_never_had_materials(tmp_path):
    from ai_calibrator.models import Project
    from ai_calibrator.store import save_project

    save_project(Project(name="p", goal="g"), tmp_path)
    result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 1
    assert "No materials found" in result.output
    assert _has_no_traceback(result.output)


def test_ingest_warns_when_only_part_of_the_corpus_was_analyzed(tmp_path, monkeypatch):
    """Files past the extractor's context cap inform nothing — say so instead of
    printing a file count that implies they all shaped the gap list."""
    import ai_calibrator.engines as engines
    import ai_calibrator.ingest as ing
    from ai_calibrator.models import Project
    from ai_calibrator.store import save_project

    class _Engine:
        name = "fake@test"

        def complete(self, prompt, *, system=None, schema=None):
            return {"facts": ["one fact"], "gaps": []}

    monkeypatch.setattr(engines, "get_engine", lambda spec: _Engine())
    monkeypatch.setattr(ing, "MAX_EXTRACT_CHARS", 60)

    save_project(Project(name="p", goal="g"), tmp_path)
    materials = tmp_path / "materials"
    materials.mkdir(exist_ok=True)
    (materials / "a.md").write_text("a" * 400)
    (materials / "b.md").write_text("b" * 400)

    result = runner.invoke(app, ["ingest", str(tmp_path), "--no-index"])
    assert result.exit_code == 0, result.output
    assert "1 of 2 file(s) fit" in result.output
    assert _has_no_traceback(result.output)
