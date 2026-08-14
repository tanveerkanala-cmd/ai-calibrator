"""CLI robustness — friendly errors instead of raw tracebacks, port validation.

Regression tests: unhandled YAMLError / ValidationError in _load across every
project command, and unvalidated --port crashing uvicorn with an OverflowError
after printing a green "success" line.
"""

from __future__ import annotations

from typer.testing import CliRunner

from ai_calibrator.cli import app

runner = CliRunner()


def _has_no_traceback(result) -> bool:
    """Whether the command handled its own error instead of crashing.

    Must be asked of the RESULT, not of ``result.output``: CliRunner catches an
    unhandled exception and stores it on ``result.exception``, writing nothing
    to the output. Checking the output for "Traceback" could therefore never be
    False — a command that printed a friendly message and then crashed passed
    every assertion in this file. ``SystemExit`` is how ``typer.Exit`` signals
    an ordinary non-zero exit, so it is not a crash.
    """
    exc = result.exception
    return exc is None or isinstance(exc, SystemExit)


def test_status_on_malformed_yaml_is_friendly(tmp_path):
    (tmp_path / "project.yaml").write_text("{ invalid: yaml: [")
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid or corrupted" in result.output
    assert _has_no_traceback(result)


def test_status_on_incomplete_project_is_friendly(tmp_path):
    # Valid YAML, but missing the required `goal` field → pydantic ValidationError.
    (tmp_path / "project.yaml").write_text("name: only-a-name\n")
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid or corrupted" in result.output
    assert _has_no_traceback(result)


def test_status_on_empty_project_file_is_friendly(tmp_path):
    # yaml.safe_load("") is None → Project.model_validate(None) raises.
    (tmp_path / "project.yaml").write_text("")
    result = runner.invoke(app, ["status", str(tmp_path)])
    assert result.exit_code == 1
    assert "invalid or corrupted" in result.output
    assert _has_no_traceback(result)


def test_compile_on_corrupt_project_is_friendly(tmp_path):
    # The friendly handling lives in the shared _load, so every command benefits.
    (tmp_path / "project.yaml").write_text(": : :\n  - broken")
    result = runner.invoke(app, ["compile", str(tmp_path)])
    assert result.exit_code == 1
    assert _has_no_traceback(result)


def test_serve_rejects_out_of_range_port(tmp_path):
    for bad in ("-1", "0", "70000", "99999", "65536"):   # 0 now rejected (would print a wrong URL)
        result = runner.invoke(app, ["serve", f"--port={bad}"])
        assert result.exit_code == 1, f"port {bad} should be rejected"
        assert "between 1 and 65535" in result.output
        assert _has_no_traceback(result)


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
        assert _has_no_traceback(result)


def test_teach_rejects_out_of_range_n(tmp_path):
    for bad in ("0", "21", "100"):
        result = runner.invoke(app, ["teach", str(tmp_path), f"--n={bad}"])
        assert result.exit_code == 1
        assert "between 1 and 20" in result.output
        assert _has_no_traceback(result)


def test_init_rejects_path_like_name():
    # The positional is a NAME (folder name); separators/traversal → friendly error.
    for bad in ["../evil", "a/b", "..", "/abs"]:
        r = runner.invoke(app, ["init", bad, "--goal", "g"])
        assert r.exit_code == 1 and "simple folder name" in r.output and _has_no_traceback(r)


def test_init_rejects_empty_name():
    r = runner.invoke(app, ["init", "", "--goal", "g"])
    assert r.exit_code == 1 and "must not be empty" in r.output and _has_no_traceback(r)


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
    assert r.exit_code == 1 and "too long" in r.output and _has_no_traceback(r)


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
    assert r.exit_code == 1 and "Could not read scorecard" in r.output and _has_no_traceback(r)


def test_eval_rounds_bounds_validated_before_project(tmp_path):
    """Parity with API EvalBody (ge=1 le=100): a bad --rounds is a validation
    error, not masked by a 'nothing to evaluate' state message."""
    d = tmp_path / "p"; d.mkdir()
    (d / "project.yaml").write_text("name: p\ngoal: g\n")   # no spec/tests
    for bad in ("0", "101"):
        r = runner.invoke(app, ["eval", str(d), f"--rounds={bad}"])
        assert r.exit_code == 1 and "between 1 and 100" in r.output and _has_no_traceback(r)


def test_train_requires_a_compiled_spec(tmp_path):
    (tmp_path / "project.yaml").write_text("name: p\ngoal: g\n")   # no spec
    r = runner.invoke(app, ["train", str(tmp_path)])
    assert r.exit_code == 1 and "compile" in r.output and _has_no_traceback(r)


def test_train_requires_examples(tmp_path):
    # a compiled spec but no examples → the Advanced tier has nothing to learn from
    import yaml
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "p", "goal": "g",
        "spec": {"goal": "g", "eval_criteria": [{"id": "c1", "description": "d", "weight": "high"}]},
    }))
    r = runner.invoke(app, ["train", str(tmp_path)])
    assert r.exit_code == 1 and "training examples" in r.output.lower() and _has_no_traceback(r)


def test_train_offers_deps_and_respects_decline(tmp_path, monkeypatch):
    """If the training stack is missing, it OFFERS to install and does nothing on
    'no' — never a silent pip install."""
    import importlib.util
    import yaml
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "p", "goal": "g",
        "spec": {"goal": "g",
                 "examples": [{"input": "hi", "good_output": "hello there", "source": "human"}],
                 "eval_criteria": [{"id": "c1", "description": "d", "weight": "high"}]},
    }))
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda m: None if m == "torch" else real(m))
    r = runner.invoke(app, ["train", str(tmp_path)], input="n\n")   # decline the install prompt
    assert r.exit_code == 1
    assert "torch" in r.output and "-e '.[train]'" in r.output   # guided, not silent
    assert _has_no_traceback(r)


def test_examples_requires_spec_then_imports(tmp_path):
    import yaml
    (tmp_path / "project.yaml").write_text("name: p\ngoal: g\n")   # no spec
    r = runner.invoke(app, ["examples", str(tmp_path)])
    assert r.exit_code == 1 and "compile" in r.output and _has_no_traceback(r)
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
        assert _has_no_traceback(result)


def test_init_rejects_empty_goal(tmp_path):
    r = runner.invoke(app, ["init", "p", "--goal", "  ", "--path", str(tmp_path / "p")])
    assert r.exit_code == 1 and "goal" in r.output.lower() and _has_no_traceback(r)


def test_init_onto_existing_file_is_friendly(tmp_path):
    (tmp_path / "taken").write_text("i am a file")
    r = runner.invoke(app, ["init", "x", "--goal", "g", "--path", str(tmp_path / "taken")])
    assert r.exit_code == 1 and "already exists" in r.output and _has_no_traceback(r)


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
    assert "report only" in r.output and _has_no_traceback(r)




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


def test_diff_prints_an_examples_only_change(tmp_path):
    """Examples are what the fine-tune trains on, and teach/absorb/merge change
    nothing else — an empty report under a "changed" verdict reviews a flipped
    verdict as no change at all."""
    from ai_calibrator.models import BehaviorSpec, Example, Project
    from ai_calibrator.store import save_project

    a, b = tmp_path / "a", tmp_path / "b"
    q = "Can I return after 30 days?"
    save_project(Project(name="a", goal="g", spec=BehaviorSpec(
        goal="g", examples=[Example(input=q, good_output="Yes — within 60.", source="human")])), a)
    # The same answer, moved from good to bad: the flip `specdiff` exists to catch.
    save_project(Project(name="b", goal="g", spec=BehaviorSpec(
        goal="g", examples=[Example(input=q, bad_output="Yes — within 60.", source="human")])), b)

    r = runner.invoke(app, ["diff", str(a), str(b)])

    assert r.exit_code == 0, r.output
    assert r.output.strip(), "a changed spec printed an empty report"
    assert q in r.output


def test_diff_never_reports_a_change_it_prints_nothing_about(tmp_path, monkeypatch):
    """The report renders one hand-written section per diff field, so a field
    added to SpecDiff and not to that list disappears silently — which is how the
    examples case came to print nothing at all."""
    import ai_calibrator.specdiff as specdiff
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import save_project

    a, b = tmp_path / "a", tmp_path / "b"
    save_project(Project(name="a", goal="g", spec=BehaviorSpec(goal="g")), a)
    save_project(Project(name="b", goal="g", spec=BehaviorSpec(goal="g")), b)

    class _UnrenderedChange(specdiff.SpecDiff):
        """`changed`, through nothing the sections below know how to render."""
        @property
        def changed(self) -> bool:
            return True

    monkeypatch.setattr(specdiff, "diff_specs", lambda before, after: _UnrenderedChange())

    r = runner.invoke(app, ["diff", str(a), str(b)])

    assert r.exit_code == 0, r.output
    assert r.output.strip(), "a changed spec printed an empty report"




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
        assert "needs a value" in r.output and _has_no_traceback(r)
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
    assert _has_no_traceback(result)

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
    assert _has_no_traceback(result)


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
    # Neither file FIT: a.md was truncated at the cap and b.md never reached the
    # extractor. Counting a partially-read file as analyzed is what let a single
    # oversized material report "1 of 1" and suppress this warning entirely.
    assert "0 of 2 file(s) fit" in result.output
    assert _has_no_traceback(result)


def test_merge_writes_the_protective_gitignore(tmp_path, monkeypatch):
    """`init` and `import` both write it; merge was the last creation path that
    did not — and a merged org project is among the likeliest to end up in git,
    where its logs/, evals/ and any .env would be committable."""
    import ai_calibrator.stakeholders as stake
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import save_project

    class _Eng:
        name = "fake@test"

    dirs = {}
    for nm in ("alpha", "beta"):
        dirs[nm] = tmp_path / nm
        save_project(Project(name=nm, goal="g",
                             spec=BehaviorSpec(goal="g", standards=[f"{nm} rule"])), dirs[nm])
    monkeypatch.setattr("ai_calibrator.engines.get_engine", lambda spec: _Eng())
    monkeypatch.setattr(stake, "detect_conflicts", lambda statements, engine: [])

    out = tmp_path / "org"
    r = runner.invoke(app, ["merge", str(out), "--from", str(dirs["alpha"]),
                            "--from", str(dirs["beta"])])
    assert r.exit_code == 0, r.output
    gitignore = out / ".gitignore"
    assert gitignore.is_file(), "merged project has no .gitignore"
    body = gitignore.read_text(encoding="utf-8")
    assert "logs/" in body and ".env" in body


# --- a person's typed work survives an abort -------------------------------

def _interview_ready(tmp_path):
    """A project with gaps and two already-generated, unanswered questions, so
    `interview` goes straight to the prompt loop without an engine call."""
    from ai_calibrator.models import Gap, InterviewItem, Project
    from ai_calibrator.store import save_project

    p = Project(name="p", goal="answer support questions")
    p.gaps = [Gap(dimension="scope"), Gap(dimension="tone")]
    p.interview = [
        InterviewItem(id="i1", dimension="scope", question="Who do you serve?", draft_answer="d1"),
        InterviewItem(id="i2", dimension="tone", question="How formal?", draft_answer="d2"),
    ]
    save_project(p, tmp_path)
    return p


def test_interview_keeps_answers_typed_before_an_abort(tmp_path):
    """Answers accumulated in memory and were written only after the LAST
    question, so a Ctrl-C at question 2 of 2 destroyed the answer to question 1.
    An answer a person typed is their work."""
    from ai_calibrator.store import load_project

    _interview_ready(tmp_path)
    # Answer the first question, then send EOF at the second.
    result = runner.invoke(app, ["interview", str(tmp_path)], input="small businesses\n")

    assert result.exit_code == 1
    assert "Stopped early" in result.output
    assert _has_no_traceback(result)

    saved = load_project(tmp_path)
    by_id = {it.id: it for it in saved.interview}
    assert by_id["i1"].answer == "small businesses"      # kept, not discarded
    assert by_id["i1"].answer_source == "human"
    assert not by_id["i2"].answer                        # never reached


def test_interview_aborted_before_any_answer_saves_nothing(tmp_path):
    from ai_calibrator.store import load_project

    _interview_ready(tmp_path)
    result = runner.invoke(app, ["interview", str(tmp_path)], input="")

    assert result.exit_code == 1
    assert "nothing to save" in result.output
    assert _has_no_traceback(result)
    assert not any(it.answer for it in load_project(tmp_path).interview)


def _project_with_criterion(tmp_path):
    from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from ai_calibrator.store import save_project

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="cites the policy", weight=Weight.HIGH)])
    save_project(p, tmp_path)
    return p


def test_add_check_rejects_an_uncompilable_regex(tmp_path):
    """A pattern that cannot compile was stored with a green success message and
    then failed its criterion on every eval, every CI gate, and — under
    `run --guard` — every live answer. The integer kinds were already validated
    here; this one was not."""
    from ai_calibrator.store import load_project

    _project_with_criterion(tmp_path)
    r = runner.invoke(app, ["add-check", str(tmp_path), "c1", "regex", "(unclosed"])

    assert r.exit_code == 1
    assert "does not compile" in r.output
    assert "✓" not in r.output                     # no success line
    assert _has_no_traceback(r)
    assert load_project(tmp_path).spec.eval_criteria[0].check is None   # nothing stored


def test_add_check_still_accepts_a_valid_regex(tmp_path):
    from ai_calibrator.store import load_project

    _project_with_criterion(tmp_path)
    r = runner.invoke(app, ["add-check", str(tmp_path), "c1", "regex", r"\d+ days?"])

    assert r.exit_code == 0 and _has_no_traceback(r)
    check = load_project(tmp_path).spec.eval_criteria[0].check
    assert check.kind == "regex" and check.value == r"\d+ days?"


def test_finetune_refuses_gate_flags_without_gate(tmp_path):
    """--baseline/--candidate were read only inside `if gate:`, so omitting
    --gate silently discarded them and ran the BUILD path instead, rewriting
    <project>/finetune/ and exiting 0."""
    _project_with_criterion(tmp_path)
    r = runner.invoke(app, ["finetune", str(tmp_path),
                            "--baseline", "run-0001", "--candidate", "run-0002"])

    assert r.exit_code == 1
    assert "only applies with --gate" in r.output
    assert _has_no_traceback(r)
    assert not (tmp_path / "finetune").exists()    # the build path did not run


def test_train_engine_refuses_candidate_without_prove(tmp_path):
    _project_with_criterion(tmp_path)
    r = runner.invoke(app, ["train-engine", "judge", str(tmp_path),
                            "--candidate", "qwen2.5:7b@ollama"])

    assert r.exit_code == 1
    assert "only applies with --prove" in r.output
    assert _has_no_traceback(r)


def test_import_rejects_an_empty_goal(tmp_path):
    """`init` rejects an empty goal with a message explaining why the pipeline
    needs it. `import` fed it straight into the extraction prompt and into
    BehaviorSpec.goal — after a billed engine call."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a support agent.", encoding="utf-8")

    r = runner.invoke(app, ["import", str(tmp_path / "proj"),
                            "--prompt", str(prompt_file), "--goal", "   "])

    assert r.exit_code == 1
    assert "must not be empty" in r.output
    assert _has_no_traceback(r)
    assert not (tmp_path / "proj").exists()        # nothing created, nothing billed


# --- train: the install it offers has to be the one the trainer needs -------

def test_the_training_install_floors_are_the_ones_the_trainer_needs():
    """`train` installs requirement STRINGS so an already-present but too-old
    package is upgraded. A floor below what the generated train.py calls leaves
    that package untouched, and the crash it causes is then reported as a
    memory problem the owner does not have."""
    import tomllib
    from pathlib import Path

    import pytest

    from ai_calibrator import cli

    pyproject = Path(cli.__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not a source checkout — the declared extras are not on disk")
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    train_extra = declared["project"]["optional-dependencies"]["train"]
    floors = {req.partition(">=")[0].strip(): req.strip() for req in train_extra}

    assert cli._TRAIN_REQS == {m: floors[m] for m in cli._TRAIN_REQS}


class _Ran:
    returncode = 0


def _trainable_project(tmp_path):
    import yaml
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "p", "goal": "g",
        "spec": {"goal": "g",
                 "examples": [{"input": "hi", "good_output": "hello there", "source": "human"}],
                 "eval_criteria": [{"id": "c1", "description": "d", "weight": "high"}]},
    }))


class _NoCuda:
    """Stands in for torch on a machine with no CUDA device."""

    class cuda:
        @staticmethod
        def is_available():
            return False

    class backends:
        pass


def test_qlora_off_a_cuda_gpu_falls_back_instead_of_demanding_bitsandbytes(tmp_path, monkeypatch):
    """`--qlora` is documented, and off CUDA the very next block trains in full
    precision instead. Asking for a CUDA-only package first turns that fallback
    into a dead end — and on a platform with no wheel for it, an unreachable one."""
    import subprocess
    import sys

    from ai_calibrator import cli

    _trainable_project(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", _NoCuda)
    monkeypatch.setattr(cli, "_dep_satisfied", lambda module, requirement: True)
    installed: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: installed.append(list(cmd)) or _Ran())

    r = runner.invoke(app, ["train", str(tmp_path), "--qlora"], input="n\n")

    assert r.exit_code == 0, r.output
    assert "full precision" in r.output
    assert not any("bitsandbytes" in part for cmd in installed for part in cmd)
    assert _has_no_traceback(r)


# --- compile: a retry the message describes has to be the retry that runs ---

class _SynthesisOnlyEngine:
    name = "half@test"

    def complete(self, prompt, *, system=None, schema=None):
        return {}


def test_compile_does_not_promise_a_retry_that_skips_synthesis(tmp_path, monkeypatch):
    """The saved spec is kept, but a re-run synthesizes again and merges the two
    independent results — so the standards and criteria grow with near-duplicates.
    Telling the owner the retry picks up at test generation hides that cost."""
    import ai_calibrator.compile as compile_mod
    import ai_calibrator.engines as engines
    from ai_calibrator.engines.base import EngineOutputError
    from ai_calibrator.models import BehaviorSpec, InterviewItem, Project
    from ai_calibrator.store import save_project

    p = Project(name="p", goal="g")
    p.interview = [InterviewItem(id="q1", dimension="tone", question="?", answer="warm")]
    save_project(p, tmp_path)
    monkeypatch.setattr(engines, "get_engine", lambda spec: _SynthesisOnlyEngine())

    def _fails_at_test_generation(project, engine, **kwargs):
        project.spec = BehaviorSpec(goal="g")     # synthesis had already succeeded
        raise EngineOutputError("could not read the generated tests", raw="{{{")

    monkeypatch.setattr(compile_mod, "compile_project", _fails_at_test_generation)

    r = runner.invoke(app, ["compile", str(tmp_path)])

    assert r.exit_code == 1, r.output
    assert "synthesis runs again" in r.output
    assert _has_no_traceback(r)


# --- examples: the compiled bundle is an artifact, not a snapshot -----------

def test_examples_import_refreshes_the_compiled_bundle(tmp_path):
    """build/ is the documented compiled bundle and every other spec-mutating
    command rewrites it. Left behind, build/spec.yaml is a spec nobody chose —
    reviewed and committed as the one that shipped."""
    import yaml

    from ai_calibrator.compile import write_build_bundle
    from ai_calibrator.models import BehaviorSpec, Example, Project
    from ai_calibrator.store import save_project

    p = Project(name="p", goal="g", spec=BehaviorSpec(goal="g", examples=[
        Example(input="compiler q", good_output="a", source="engine")]))
    save_project(p, tmp_path)
    write_build_bundle(p.spec, p.tests, tmp_path)
    csv = tmp_path / "qa.csv"
    csv.write_text("question,answer\nHuman q?,Human a\n", encoding="utf-8")

    r = runner.invoke(app, ["examples", str(tmp_path), "--import", str(csv)])

    assert r.exit_code == 0, r.output
    built = yaml.safe_load((tmp_path / "build" / "spec.yaml").read_text(encoding="utf-8"))
    assert [e["input"] for e in built["examples"]] == ["compiler q", "Human q?"]


def test_examples_dedup_refreshes_the_compiled_bundle(tmp_path):
    import yaml

    from ai_calibrator.compile import write_build_bundle
    from ai_calibrator.models import BehaviorSpec, Example, Project
    from ai_calibrator.store import save_project

    dupe = [Example(input="q", good_output="a", source="human"),
            Example(input="q", good_output="a", source="human")]
    p = Project(name="p", goal="g", spec=BehaviorSpec(goal="g", examples=dupe))
    save_project(p, tmp_path)
    write_build_bundle(p.spec, p.tests, tmp_path)

    r = runner.invoke(app, ["examples", str(tmp_path), "--dedup"])

    assert r.exit_code == 0, r.output
    built = yaml.safe_load((tmp_path / "build" / "spec.yaml").read_text(encoding="utf-8"))
    assert len(built["examples"]) == 1
