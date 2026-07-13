"""CLI robustness — friendly errors instead of raw tracebacks, port validation.

Regression tests for the stress-found CLI crashes: unhandled YAMLError /
ValidationError in _load across every project command, and unvalidated --port
crashing uvicorn with an OverflowError after printing a green "success" line.
"""

from __future__ import annotations

from typer.testing import CliRunner

from calibrator.cli import app

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
    # Sanity: the command still parses and the option is documented.
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output


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
    """init must protect a future `git init` (audit: no .gitignore template) and must
    not claim one engine covers 'all roles' when judge/subject differ (audit)."""
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
    """Audit: `calibrate --help` crashed with UnicodeEncodeError under limited
    encodings (Rich rendering the → glyph). Glyphs must degrade, never crash."""
    import os
    import subprocess
    import sys

    for enc in ("ascii", "cp1252"):
        env = {**os.environ, "PYTHONIOENCODING": enc}
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.argv=['calibrate','--help']; from calibrator.cli import main; main()"],
            env=env, capture_output=True, timeout=60)  # bytes: the child writes in `enc`
        stderr = r.stderr.decode(enc, errors="replace")
        assert r.returncode == 0, f"{enc}: {stderr[-300:]}"
        assert "UnicodeEncodeError" not in stderr


def test_snapshot_on_corrupt_scorecard_is_friendly(tmp_path):
    """Audit #5: snapshot loaded the scorecard with no error handling."""
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
    assert "torch" in r.output and "ai-calibrator[train]" in r.output   # guided, not silent
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
    from calibrator.store import load_project
    assert len(load_project(tmp_path).spec.examples) == 2
