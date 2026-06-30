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
    for bad in ("-1", "70000", "99999", "65536"):
        result = runner.invoke(app, ["serve", f"--port={bad}"])
        assert result.exit_code == 1, f"port {bad} should be rejected"
        assert "between 0 and 65535" in result.output
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
