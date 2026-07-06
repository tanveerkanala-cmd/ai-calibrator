"""`calibrate engines` — show and set role bindings."""

from typer.testing import CliRunner

from calibrator.cli import app
from calibrator.store import load_project

runner = CliRunner()


def _init(tmp_path):
    d = tmp_path / "p"
    assert runner.invoke(app, ["init", "p", "--goal", "g", "--path", str(d)]).exit_code == 0
    return d


def test_engines_show(tmp_path):
    d = _init(tmp_path)
    r = runner.invoke(app, ["engines", str(d)])
    assert r.exit_code == 0 and "subject" in r.output and "@anthropic" in r.output


def test_engines_set_one_role(tmp_path):
    d = _init(tmp_path)
    r = runner.invoke(app, ["engines", str(d), "subject", "gpt-4o-mini@openai"])
    assert r.exit_code == 0 and "Rebound subject" in r.output
    p = load_project(d)
    assert p.engines.subject == "gpt-4o-mini@openai"
    assert p.engines.judge != "gpt-4o-mini@openai"   # only that role changed


def test_engines_set_all(tmp_path):
    d = _init(tmp_path)
    r = runner.invoke(app, ["engines", str(d), "--all", "gemma4:e4b@ollama"])
    assert r.exit_code == 0 and "all roles" in r.output
    p = load_project(d)
    assert all(v == "gemma4:e4b@ollama" for v in p.engines.model_dump().values())


def test_engines_validation(tmp_path):
    d = _init(tmp_path)
    # bad provider
    r = runner.invoke(app, ["engines", str(d), "subject", "gpt-4o@bogus"])
    assert r.exit_code == 1 and "Valid providers" in r.output
    # unknown role
    r2 = runner.invoke(app, ["engines", str(d), "wizard", "gpt-4o@openai"])
    assert r2.exit_code == 1 and "Unknown role" in r2.output
    # role without model
    r3 = runner.invoke(app, ["engines", str(d), "subject"])
    assert r3.exit_code == 1 and "BOTH" in r3.output
    # empty model name
    r4 = runner.invoke(app, ["engines", str(d), "subject", "@openai"])
    assert r4.exit_code == 1 and "no model name" in r4.output
