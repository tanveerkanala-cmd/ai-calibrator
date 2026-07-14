"""Regression tests for the environment/portability audit fixes."""

import os
import subprocess
import sys

import pytest


# encoding class: a full non-ASCII round-trip must work under a NON-UTF8 locale
# (the Windows cp1252 / LC_ALL=C proxy). Runs in a subprocess so the locale is
# actually applied to a fresh interpreter.
def test_non_ascii_roundtrip_under_c_locale(tmp_path):
    script = (
        "import sys, pathlib\n"
        "d = pathlib.Path(sys.argv[1])\n"
        "from calibrator.models import Project, BehaviorSpec, EvalCriterion, Weight, TestCase\n"
        "from calibrator.store import save_project, load_project\n"
        "from calibrator.snapshot import save_golden, load_golden\n"
        "p = Project(name='café-测试', goal='回答问题 ☕🎯')\n"
        "p.spec = BehaviorSpec(goal=p.goal, standards=['用中文回答 ☕'],\n"
        "    eval_criteria=[EvalCriterion(id='c1', description='准确 accurate', weight=Weight.HIGH)])\n"
        "p.tests = [TestCase(id='t1', input='退货政策?', expects=['c1'])]\n"
        "save_project(p, d)\n"
        "q = load_project(d)\n"
        "assert q.name == 'café-测试' and q.spec.standards == ['用中文回答 ☕'], q.name\n"
        "save_golden(d, {'t1': '退款 refund ✓'})\n"
        "assert load_golden(d) == {'t1': '退款 refund ✓'}\n"
        "print('OK')\n"
    )
    # Write the (non-ASCII) script to a UTF-8 FILE and run the file — passing it via
    # `python -c` fails on Linux under LC_ALL=C because the OS can't encode a
    # non-ASCII argv (works on macOS, which uses UTF-8 for argv). The file path is
    # ASCII; Python reads the source as UTF-8 (PEP 3120) regardless of locale.
    script_file = tmp_path / "roundtrip.py"
    script_file.write_text(script, encoding="utf-8")
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0"}
    r = subprocess.run([sys.executable, str(script_file), str(tmp_path)],
                       capture_output=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "OK" in r.stdout


def test_project_name_rejects_path_unsafe_chars():
    from pydantic import ValidationError

    from calibrator.models import Project
    for bad in ["a/b", "a\\b", "c:name", "a*b", "a?b", 'a"b', "a<b", "a|b", ".", ".."]:
        with pytest.raises(ValidationError):
            Project(name=bad, goal="g")
    # normal names still fine
    assert Project(name="my-support-ai", goal="g").name == "my-support-ai"
    assert Project(name="Café Bot 2", goal="g").name == "Café Bot 2"   # spaces/accents OK


def test_ollama_timeout_rejects_non_finite(monkeypatch):
    from calibrator.engines.ollama import DEFAULT_TIMEOUT, _default_timeout
    for junk in ["inf", "-inf", "nan", "1e999", "-5", "0", "abc"]:
        monkeypatch.setenv("CALIBRATOR_OLLAMA_TIMEOUT", junk)
        assert _default_timeout() == DEFAULT_TIMEOUT, junk    # falls back, no inf/nan
    monkeypatch.setenv("CALIBRATOR_OLLAMA_TIMEOUT", "300")
    assert _default_timeout() == 300.0                        # a valid value still works


def test_gitignore_cleanup_on_non_oserror(tmp_path, monkeypatch):
    import calibrator.store as store

    def boom(*a, **k):
        raise RuntimeError("not an OSError")
    monkeypatch.setattr(store.os, "fdopen", boom)
    with pytest.raises(RuntimeError):
        store.write_project_gitignore(tmp_path)
    assert not (tmp_path / ".gitignore").exists()             # 0-byte stub removed


def test_project_name_rejects_windows_reserved(tmp_path):
    from pydantic import ValidationError

    from calibrator.models import Project
    for bad in ["CON", "con", "PRN", "nul", "COM1", "LPT9", "aux.txt", "NUL.log", "name."]:
        with pytest.raises(ValidationError):
            Project(name=bad, goal="g")
    assert Project(name="name ", goal="g").name == "name"   # trailing space is stripped, not rejected
    # names that merely CONTAIN a reserved word are fine
    assert Project(name="console", goal="g").name == "console"
    assert Project(name="my-con-figs", goal="g").name == "my-con-figs"


def test_api_returns_clean_error_not_500_on_corrupt_scorecard(tmp_path):
    """A corrupt scorecard must yield a clean 4xx from the API, never a raw 500."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from calibrator.api import create_app
    from calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from calibrator.store import save_project

    p = Project(name="apibot", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="desc long enough", weight=Weight.HIGH)])
    save_project(p, tmp_path / "apibot")
    ev = tmp_path / "apibot" / "evals" / "run-0001"
    ev.mkdir(parents=True)
    (ev / "scorecard.json").write_text("{ corrupt not json")
    c = TestClient(create_app(tmp_path))
    for path in ("/api/projects/apibot/judge-check", "/api/projects/apibot/snapshot"):
        r = c.get(path)
        assert r.status_code < 500, f"{path} → {r.status_code} (must not 500 on a corrupt scorecard)"
