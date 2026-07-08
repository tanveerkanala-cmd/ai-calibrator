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
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0"}
    r = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                       capture_output=True, text=True, env=env)
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
