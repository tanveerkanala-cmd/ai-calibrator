"""Generated files (train.py/run.py/Modelfile) must never carry injected code.

Audit round 18: base_model / subject-model tokens were string-replaced into
templates unescaped, so a crafted value could break out of a Python string
literal and execute during training. safe_token now gates them."""

import ast

import pytest

from calibrator.coerce import safe_token
from calibrator.models import BehaviorSpec, EvalCriterion, Project, TestCase, Weight


def test_safe_token_accepts_real_ids_rejects_injection():
    for good in ["Qwen/Qwen2.5-7B-Instruct", "gemma4:e4b", "mistralai/Mistral-7B-Instruct-v0.3", "adapter"]:
        assert safe_token(good, "x") == good
    for bad in ['m"; import os; os.system("id"); x="', "a\nimport os", "a`id`", "a$(id)", "a b", "a'b", "", None]:
        with pytest.raises(ValueError):
            safe_token(bad, "base model")


def test_finetune_rejects_malicious_base(tmp_path):
    from calibrator.finetune import export_finetune
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", examples=[])
    with pytest.raises(ValueError, match="model/path token"):
        export_finetune(p, project_dir=tmp_path, base_model='x"; import os; os.system("id"); y="')


def test_export_run_py_is_valid_python_and_uninjected(tmp_path):
    """A normal export produces a parseable run.py with the model as a plain literal."""
    from calibrator.export import export_bundle

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c", description="d", weight=Weight.HIGH)])
    p.tests = [TestCase(id="t1", input="q", expects=["c"])]
    p.engines.subject = "gemma4:e4b@ollama"
    export_bundle(p, project_dir=tmp_path)
    run_py = (tmp_path / "export" / "run.py").read_text(encoding="utf-8")
    ast.parse(run_py)                                  # parses
    assert 'gemma4:e4b' in run_py and "import os" not in run_py.split("gemma4")[1][:40]


def test_export_rejects_malicious_subject_binding(tmp_path):
    from calibrator.export import export_bundle
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c", description="d", weight=Weight.HIGH)])
    p.engines.subject = 'evil"; import os#@ollama'      # breakout attempt in the model part
    with pytest.raises(ValueError, match="model/path token"):
        export_bundle(p, project_dir=tmp_path)


def test_train_engine_rejects_traversal_role(tmp_path):
    from calibrator.train_engine import export_engine_bundle
    with pytest.raises(ValueError, match="role must be one of"):
        export_engine_bundle(tmp_path, "../../etc/x")
