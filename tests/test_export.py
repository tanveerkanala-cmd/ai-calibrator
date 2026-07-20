"""M5 export — fully deterministic, no engine needed."""

import ast

from ai_calibrator.export import DEFAULT_LOCAL_BASE, export_bundle
from ai_calibrator.models import (
    BehaviorSpec,
    EngineBinding,
    EvalCriterion,
    Project,
    Weight,
)
from ai_calibrator.models import TestCase as CaseModel


def _project(subject="claude-sonnet-4-6@anthropic"):
    p = Project(
        name="My Support AI",
        goal="Answer product questions in our brand voice.",
        engines=EngineBinding(subject=subject),
    )
    p.spec = BehaviorSpec(
        goal=p.goal,
        standards=["Be warm and concise."],
        eval_criteria=[EvalCriterion(id="c1", description="on-policy", weight=Weight.HIGH)],
    )
    p.tests = [CaseModel(id="t1", input="hi", expects=["c1"])]
    return p


def test_export_writes_full_bundle(tmp_path):
    result = export_bundle(_project(), project_dir=tmp_path)
    exp = tmp_path / "export"
    for fn in ["system_prompt.txt", "spec.yaml", "rubric.yaml", "rag.config.yaml",
               "tests.jsonl", "Modelfile", "run.py", "README.md"]:
        assert (exp / fn).exists(), fn
    assert result.name == "my-support-ai"

    modelfile = (exp / "Modelfile").read_text(encoding="utf-8")
    assert "FROM" in modelfile and "SYSTEM" in modelfile
    assert "Answer product questions" in modelfile  # system prompt embedded


def test_cloud_subject_falls_back_to_local_base(tmp_path):
    result = export_bundle(_project("gpt-4o@openai"), project_dir=tmp_path)
    assert result.base_model == DEFAULT_LOCAL_BASE
    assert f"FROM {DEFAULT_LOCAL_BASE}" in (tmp_path / "export" / "Modelfile").read_text(encoding="utf-8")


def test_ollama_subject_uses_that_model(tmp_path):
    result = export_bundle(_project("llama3.1:8b@ollama"), project_dir=tmp_path)
    assert result.base_model == "llama3.1:8b"
    assert "FROM llama3.1:8b" in (tmp_path / "export" / "Modelfile").read_text(encoding="utf-8")


def test_generated_runner_is_valid_python_with_base(tmp_path):
    export_bundle(_project("llama3.1:8b@ollama"), project_dir=tmp_path)
    src = (tmp_path / "export" / "run.py").read_text(encoding="utf-8")
    assert "llama3.1:8b" in src
    ast.parse(src)  # the generated runner is syntactically valid Python
