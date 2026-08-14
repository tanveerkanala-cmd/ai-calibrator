"""M5 export — fully deterministic, no engine needed."""

import ast
import importlib.util
import json
import sys
import urllib.request

from ai_calibrator.eval import conversation_prompt
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


def _run_bundle_runner(tmp_path, question, monkeypatch):
    """Execute the generated run.py against a stubbed Ollama → the posted payload."""
    sent = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"message": {"content": "an answer"}}).encode("utf-8")

    def _urlopen(request, *args, **kwargs):
        sent["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(sys, "argv", ["run.py", question])
    path = tmp_path / "export" / "run.py"
    spec = importlib.util.spec_from_file_location("bundle_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    return sent["payload"]


def test_generated_runner_asks_the_question_the_way_the_eval_graded_it(tmp_path, monkeypatch):
    """`calibrate eval` grades `User: …\\nAssistant:` (eval.conversation_prompt), and
    the served endpoint sends the same encoding. A bundle that asks the bare question
    is asking something the pass rate stamped beside it was never earned on."""
    export_bundle(_project("llama3.1:8b@ollama"), project_dir=tmp_path)

    payload = _run_bundle_runner(tmp_path, "How do I return an item?", monkeypatch)

    system, user = payload["messages"]
    assert system["role"] == "system" and "Answer product questions" in system["content"]
    assert user["role"] == "user"
    assert user["content"] == conversation_prompt([], "How do I return an item?")


def test_bundle_says_where_the_interactive_path_diverges(tmp_path):
    """`ollama run` sends the bare question — this format cannot carry the
    encoding the eval graded, so the bundle says so instead of implying the
    certificate covers it."""
    export_bundle(_project("llama3.1:8b@ollama"), project_dir=tmp_path)
    exp = tmp_path / "export"

    readme = (exp / "README.md").read_text(encoding="utf-8")
    modelfile = (exp / "Modelfile").read_text(encoding="utf-8")

    assert "bare question" in readme and "run.py" in readme
    assert "bare question" in modelfile
