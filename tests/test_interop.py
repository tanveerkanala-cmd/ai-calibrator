"""Eval-format interop — promptfoo export (deterministic, no engine)."""

import yaml

from calibrator.interop import _provider_id, export_promptfoo, to_promptfoo
from calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from calibrator.models import TestCase as Case


def _project():
    p = Project(name="p", goal="answer support questions")
    p.spec = BehaviorSpec(goal="answer support questions", standards=["Be concise."],
                          eval_criteria=[EvalCriterion(id="cite", description="cites the policy", weight=Weight.HIGH),
                                         EvalCriterion(id="tone", description="is friendly", weight=Weight.LOW)])
    p.tests = [Case(id="t1", input="refund?", expects=["cite"]),
               Case(id="t2", input="hello", expects=[])]  # empty expects → grade against all criteria
    return p


def test_to_promptfoo_is_valid_yaml_and_structured():
    cfg = yaml.safe_load(to_promptfoo(_project()))
    assert cfg["description"] == "answer support questions"
    assert "{{input}}" in cfg["prompts"][0] and "Be concise" in cfg["prompts"][0]
    assert len(cfg["tests"]) == 2

    t1 = cfg["tests"][0]
    assert t1["vars"]["input"] == "refund?"
    assert [a["value"] for a in t1["assert"]] == ["cites the policy"]   # only the targeted criterion
    assert all(a["type"] == "llm-rubric" for a in t1["assert"])
    assert len(cfg["tests"][1]["assert"]) == 2                          # empty expects → all criteria


def test_provider_id_mapping():
    assert _provider_id("gpt-4o@openai") == "openai:chat:gpt-4o"
    assert _provider_id("claude-opus-4-8@anthropic") == "anthropic:messages:claude-opus-4-8"
    assert _provider_id("gemma4:e4b@ollama") == "ollama:chat:gemma4:e4b"


def test_export_promptfoo_writes_file(tmp_path):
    out = export_promptfoo(_project(), project_dir=tmp_path)
    assert out.exists() and out.name == "promptfooconfig.yaml"
    assert yaml.safe_load(out.read_text())["tests"]  # round-trips to valid YAML
