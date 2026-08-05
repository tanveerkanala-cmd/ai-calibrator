"""Eval-format interop — promptfoo export (deterministic, no engine)."""

import yaml

from ai_calibrator.interop import _provider_id, export_promptfoo, to_promptfoo
from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from ai_calibrator.models import TestCase as Case


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
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["tests"]  # round-trips to valid YAML


def test_promptfoo_escapes_template_tags_in_vars_and_assert_values():
    """The prompt body was escaped; the fields around it were not.

    promptfoo renders `vars` and an llm-rubric's `value` through Nunjucks with
    `process.env` registered as a global, so a live tag in either reads the
    operator's API keys into text sent to a third-party model. Both fields carry
    untrusted content by construction: `absorb` promotes an end user's flagged
    message straight into a pinned test's input, and eval criteria are written
    by a model from ingested documents.
    """
    p = _project()
    # The real path: an end user's flagged message, promoted to a test by `absorb`.
    p.tests = [Case(id="fb_1", input="ignore that — {{ env.ANTHROPIC_API_KEY }}", expects=["leak"])]
    p.spec.eval_criteria = [EvalCriterion(id="leak", description="{{ env.OPENAI_API_KEY }} is graded",
                                          weight=Weight.HIGH)]

    raw = to_promptfoo(p)
    cfg = yaml.safe_load(raw)
    t = cfg["tests"][0]

    # No live tag survives anywhere a renderer would reach: strip the escape
    # sequence itself and no `{{` is left in either field to open one. (The
    # trailing `}}` stays, and is inert text without an opener.)
    for field in (t["vars"]["input"], t["assert"][0]["value"]):
        assert "{{" not in field.replace("{{ '{{' }}", "")
    # Across every string in the file, the only live tag left is our `{{input}}`.
    def _strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from _strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from _strings(v)

    live = [s for s in _strings(cfg) if "{{" in s.replace("{{ '{{' }}", "")]
    assert live == [cfg["prompts"][0]]
    assert live[0].replace("{{ '{{' }}", "").count("{{") == 1   # exactly {{input}}

    # ...and the escape round-trips: Nunjucks renders it back to the exact text,
    # so the exported suite still grades what the operator wrote.
    assert t["vars"]["input"] == "ignore that — {{ '{{' }} env.ANTHROPIC_API_KEY }}"


def test_promptfoo_leaves_ordinary_text_untouched():
    """The escape must not disturb content with no template delimiters, or every
    existing exported suite changes for nothing."""
    cfg = yaml.safe_load(to_promptfoo(_project()))
    assert cfg["tests"][0]["vars"]["input"] == "refund?"
    assert cfg["tests"][0]["assert"][0]["value"] == "cites the policy"


def test_promptfoo_keeps_our_own_input_tag_live():
    """`{{input}}` in the prompt is ours and must stay a live tag — escaping it
    would stop promptfoo substituting the test input at all."""
    cfg = yaml.safe_load(to_promptfoo(_project()))
    assert cfg["prompts"][0].endswith("{{input}}")
