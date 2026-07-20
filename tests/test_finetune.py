"""v1 Advanced tier — dataset assembly, recipe, prove-it gate, generated script."""

import ast
import json

from ai_calibrator.finetune import (
    DEFAULT_BASE,
    assemble_dataset,
    beats_baseline,
    export_finetune,
    recommend_recipe,
)
from ai_calibrator.models import (
    BehaviorSpec,
    CriterionResult,
    Example,
    Project,
    Scorecard,
)
from ai_calibrator.models import TestResult as ResultRow  # aliased: avoid pytest collecting the model


def _project_with_examples():
    p = Project(name="t", goal="answer questions")
    p.spec = BehaviorSpec(
        goal="answer questions",
        standards=["Be concise."],
        examples=[
            Example(input="Can I return this?", good_output="Yes, within 30 days.", why="cites policy"),
            Example(input="cure my acne?", good_output="I can't make medical claims; see a dermatologist."),
            Example(input="no target here", good_output=None),  # skipped (no good_output)
        ],
    )
    return p


def test_assemble_dataset_chat_format():
    rows = assemble_dataset(_project_with_examples())
    assert len(rows) == 2  # the example without good_output is skipped
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == "Can I return this?"
    assert msgs[2]["content"] == "Yes, within 30 days."
    assert "answer questions" in msgs[0]["content"]  # system prompt embedded


def test_recommend_recipe_defaults():
    r = recommend_recipe(10)
    assert r["method"] == "lora" and r["base_model"] == DEFAULT_BASE
    assert r["epochs"] == 5  # small dataset → more epochs
    assert recommend_recipe(100)["epochs"] == 3


def test_export_writes_bundle_and_valid_script(tmp_path):
    result = export_finetune(_project_with_examples(), project_dir=tmp_path, base_model="meta-llama/Llama-3.1-8B-Instruct")
    ft = tmp_path / "finetune"
    for fn in ["dataset.jsonl", "recipe.yaml", "train.py", "README.md"]:
        assert (ft / fn).exists(), fn
    assert result.examples == 2

    lines = [ln for ln in (ft / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2 and "messages" in json.loads(lines[0])

    src = (ft / "train.py").read_text(encoding="utf-8")
    assert "meta-llama/Llama-3.1-8B-Instruct" in src
    ast.parse(src)  # the generated training script is valid Python


def _card(passes):
    return Scorecard(run_id="r", results=[
        ResultRow(test_id=f"t{i}", output="x",
                  criteria=[CriterionResult(criterion_id="c", passed=p)])
        for i, p in enumerate(passes)
    ])


def test_prove_it_gate():
    assert beats_baseline(_card([True, False]), _card([True, True]))      # 50% → 100%
    assert not beats_baseline(_card([True, True]), _card([True, False]))  # 100% → 50%
    assert not beats_baseline(_card([True, False]), _card([True, False]))  # equal, no margin
    assert not beats_baseline(_card([True, True]), _card([True, True]), margin=0.01)  # tie loses with margin
