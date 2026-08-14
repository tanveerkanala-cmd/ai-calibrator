"""v1 Advanced tier — dataset assembly, recipe, prove-it gate, generated script."""

import ast
import json
import types

import yaml

from ai_calibrator.finetune import (
    DEFAULT_BASE,
    assemble_dataset,
    beats_baseline,
    export_finetune,
    recommend_recipe,
    render_train_py,
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
            Example(input="Can I return this?", good_output="Yes, within 30 days.",
                    why="cites policy", source="human"),
            Example(input="cure my acne?", source="human",
                    good_output="I can't make medical claims; see a dermatologist."),
            Example(input="no target here", good_output=None, source="human"),  # skipped (no good_output)
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




def test_bundle_install_line_matches_the_trainer_it_ships(tmp_path):
    """The generated train.py calls SFTConfig(max_length=...), which trl renamed
    in 1.x — a printed `trl>=0.12` installs a version that raises on the very
    file the bundle just wrote."""
    export_finetune(_project_with_examples(), project_dir=tmp_path)
    ft = tmp_path / "finetune"
    script = (ft / "train.py").read_text(encoding="utf-8")
    readme = (ft / "README.md").read_text(encoding="utf-8")
    assert "max_length=" in script                      # the argument that needs trl 1.x
    for text in (script, readme):
        assert '"trl>=1.0"' in text and "0.12" not in text




def test_emitted_install_lines_match_the_declared_trl_floor(tmp_path):
    """The bundle told the user to install a trl the emitted code rejects:
    train.py passes SFTConfig(max_length=...), which needs trl>=1.0."""
    export_finetune(_project_with_examples(), project_dir=tmp_path)
    ft = tmp_path / "finetune"
    for fn in ("train.py", "README.md"):
        body = (ft / fn).read_text(encoding="utf-8")
        assert "trl>=1.0" in body and "trl>=0.12" not in body
        assert "pyyaml" in body  # train.py reads recipe.yaml at run time


def test_training_overlap_matches_by_recorded_input_not_current_suite():
    """The memorization check reads the CURRENT suite's input for an id, but the
    card is an OLD run. `compile` re-mints t1..tN, so an id that names a
    memorized training prompt today may have named a genuinely held-out
    question when the run happened — and answering the question about a prompt
    the run never sent either invents an overlap or hides one."""
    from ai_calibrator.finetune import training_overlap
    from ai_calibrator.models import TestCase, test_input_hash

    p = _project_with_examples()
    memorized = TestCase(id="ex_1", input="Can I return this?", expects=["c1"])
    held_out = TestCase(id="ex_1", input="something never trained on", expects=["c1"])
    p.tests = [memorized]

    # A run that graded THIS test really did send a memorized prompt.
    same = Scorecard(run_id="run-0001", results=[ResultRow(
        test_id="ex_1", output="o", input_hash=test_input_hash(memorized),
        criteria=[CriterionResult(criterion_id="c1", passed=True)])])
    assert training_overlap(p, same) == ["ex_1"]

    # A run from before the re-mint asked something else under that id. Reading
    # today's input for it would report an overlap the run never had.
    stale = Scorecard(run_id="run-0000", results=[ResultRow(
        test_id="ex_1", output="o", input_hash=test_input_hash(held_out),
        criteria=[CriterionResult(criterion_id="c1", passed=True)])])
    assert training_overlap(p, stale) == []

    # Back-compat: a pre-hash scorecard records None and still matches by id.
    legacy = Scorecard(run_id="run-0000", results=[ResultRow(
        test_id="ex_1", output="o",
        criteria=[CriterionResult(criterion_id="c1", passed=True)])])
    assert training_overlap(p, legacy) == ["ex_1"]


def test_training_overlap_sees_a_memorized_follow_up_turn():
    """Live feedback on a multi-turn chat trains on the LAST turn (the flywheel
    writes Example(input=turns[-1])) but pins the test on the FIRST
    (TestCase(input=turns[0], follow_ups=turns[1:])). Reading only the opening
    turn can never match, so eval replays the memorized exchange, the judge grades
    the transcript containing it, and the gate counts that test as evidence the
    fine-tune generalizes — the one thing the held-out rate exists to rule out."""
    from ai_calibrator.finetune import training_overlap
    from ai_calibrator.models import TestCase, test_input_hash

    p = Project(name="t", goal="g")
    p.spec = BehaviorSpec(goal="g", examples=[
        Example(input="That did not work", source="human_ratified",
                good_output="Hold the cap down for five seconds, then release."),
    ])
    memorized = TestCase(id="fb_1", input="How do I reset my cap?",
                         follow_ups=["That did not work"], expects=["c1"])
    untrained = TestCase(id="fb_2", input="Where do I buy refills?",
                         follow_ups=["Still stuck"], expects=["c1"])
    p.tests = [memorized, untrained]
    card = Scorecard(run_id="run-0001", results=[
        ResultRow(test_id=t.id, output="o", input_hash=test_input_hash(t),
                  criteria=[CriterionResult(criterion_id="c1", passed=True)])
        for t in (memorized, untrained)])

    assert training_overlap(p, card) == ["fb_1"]


def _from_generated_trainer(names: set[str]) -> dict:
    """Execute named top-level definitions lifted out of the generated train.py.

    The bundle's train.py imports torch/trl/peft at module scope, so the only way
    to exercise the logic a user will actually run is to lift the definitions
    under test out of the emitted source and run those."""
    tree = ast.parse(render_train_py(recommend_recipe(40)))
    body = [n for n in tree.body
            if (isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) in names for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in names)]
    assert len(body) == len(names), f"generated train.py is missing {names}"
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "train.py", "exec"), ns)
    return ns


def test_recipe_learning_rate_in_exponent_form_reaches_the_trainer(tmp_path, monkeypatch):
    """YAML 1.1 resolves `5e-5` (no decimal point) to a STRING — and that is how a
    learning rate is normally written. recipe.yaml is documented as editable, so a
    dropped edit trains at 4x the requested rate and reports nothing."""
    ns = _from_generated_trainer({"DEFAULTS", "_recipe"})
    (tmp_path / "recipe.yaml").write_text(
        "learning_rate: 5e-5\nlora_r: 64\nmax_seq_len: 4096\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = ns["_recipe"]()
    assert float(cfg["learning_rate"]) == 5e-5
    assert int(cfg["lora_r"]) == 64 and int(cfg["max_seq_len"]) == 4096


def test_trainer_trains_in_fp16_on_pre_ampere_cuda():
    """bf16 needs Ampere (compute capability 8.0+). Forcing it on a T4/V100/RTX
    20xx either has TrainingArguments refuse before step 0 or falls back to a
    software emulation slower than fp16 — and neither `--qlora` nor a smaller
    `--base`, the two fixes `calibrate train` suggests, changes the dtype."""
    def _torch(major: int) -> types.SimpleNamespace:
        cuda = types.SimpleNamespace(get_device_capability=lambda: (major, 0))
        return types.SimpleNamespace(cuda=cuda, bfloat16="bf16", float16="fp16")

    ns = _from_generated_trainer({"_cuda_dtype"})
    ns["torch"] = _torch(7)                      # Turing (T4 / RTX 20xx)
    assert ns["_cuda_dtype"]() == "fp16"
    ns["torch"] = _torch(8)                      # Ampere and newer
    assert ns["_cuda_dtype"]() == "bf16"

    src = render_train_py(recommend_recipe(40))
    assert 'bf16=(device == "cuda")' not in src   # capability, not "it has CUDA"
    assert "fp16=" in src


def test_merge_script_loads_the_base_at_the_size_it_saves(tmp_path):
    """merge.py runs on the host that trained the model — the README's QLoRA route
    targets a 10-12 GB card, i.e. a 16-32 GB machine. fp32 materialises the default
    7B at 4 bytes/param (~30 GB of RAM) to save a bf16 copy, so the mandatory merge
    step is OOM-killed on the hardware the bundle documents."""
    export_finetune(_project_with_examples(), project_dir=tmp_path)
    src = (tmp_path / "finetune" / "merge.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "dtype=torch.float32" not in src
    assert "from_pretrained(BASE, dtype=torch.bfloat16)" in src


def test_readme_names_the_extra_transformers_serve_needs(tmp_path):
    """`transformers serve` raises ImportError without fastapi/uvicorn/openai, and
    none of them come with the train extra — so the documented serve-then-gate path
    dead-ends right after a multi-GB train and merge."""
    export_finetune(_project_with_examples(), project_dir=tmp_path)
    readme = (tmp_path / "finetune" / "README.md").read_text(encoding="utf-8")
    assert '"transformers[serving]"' in readme
    assert "nothing else to install" not in readme


def test_readme_ollama_import_is_one_copyable_command(tmp_path):
    """The README is an f-string: an unescaped \\n splits the Modelfile command
    across three lines, so a reader copying it out runs `--experimental` as its own
    command and imports the model without the flag the import needs."""
    export_finetune(_project_with_examples(), project_dir=tmp_path)
    readme = (tmp_path / "finetune" / "README.md").read_text(encoding="utf-8")
    assert "printf 'FROM ./merged\\n' > Modelfile" in readme
    assert "ollama create my-ft -f Modelfile --experimental" in readme


def test_regenerating_the_bundle_keeps_hand_edited_hyperparameters(tmp_path):
    """`calibrate train` rebuilds the bundle before train.py reads it, so an
    unconditional rewrite reverts the knobs USAGE documents as editable — silently,
    and only on the one-command route the CLI recommends."""
    p = _project_with_examples()
    export_finetune(p, project_dir=tmp_path)
    recipe_file = tmp_path / "finetune" / "recipe.yaml"
    edited = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
    edited.update({"learning_rate": "5e-5", "lora_r": 64, "max_seq_len": 4096})
    recipe_file.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")

    export_finetune(p, project_dir=tmp_path, base_model="Qwen/Qwen2.5-3B-Instruct", epochs=2)
    after = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
    assert after["learning_rate"] == "5e-5"
    assert after["lora_r"] == 64 and after["max_seq_len"] == 4096
    # …while this run's flags still win: they are baked into train.py, not read
    # from the file, so a stale value here would describe a run that never happened.
    assert after["base_model"] == "Qwen/Qwen2.5-3B-Instruct" and after["epochs"] == 2
    assert "Qwen/Qwen2.5-3B-Instruct" in (tmp_path / "finetune" / "train.py").read_text(
        encoding="utf-8")


def test_preserved_knobs_are_exactly_what_the_generated_trainer_reads():
    """The two halves of one rule: a key carried across regeneration that train.py
    does not read would look honoured and change nothing, and a key train.py reads
    that is not carried would keep being reverted."""
    from ai_calibrator.finetune import TUNABLE_KEYS

    ns = _from_generated_trainer({"DEFAULTS"})
    assert set(ns["DEFAULTS"]) == set(TUNABLE_KEYS)
