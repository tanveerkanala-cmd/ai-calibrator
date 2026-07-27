"""v1 — Advanced tier: the fine-tuning toolchain (opt-in, technical users).

Assembles a fine-tuning dataset from the spec's examples (and human corrections,
when captured), emits a runnable LoRA recipe + training script, and provides the
**prove-it gate**: a fine-tune is only worth keeping if it BEATS the configured
prompt+RAG baseline on the same eval harness. Non-technical users never see this.

The dataset should be dominated by human-authored/corrected outputs: a model
writing its own training targets teaches it nothing new. Note that
`spec.examples` also holds compiler-synthesized rows and `Example` carries no
provenance field, so this is guidance the code cannot yet enforce — the gate
(`beats_baseline`) is the safeguard that the result actually helps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .coerce import safe_token
from .compile import render_system_prompt
from .models import Project, Scorecard
from .store import atomic_write_text

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"  # an open base you can actually LoRA

_TRAIN_PY = '''#!/usr/bin/env python3
"""LoRA fine-tune from dataset.jsonl. Runs on CUDA, Apple Silicon (MPS), or CPU.

    pip install "transformers>=4.46" "trl>=0.12" peft datasets accelerate
    python train.py            # writes the LoRA adapter to ./__OUT__/

On a memory-limited CUDA GPU: `pip install bitsandbytes` and set QLORA=1 to load
the base in 4-bit (CUDA only -- not supported on Apple Silicon). On MPS, if an op
is unimplemented, export PYTORCH_ENABLE_MPS_FALLBACK=1."""
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = "__BASE__"
OUT = "__OUT__"

# Hyperparameters live in recipe.yaml so editing that file actually changes the
# run (it is documented as editable). Values here are the fallback when the file
# is missing or unreadable.
DEFAULTS = {"learning_rate": 2e-4, "lora_r": 16, "lora_alpha": 32,
            "lora_dropout": 0.05, "max_seq_len": 2048}


def _recipe() -> dict:
    cfg = dict(DEFAULTS)
    try:
        import yaml
        with open("recipe.yaml", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for k in DEFAULTS:
            if isinstance(loaded.get(k), (int, float)):
                cfg[k] = loaded[k]
    except Exception:
        pass
    return cfg


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    with open("dataset.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit("dataset.jsonl is empty -- add training examples first.")

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    dataset = Dataset.from_list([{"text": t} for t in texts])

    qlora = bool(os.getenv("QLORA")) and device == "cuda"
    if qlora:                                          # 4-bit -- CUDA + bitsandbytes only
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
    else:
        # fp32 off-CUDA (MPS/CPU are unstable in half precision for training)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype).to(device)

    cfg = _recipe()
    config = SFTConfig(
        output_dir=OUT,
        num_train_epochs=__EPOCHS__,
        max_steps=__MAX_STEPS__,   # -1 = unbounded (epochs decide); >0 caps total steps
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=5,
        dataset_text_field="text",
        max_length=int(cfg["max_seq_len"]),
        bf16=(device == "cuda"),
    )
    peft_config = LoraConfig(r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
                             lora_dropout=float(cfg["lora_dropout"]), task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, peft_config=peft_config)
    trainer.train()
    trainer.save_model(OUT)
    print(f"saved LoRA adapter to {OUT}/ (trained on {device})")


if __name__ == "__main__":
    main()
'''


_MERGE_PY = '''#!/usr/bin/env python3
"""Merge the trained LoRA adapter into the base weights → ./__MERGE_OUT__/.

    pip install "transformers>=4.46" peft torch
    python merge.py

Then serve the merged model (see README.md) and point the project's `subject`
engine at it to run the prove-it gate."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "__BASE__"
ADAPTER = "__OUT__"
MERGE_OUT = "__MERGE_OUT__"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    merged.save_pretrained(MERGE_OUT)
    tokenizer.save_pretrained(MERGE_OUT)
    print(f"merged model saved to {MERGE_OUT}/")


if __name__ == "__main__":
    main()
'''

MERGE_OUT = "merged"

# Every placeholder the trainer template carries. Rendering must substitute ALL of
# them: a leftover `__EPOCHS__` is a valid Python identifier, so the emitted file
# still parses (an ast.parse check passes) and only fails with NameError at run
# time — after the multi-GB base model has been downloaded.
_TRAIN_PLACEHOLDERS = ("__BASE__", "__OUT__", "__EPOCHS__", "__MAX_STEPS__")


def render_train_py(recipe: dict) -> str:
    """Render the trainer template for ``recipe`` — the ONE place that substitutes.

    Both bundle writers (``export_finetune`` and ``train_engine.export_engine_bundle``)
    go through here so neither can drift from the template's placeholder set."""
    src = (_TRAIN_PY
           .replace("__BASE__", str(recipe["base_model"]))
           .replace("__OUT__", str(recipe["output_dir"]))
           .replace("__EPOCHS__", str(int(recipe["epochs"])))
           .replace("__MAX_STEPS__", str(int(recipe["max_steps"]))))
    left = [p for p in _TRAIN_PLACEHOLDERS if p in src]
    if left:  # a template edit added a placeholder this function doesn't know
        raise ValueError(f"train.py template not fully rendered — left: {', '.join(left)}")
    return src


@dataclass
class FinetuneResult:
    examples: int
    base_model: str
    method: str
    bundle_dir: str
    files: list[str]


def assemble_dataset(project: Project) -> list[dict]:
    """Chat-format SFT rows from the spec's good examples.

    IMPORTANT — provenance. ``spec.examples`` is a mixed bag: rows imported by the
    owner, rows ratified through `teach`, corrections absorbed from live feedback,
    AND rows the compiler engine invented during synthesis. `Example` carries no
    provenance field, so this cannot filter to human-authored rows only, which
    means a fine-tune trained here CAN include model-written targets (partial
    self-distillation). Prefer a dataset dominated by imported or corrected
    examples; see docs/USAGE.md. The eval-loop corrections are the signal that
    actually teaches something new."""
    spec = project.spec
    if spec is None:
        raise ValueError("No spec — run `calibrate compile` first.")
    system = render_system_prompt(spec)
    rows: list[dict] = []
    for ex in spec.examples:
        if ex.input and ex.good_output:
            rows.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": ex.input},
                    {"role": "assistant", "content": ex.good_output},
                ]
            })
    return rows


def recommend_recipe(
    n_examples: int, *, base_model: str | None = None,
    epochs: int | None = None, max_steps: int | None = None,
) -> dict:
    # base_model comes from `--base` (or a hand-edited binding) and gets baked
    # into the generated, later-executed train.py — validate it can't inject.
    base = safe_token(base_model or DEFAULT_BASE, "base model")
    return {
        "method": "lora",
        "base_model": base,
        # epochs (and the optional step cap) are BAKED into the generated
        # train.py, so editing them here actually changes training — an override
        # from `--epochs` / `--max-steps` wins over the size-based default.
        "epochs": epochs if epochs is not None else (3 if n_examples >= 50 else 5),
        "max_steps": max_steps if max_steps is not None else -1,  # -1 = epochs decide
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_seq_len": 2048,
        "output_dir": "adapter",
        "hardware": "Runs on CUDA/MPS/CPU (auto). fp16 LoRA of a 7B ~16GB VRAM; "
                    "below that use QLORA=1 (4-bit, CUDA only) or a smaller --base.",
    }


def beats_baseline(baseline: Scorecard, candidate: Scorecard, margin: float = 0.0) -> bool:
    """The prove-it gate: keep the fine-tune only if it beats the baseline."""
    return candidate.pass_rate > baseline.pass_rate + margin


def training_overlap(project: Project, card: Scorecard) -> list[str]:
    """Test ids in ``card`` whose input was also a TRAINING prompt.

    The dataset is built from ``spec.examples`` and `examples-to-tests` /
    `absorb` turn those same examples into ``ex_*`` / ``fb_*`` tests, so the gate
    can end up grading a model on prompts it memorized. That is not a held-out
    comparison, and a memorizing fine-tune would pass it. Callers report the
    overlap so the number is read for what it is."""
    trained = {
        ex.input for ex in (project.spec.examples if project.spec else [])
        if ex.input and ex.good_output
    }
    by_id = {t.id: t.input for t in project.tests}
    return sorted(r.test_id for r in card.results
                  if by_id.get(r.test_id) in trained)


def _readme(recipe: dict, n: int) -> str:
    base = recipe["base_model"]
    return f"""# Advanced tier — fine-tune `{base}`

Opt-in step-up. Your behavior is already captured in the spec + system prompt;
fine-tuning bakes it into the weights. **Only worth keeping if it beats your
configured baseline on the same evals** — that's the gate below.

Dataset: **{n} example(s)** assembled from your spec's good examples. More (and
human-corrected) examples → a better fine-tune.

## 1. Train (runs on CUDA, Apple Silicon/MPS, or CPU — auto-detected)
```bash
pip install "transformers>=4.46" "trl>=0.12" peft datasets accelerate
python train.py        # writes the LoRA adapter to ./{recipe["output_dir"]}/
```
Fitting the base to your hardware: a fp16 LoRA of a 7B wants ~16GB VRAM. Below
that, either (a) `pip install bitsandbytes` and `QLORA=1 python train.py` to load
the 7B in 4-bit (CUDA only — fits a 10-12GB card), or (b) regenerate with a
smaller base: `calibrate finetune --base Qwen/Qwen2.5-3B-Instruct` (or `-1.5B-`).
On Apple Silicon it trains on the MPS GPU (fp32; no bitsandbytes) — a 0.5–3B base
fits comfortably; the 7B needs ~24GB+ unified memory.

## 2. Serve the fine-tune (pick one)
The adapter in `./{recipe["output_dir"]}/` is a LoRA delta — merge it into the
base first:
```bash
python merge.py            # writes the merged model to ./{MERGE_OUT}/
```
Then either:
- **Ollama:** convert to GGUF and `ollama create my-ft -f Modelfile` (with
  `FROM ./{MERGE_OUT}`), then bind `calibrate engines <project> subject my-ft@ollama`.
- **OpenAI-compatible endpoint:** `transformers serve ./{MERGE_OUT}` (or vLLM),
  then `export OPENAI_BASE_URL=http://localhost:8000/v1` and bind
  `calibrate engines <project> subject ./{MERGE_OUT}@openai`.

## 3. Prove it beats the baseline
1. With the fine-tune served + bound as `subject`, run `calibrate eval`.
2. Compare to your pre-fine-tune baseline scorecard:
   ```bash
   calibrate finetune --gate --baseline <baseline-run> --candidate <new-run>
   ```
   Keep the fine-tune **only if the gate says it wins.** Otherwise stay on the
   configured prompt+RAG — it's cheaper and already good enough.

> Never copy the model's own output into the dataset as the "ideal" answer.
> Corrections must reflect *your* judgment, or the fine-tune learns nothing new.
"""


def export_finetune(
    project: Project, *, project_dir, base_model: str | None = None,
    epochs: int | None = None, max_steps: int | None = None,
) -> FinetuneResult:
    rows = assemble_dataset(project)
    recipe = recommend_recipe(len(rows), base_model=base_model, epochs=epochs, max_steps=max_steps)

    out = Path(project_dir) / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def _write(fn: str, content: str) -> None:
        atomic_write_text(out / fn, content)
        files.append(f"finetune/{fn}")

    train_py = render_train_py(recipe)
    merge_py = (_MERGE_PY
                .replace("__BASE__", recipe["base_model"])
                .replace("__OUT__", recipe["output_dir"])
                .replace("__MERGE_OUT__", MERGE_OUT))

    _write("dataset.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _write("recipe.yaml", yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
    _write("train.py", train_py)
    _write("merge.py", merge_py)
    _write("README.md", _readme(recipe, len(rows)))

    return FinetuneResult(
        examples=len(rows), base_model=recipe["base_model"],
        method=recipe["method"], bundle_dir=str(out), files=files,
    )
