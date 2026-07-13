"""v1 — Advanced tier: the fine-tuning toolchain (opt-in, technical users).

Assembles a fine-tuning dataset from the spec's examples (and human corrections,
when captured), emits a runnable LoRA recipe + training script, and provides the
**prove-it gate**: a fine-tune is only worth keeping if it BEATS the configured
prompt+RAG baseline on the same eval harness. Non-technical users never see this.

The dataset must come from human-authored/corrected outputs — never self-distill
(a model writing its own training targets teaches it nothing new). The gate
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

    config = SFTConfig(
        output_dir=OUT,
        num_train_epochs=5,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=5,
        dataset_text_field="text",
        max_length=2048,
        bf16=(device == "cuda"),
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, peft_config=peft_config)
    trainer.train()
    trainer.save_model(OUT)
    print(f"saved LoRA adapter to {OUT}/ (trained on {device})")


if __name__ == "__main__":
    main()
'''


@dataclass
class FinetuneResult:
    examples: int
    base_model: str
    method: str
    bundle_dir: str
    files: list[str]


def assemble_dataset(project: Project) -> list[dict]:
    """Chat-format SFT rows from the spec's good examples.

    (Future: also append the human-corrected outputs captured during eval — those
    are the highest-value signal. We never use the model's own passing outputs as
    targets, which would be self-distillation.)
    """
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


def recommend_recipe(n_examples: int, *, base_model: str | None = None) -> dict:
    # base_model comes from `--base` (or a hand-edited binding) and gets baked
    # into the generated, later-executed train.py — validate it can't inject.
    base = safe_token(base_model or DEFAULT_BASE, "base model")
    return {
        "method": "lora",
        "base_model": base,
        "epochs": 3 if n_examples >= 50 else 5,  # tiny sets → more passes
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

## 2. Prove it beats the baseline
1. Serve the fine-tuned model (merge + `ollama create`, or an endpoint).
2. Point the project's `subject` engine at it and run `calibrate eval`.
3. Compare to your pre-fine-tune baseline scorecard:
   ```bash
   calibrate finetune --gate --baseline <baseline-run> --candidate <new-run>
   ```
   Keep the fine-tune **only if the gate says it wins.** Otherwise stay on the
   configured prompt+RAG — it's cheaper and already good enough.

> Never copy the model's own output into the dataset as the "ideal" answer.
> Corrections must reflect *your* judgment, or the fine-tune learns nothing new.
"""


def export_finetune(project: Project, *, project_dir, base_model: str | None = None) -> FinetuneResult:
    rows = assemble_dataset(project)
    recipe = recommend_recipe(len(rows), base_model=base_model)

    out = Path(project_dir) / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def _write(fn: str, content: str) -> None:
        atomic_write_text(out / fn, content)
        files.append(f"finetune/{fn}")

    _write("dataset.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _write("recipe.yaml", yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
    _write("train.py", _TRAIN_PY.replace("__BASE__", recipe["base_model"]).replace("__OUT__", recipe["output_dir"]))
    _write("README.md", _readme(recipe, len(rows)))

    return FinetuneResult(
        examples=len(rows), base_model=recipe["base_model"],
        method=recipe["method"], bundle_dir=str(out), files=files,
    )
