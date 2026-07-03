"""Engine-Trainer — localize a cloud role onto your own model (the autonomy loop).

After running with logging on, ``<project>/logs/<role>.jsonl`` holds the cloud
engine's decisions for that role. This turns them into a fine-tuning dataset +
recipe for a small local model, and provides the **prove-it gate**: replay the
logged inputs through a candidate (local) engine and measure how well it
*reproduces* the cloud outputs (agreement). Only swap the local engine into the
role's binding once it clears your threshold — then repeat per role until the
tool runs on your own private, free, specialized engines.

The actual GPU training is handed off (like the fine-tune toolchain); the novel,
testable pieces here are dataset assembly from logs and the agreement gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .engines.base import Engine
from .finetune import _TRAIN_PY, recommend_recipe
from .store import atomic_write_text

TRAINABLE_ROLES = {"extractor", "interviewer", "predictor", "compiler", "judge"}


def read_log(project_dir: str | Path, role: str) -> list[dict]:
    """Read ``logs/<role>.jsonl``, skipping any malformed (e.g. interleaved) lines."""
    f = Path(project_dir) / "logs" / f"{role}.jsonl"
    if not f.exists():
        return []
    rows: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def assemble_role_dataset(project_dir: str | Path, role: str) -> list[dict]:
    """Chat-format SFT rows from logged (input → cloud output) pairs, de-duplicated.

    A structured (schema) output is serialized to a JSON string as the target, so
    the local model learns to emit the same structured decision."""
    seen: set[tuple] = set()
    rows: list[dict] = []
    for r in read_log(project_dir, role):
        prompt = r.get("prompt")
        output = r.get("output")
        if not isinstance(prompt, str) or not prompt.strip() or output is None:
            continue
        target = output if isinstance(output, str) else json.dumps(output, sort_keys=True)
        system = r.get("system") if isinstance(r.get("system"), str) else None
        key = (system or "", prompt, target)
        if key in seen:
            continue
        seen.add(key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": target})
        rows.append({"messages": messages})
    return rows


def human_judge_rows(project_dir: str | Path) -> list[dict]:
    """Ground-truth judge rows from saved judge-check labels.

    A logged judge row teaches the local model to *imitate the cloud judge* —
    including its mistakes. A judge-check label is what a HUMAN said the verdict
    should be, so it trains toward the truth instead. Each label is rebuilt into
    the exact prompt the judge role sees (eval.judge_prompt, single-criterion
    block); the target is the JSON the judge *should* have returned.

    Labels whose test, criterion, or scorecard no longer exists are skipped — the
    prompt can't be faithfully reconstructed without them."""
    from pydantic import ValidationError

    from .drift import load_scorecard
    from .eval import JUDGE_SYSTEM, judge_prompt
    from .judge_check import all_labels
    from .store import load_project

    try:
        project = load_project(project_dir)
    except (FileNotFoundError, ValueError, ValidationError):
        return []
    if project.spec is None:
        return []
    desc_by_id = {c.id: c.description for c in project.spec.eval_criteria}
    input_by_test = {t.id: t.input for t in project.tests}

    rows: list[dict] = []
    seen: set[tuple] = set()
    for run_id, labels in all_labels(project_dir):
        try:
            outputs = {r.test_id: r.output for r in load_scorecard(project_dir, run_id).results}
        except (FileNotFoundError, ValueError, ValidationError):
            continue
        for label in labels:
            tid, cid = label.get("test_id"), label.get("criterion_id")
            if tid not in outputs or cid not in desc_by_id or tid not in input_by_test:
                continue
            passed = bool(label.get("passed"))
            prompt = judge_prompt(input_by_test[tid], outputs[tid], [(cid, desc_by_id[cid])])
            target = json.dumps({"results": [{
                "criterion_id": cid, "passed": passed, "score": 1.0 if passed else 0.0,
                "rationale": "human-labeled ground truth (judge-check)",
            }]}, sort_keys=True)
            key = (JUDGE_SYSTEM, prompt)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]})
    return rows


@dataclass
class TrainEngineResult:
    role: str
    examples: int
    base_model: str
    bundle_dir: str
    files: list[str]
    human_examples: int = 0  # of `examples`, how many are human ground truth


def _engine_readme(role: str, recipe: dict, n: int, human: int = 0) -> str:
    base = recipe["base_model"]
    source = (f"**{n}** example(s): {n - human} logged cloud decision(s) + {human} human "
              "ground-truth label(s) from `calibrate judge-check` (ground truth wins on conflict)"
              if human else f"**{n}** logged interaction(s)")
    return f"""# Engine-Trainer — localize the `{role}` role onto `{base}`

Fine-tunes a LOCAL model to reproduce your cloud engine's **{role}** decisions,
from {source}. The end state: the tool runs this role on your
own private, free model — once it's PROVEN to match the cloud one.

## 1. Train (GPU — LoRA of a 7B ~16GB VRAM, else a rented cloud GPU)
```bash
pip install "transformers>=4.44" "trl>=0.9" peft datasets accelerate
python train.py        # → ./{recipe["output_dir"]}/
```

## 2. Serve it (e.g. merge the adapter + `ollama create my-{role}`), then PROVE it:
```bash
calibrate train-engine {role} --prove --candidate my-{role}@ollama
```
This replays your logged inputs through the local engine and reports how often it
agrees with the cloud engine. Swap it into `engines.{role}` in `project.yaml`
**only** once agreement clears your threshold — otherwise keep the cloud engine.

> Logged from your own runs (logging is opt-in: `calibrate log --on`). The data
> never leaves your machine.
"""


def export_engine_bundle(project_dir: str | Path, role: str, *, base_model: str | None = None) -> TrainEngineResult:
    """Write ``<project>/engines/<role>/`` : dataset.jsonl, recipe.yaml, train.py, README.

    For the judge role, human judge-check labels become ground-truth rows: they
    are added to the dataset, and any logged (imitation) row asking the exact
    same question is dropped in favor of the human answer."""
    rows = assemble_role_dataset(project_dir, role)
    human: list[dict] = human_judge_rows(project_dir) if role == "judge" else []
    if human:
        claimed = {(r["messages"][0]["content"], r["messages"][-2]["content"]) for r in human}

        def _key(row: dict) -> tuple:
            msgs = row["messages"]
            system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
            return (system, msgs[-2]["content"])

        rows = [r for r in rows if _key(r) not in claimed] + human
    recipe = recommend_recipe(len(rows), base_model=base_model)

    out = Path(project_dir) / "trained-engines" / role
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def _write(fn: str, content: str) -> None:
        atomic_write_text(out / fn, content)
        files.append(f"trained-engines/{role}/{fn}")

    _write("dataset.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _write("recipe.yaml", yaml.safe_dump(recipe, sort_keys=False))
    _write("train.py", _TRAIN_PY.replace("__BASE__", recipe["base_model"]).replace("__OUT__", recipe["output_dir"]))
    _write("README.md", _engine_readme(role, recipe, len(rows), len(human)))

    return TrainEngineResult(role=role, examples=len(rows), base_model=recipe["base_model"],
                             bundle_dir=str(out), files=files, human_examples=len(human))


# --- the prove-it (agreement) gate ------------------------------------------

def _normalize(out: Any) -> str:
    """Canonical comparable form: structured → key-sorted JSON; text → stripped."""
    if isinstance(out, (dict, list)):
        return json.dumps(out, sort_keys=True)
    return str(out).strip()


def _judge_verdicts(out: Any) -> dict | None:
    """Extract {criterion_id: passed} from a judge output, or None if not one."""
    if isinstance(out, dict) and isinstance(out.get("results"), list):
        return {r.get("criterion_id"): bool(r.get("passed"))
                for r in out["results"] if isinstance(r, dict)}
    return None


def agreement(reference: list, candidate: list, *, role: str | None = None) -> float:
    """How well ``candidate`` outputs reproduce ``reference`` outputs, in [0,1].

    For the judge, agreement is per-criterion pass/fail match (semantic — rationale
    wording is ignored). For other roles it's normalized exact match (conservative).
    Missing candidate outputs count as non-agreement (denominator = len(reference))."""
    if not reference:
        return 0.0
    n = min(len(reference), len(candidate))
    if n == 0:
        return 0.0
    if role == "judge":
        total = 0.0
        for i in range(n):
            rv, cv = _judge_verdicts(reference[i]), _judge_verdicts(candidate[i])
            if rv is None or cv is None:
                total += 1.0 if _normalize(reference[i]) == _normalize(candidate[i]) else 0.0
                continue
            keys = set(rv) | set(cv)
            total += 1.0 if not keys else sum(1 for k in keys if rv.get(k) == cv.get(k)) / len(keys)
        return total / len(reference)
    matches = sum(1 for i in range(n) if _normalize(reference[i]) == _normalize(candidate[i]))
    return matches / len(reference)


@dataclass
class ProveResult:
    role: str
    samples: int
    agreement: float
    threshold: float

    @property
    def passes(self) -> bool:
        return self.samples > 0 and self.agreement >= self.threshold


def prove_engine(
    project_dir: str | Path,
    role: str,
    candidate: Engine,
    *,
    threshold: float = 0.9,
    limit: int | None = None,
) -> ProveResult:
    """Replay logged inputs through ``candidate`` and measure agreement vs the
    logged cloud outputs — the gate for trusting a localized engine."""
    records = [r for r in read_log(project_dir, role) if isinstance(r.get("prompt"), str)]
    if limit is not None:
        records = records[:limit]
    reference = [r.get("output") for r in records]
    produced = [candidate.complete(r["prompt"], system=r.get("system"), schema=r.get("schema"))
                for r in records]
    return ProveResult(role=role, samples=len(records),
                       agreement=agreement(reference, produced, role=role), threshold=threshold)
