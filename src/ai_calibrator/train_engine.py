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

from .coerce import as_bool
from .engines.base import Engine
from .finetune import recommend_recipe, render_train_py
from .identity import result_matches_test
from .store import atomic_write_text

TRAINABLE_ROLES = {"extractor", "interviewer", "predictor", "compiler", "judge"}

# Of those, the roles anything actually logs today: the judge is wrapped during
# `eval` and `ci`, the compiler during `eval --refine`. Nothing wraps the other
# three, so pointing their user at "run eval and retry" sends them after data
# that will never appear.
LOGGED_ROLES = {"judge", "compiler"}


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


HUMAN_RATIONALE = "human-labeled ground truth (judge-check)"


def _ground_truth_result(criterion_id: str, passed: bool) -> dict:
    """One graded criterion as the judge *should* have returned it."""
    return {"criterion_id": criterion_id, "passed": passed,
            "score": 1.0 if passed else 0.0,
            "rationale": HUMAN_RATIONALE}


def _dedup_rows(rows: list[dict]) -> list[dict]:
    """Drop repeat (system, prompt, target) rows, keeping order.

    ``assemble_role_dataset`` already applies this key to the log, but patching
    changes targets: the same prompt logged twice with different verdicts
    (``judge_passes`` self-consistency, or a re-run) survives that first pass and
    then collapses to identical rows once a human label overwrites both."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for row in rows:
        messages = row["messages"]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        key = (system, messages[-2]["content"], messages[-1]["content"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _graded_item(test_input: str, output: str) -> str:
    """The head of a judge prompt: WHICH answer was graded, before the criteria.

    Every judge call about the same (test, output) pair shares it, whatever
    criteria that call asked about — which is what lets a single-criterion human
    label find the (usually multi-criterion) call it corrects. Derived from
    ``judge_prompt`` itself so the two can't drift; if that format ever loses the
    marker no logged row matches, and each label falls back to a row of its own.

    Split on the LAST marker, not the first. The criteria block is the tail of
    the prompt, so the final occurrence is the real boundary — while the first is
    whichever one the graded question or answer happened to contain, which
    truncates the item to a short prefix that then claims unrelated logged rows
    and overwrites the training target of a test nobody labeled."""
    from .eval import judge_prompt

    head, marker, _ = judge_prompt(test_input, output, []).rpartition("CRITERIA:")
    return head + marker


def _ground_truth(project_dir: str | Path) -> list[tuple[str, str, bool, dict | None]]:
    """``(graded item, criterion id, human verdict, standalone row)`` per label.

    A logged judge row teaches the local model to *imitate the cloud judge* —
    including its mistakes. A judge-check label is what a HUMAN said the verdict
    should be, so it trains toward the truth instead. The standalone row rebuilds
    the prompt the judge role sees when a test targets that one criterion; the
    graded item is what ties the label back to the call that actually graded it.

    Labels whose test, criterion, or scorecard no longer exists are skipped — the
    prompt can't be faithfully reconstructed without them."""
    from pydantic import ValidationError

    from .drift import load_scorecard
    from .compile import render_system_prompt
    from .eval import judge_prompt, judge_system
    from .judge_check import all_labels
    from .store import load_project

    try:
        project = load_project(project_dir)
    except (FileNotFoundError, ValueError, ValidationError):
        return []
    if project.spec is None:
        return []
    # Every criterion the spec defines, plus which ones the judge is still asked
    # about. A criterion carrying a deterministic check is graded by code and
    # never reaches the judge (eval.run_eval), so no NEW standalone row should be
    # invented for it — but a logged judge call from before the check was attached
    # still exists in the dataset, and a human verdict must still correct it.
    # Dropping the label outright let attaching a check silently restore the very
    # judgment the human overturned.
    # The judge grades with the AI's own instructions in its system message, so a
    # ground-truth row must carry the identical one — a row that trains on a
    # different system message teaches the local judge a distribution the cloud
    # judge never graded under.
    system = judge_system(render_system_prompt(project.spec))
    desc_by_id = {c.id: c.description for c in project.spec.eval_criteria}
    judged_ids = {c.id for c in project.spec.eval_criteria if c.check is None}
    test_by_id = {t.id: t for t in project.tests}

    truth: list[tuple[str, str, bool, dict | None]] = []
    seen: set[tuple] = set()
    for run_id, labels in all_labels(project_dir):
        try:
            results = {r.test_id: r for r in load_scorecard(project_dir, run_id).results}
        except (FileNotFoundError, ValueError, ValidationError):
            continue
        for label in labels:
            tid, cid = label.get("test_id"), label.get("criterion_id")
            if tid not in results or cid not in desc_by_id or tid not in test_by_id:
                continue
            # The saved run holds the ANSWER; the current suite holds the
            # QUESTION. `compile` re-mints t1..tN, so pairing them by id alone
            # can ask the model to grade an answer to a question that was never
            # put to it — and then stamp a human's verdict on that invented
            # pair as ground truth. A label whose test no longer asks what the
            # run asked is not recoverable, so it is dropped.
            if not result_matches_test(results[tid], test_by_id[tid]):
                continue
            passed = as_bool(label.get("passed"))
            prompt = judge_prompt(test_by_id[tid].input, results[tid].output, [(cid, desc_by_id[cid])])
            target = json.dumps({"results": [_ground_truth_result(cid, passed)]}, sort_keys=True)
            key = (system, prompt)
            if key in seen:
                continue
            seen.add(key)
            # A standalone row teaches the judge to answer this question. Only
            # mint one for a criterion the judge is actually asked — a code-graded
            # one gets the correction applied to its logged row (below) and
            # nothing new invented for it.
            row = {"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]} if cid in judged_ids else None
            truth.append((_graded_item(test_by_id[tid].input, results[tid].output), cid, passed, row))
    return truth


def human_judge_rows(project_dir: str | Path) -> list[dict]:
    """Ground-truth judge rows from saved judge-check labels — see ``_ground_truth``."""
    return [row for _, _, _, row in _ground_truth(project_dir) if row is not None]


def _apply_ground_truth(row: dict, verdicts: dict[str, bool]) -> tuple[dict, set[str]]:
    """``row`` with every human-labeled criterion's verdict replacing the judge's,
    plus the ids that were replaced.

    The correction is a PATCH, not a swap: one judge call grades all of a test's
    judged criteria at once, so dropping the row would discard the verdicts the
    human never disputed, and appending beside it would leave the dataset teaching
    the very verdict they overturned."""
    try:
        target = json.loads(row["messages"][-1]["content"])
    except ValueError:
        return row, set()  # a plain-text target (assemble_role_dataset passes those through)
    results = target.get("results") if isinstance(target, dict) else None
    if not isinstance(results, list):
        return row, set()
    applied: set[str] = set()
    for r in results:
        # String ids only — an unhashable one would raise TypeError on the lookup.
        if isinstance(r, dict) and isinstance(r.get("criterion_id"), str) and r["criterion_id"] in verdicts:
            cid = r["criterion_id"]
            r.update(_ground_truth_result(cid, verdicts[cid]))
            applied.add(cid)
    if not applied:
        return row, applied
    return {"messages": row["messages"][:-1] + [
        {"role": "assistant", "content": json.dumps(target, sort_keys=True)}]}, applied


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
pip install "transformers>=4.56.2" "trl>=1.0" peft datasets accelerate pyyaml
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

    For the judge role, human judge-check labels are ground truth: each one
    overwrites the verdict the logged judge call gave that criterion, and a label
    no logged call answers becomes a row of its own."""
    # role becomes a directory component — gate it to the known roles so no caller
    # (Core included) can traverse out of trained-engines/ with e.g. "../../x".
    if role not in TRAINABLE_ROLES:
        raise ValueError(f"role must be one of: {', '.join(sorted(TRAINABLE_ROLES))} (got {role!r})")
    rows = assemble_role_dataset(project_dir, role)
    truth = _ground_truth(project_dir) if role == "judge" else []
    human = 0
    if truth:
        # Match on the GRADED ITEM (the head of the user turn, always messages[-2])
        # rather than on the whole prompt: a label names ONE criterion while the
        # call that graded it asked about every judged criterion of that test, so
        # comparing prompts whole only ever matched single-criterion tests — and
        # left the contradicted verdict in the dataset for all the others.
        verdicts: dict[str, dict[str, bool]] = {}
        for item, cid, passed, _ in truth:
            verdicts.setdefault(item, {})[cid] = passed
        corrected: set[tuple[str, str]] = set()
        patched: list[dict] = []
        for row in rows:
            prompt = row["messages"][-2]["content"]
            # A different name from the loop variable above, which is always a str.
            match = next((i for i in verdicts if prompt.startswith(i)), None)
            if match is not None:
                row, applied = _apply_ground_truth(row, verdicts[match])
                corrected |= {(match, cid) for cid in applied}
            patched.append(row)
        unanswered = [row for item, cid, _, row in truth
                      if row is not None and (item, cid) not in corrected]
        rows = _dedup_rows(patched + unanswered)
        # Count the rows that ended up carrying a human verdict, after dedup —
        # counting patches instead reported one human decision as several
        # whenever the same prompt had been logged more than once.
        human = sum(1 for r in rows if HUMAN_RATIONALE in r["messages"][-1]["content"])
    recipe = recommend_recipe(len(rows), base_model=base_model)

    out = Path(project_dir) / "trained-engines" / role
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def _write(fn: str, content: str) -> None:
        atomic_write_text(out / fn, content)
        files.append(f"trained-engines/{role}/{fn}")

    _write("dataset.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _write("recipe.yaml", yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
    _write("train.py", render_train_py(recipe))
    _write("README.md", _engine_readme(role, recipe, len(rows), human))

    return TrainEngineResult(role=role, examples=len(rows), base_model=recipe["base_model"],
                             bundle_dir=str(out), files=files, human_examples=human)


# --- the prove-it (agreement) gate ------------------------------------------

def _normalize(out: Any) -> str:
    """Canonical comparable form: structured → key-sorted JSON; text → stripped."""
    if isinstance(out, (dict, list)):
        return json.dumps(out, sort_keys=True)
    return str(out).strip()


def _judge_verdicts(out: Any) -> dict | None:
    """Extract {criterion_id: passed} from a judge output, or None if not one."""
    if isinstance(out, dict) and isinstance(out.get("results"), list):
        # String ids only: an unhashable criterion_id would raise TypeError and
        # abort the whole prove-it gate, exactly as in eval._judge. as_bool, not
        # bool: a local candidate judge that emits the STRING "false" would
        # otherwise have every verdict read as a pass, and this is the gate that
        # decides whether that judge is trustworthy enough to replace the cloud one.
        return {r["criterion_id"]: as_bool(r.get("passed"))
                for r in out["results"]
                if isinstance(r, dict) and isinstance(r.get("criterion_id"), str)}
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
    # Skip rows with no logged output (same filter as assemble_role_dataset) —
    # a None reference would otherwise skew the agreement the prove-it gate reports.
    records = [r for r in read_log(project_dir, role)
               if isinstance(r.get("prompt"), str) and r.get("output") is not None]
    if limit is not None:
        records = records[:limit]
    reference = [r.get("output") for r in records]
    produced = [candidate.complete(r["prompt"], system=r.get("system"), schema=r.get("schema"))
                for r in records]
    return ProveResult(role=role, samples=len(records),
                       agreement=agreement(reference, produced, role=role), threshold=threshold)
