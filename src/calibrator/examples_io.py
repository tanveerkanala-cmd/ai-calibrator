"""Bulk import + curation of training examples — the fuel for the Advanced tier.

Fine-tuning is only as good as its data, and the data is the hard part. Most
owners already HAVE examples (past support replies, an FAQ, a spreadsheet of
Q&A). This turns "type 50 examples one at a time" into "import your existing
file", plus dedup so the set stays clean. Everything here is engine-free.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import yaml

from .coerce import as_opt_str, is_str
from .models import BehaviorSpec, Example

# accepted column / key names, in priority order
_INPUT_KEYS = ("input", "question", "prompt", "user", "q")
_OUTPUT_KEYS = ("good_output", "output", "answer", "response", "assistant", "a", "ideal")
_BAD_KEYS = ("bad_output", "bad", "rejected")


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    for k in keys:
        if k in lower and is_str(lower[k]):
            return lower[k]
    return None


def _row_to_example(row: dict) -> Example | None:
    inp = _pick(row, _INPUT_KEYS)
    out = _pick(row, _OUTPUT_KEYS)
    if not is_str(inp):
        return None                       # an example must at least have an input
    return Example(input=inp, good_output=as_opt_str(out), bad_output=as_opt_str(_pick(row, _BAD_KEYS)))


def load_examples_file(path: str | Path) -> list[Example]:
    """Parse input→output example pairs from a .csv / .jsonl / .json / .yaml file.

    Column/key names are matched flexibly (input|question|prompt…, good_output|
    output|answer…). A CSV with no recognizable header falls back to
    first-column=input, second-column=output. Raises ValueError on an unreadable
    or unrecognized file (friendly message)."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"No such file: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {p}: {exc}") from exc
    suffix = p.suffix.lower()

    rows: list[dict]
    if suffix == ".csv":
        rows = _parse_csv(text)
    elif suffix == ".jsonl":
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    elif suffix == ".json":
        data = json.loads(text or "[]")
        rows = data if isinstance(data, list) else [data]
    elif suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text) or []
        rows = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported example file type {suffix!r}. Use .csv, .jsonl, .json, or .yaml.")

    examples = [ex for row in rows if isinstance(row, dict) and (ex := _row_to_example(row))]
    if not examples:
        raise ValueError(
            f"No usable examples in {p.name} — each row needs an input "
            f"(a column/key named one of {', '.join(_INPUT_KEYS)}) and ideally an output."
        )
    return examples


def _parse_csv(text: str) -> list[dict]:
    reader = csv.reader(io.StringIO(text))
    all_rows = [r for r in reader if any(c.strip() for c in r)]
    if not all_rows:
        return []
    header = [c.strip().lower() for c in all_rows[0]]
    has_header = any(h in _INPUT_KEYS for h in header)
    if has_header:
        return [dict(zip(header, r, strict=False)) for r in all_rows[1:]]
    # headerless: first col is input, second (if present) is output
    return [{"input": r[0], "good_output": r[1] if len(r) > 1 else None} for r in all_rows]


def merge_examples(spec: BehaviorSpec, new: list[Example], *, dedup: bool = True) -> tuple[int, int]:
    """Append ``new`` to ``spec.examples``. With ``dedup``, skip inputs already
    present (existing OR earlier in this batch). Returns (added, skipped)."""
    seen = {e.input for e in spec.examples} if dedup else set()
    added = skipped = 0
    for ex in new:
        if dedup and ex.input in seen:
            skipped += 1
            continue
        spec.examples.append(ex)
        seen.add(ex.input)
        added += 1
    return added, skipped


def dedup_examples(spec: BehaviorSpec) -> int:
    """Remove later examples whose input duplicates an earlier one. Returns the
    number removed (order-preserving — the first occurrence wins)."""
    seen: set[str] = set()
    kept: list[Example] = []
    for ex in spec.examples:
        if ex.input in seen:
            continue
        seen.add(ex.input)
        kept.append(ex)
    removed = len(spec.examples) - len(kept)
    spec.examples = kept
    return removed


# Fine-tuning rule of thumb: below this, a LoRA rarely beats the prompt+RAG baseline.
RECOMMENDED_EXAMPLES = 50


def examples_status(spec: BehaviorSpec) -> dict:
    """A summary for review/curation: totals, duplicate count, and how far the set
    is from the recommended size for a fine-tune."""
    inputs = [e.input for e in spec.examples]
    total = len(inputs)
    unique = len(set(inputs))
    with_output = sum(1 for e in spec.examples if is_str(e.good_output))
    return {
        "total": total,
        "unique_inputs": unique,
        "duplicates": total - unique,
        "with_output": with_output,
        "recommended": RECOMMENDED_EXAMPLES,
        "short_by": max(0, RECOMMENDED_EXAMPLES - unique),
        "enough_to_finetune": unique >= RECOMMENDED_EXAMPLES,
    }
