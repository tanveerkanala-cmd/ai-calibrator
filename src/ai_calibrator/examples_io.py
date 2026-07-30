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
from dataclasses import dataclass, field
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
    # Imported from the owner's own file (past replies, an FAQ, a spreadsheet):
    # human-authored, so trainable.
    return Example(input=inp, good_output=as_opt_str(out),
                   bad_output=as_opt_str(_pick(row, _BAD_KEYS)), source="human")


@dataclass
class ImportReport:
    """The outcome of a bulk import: usable examples plus an itemized account of
    everything skipped or hollowed out, so counts reconcile against the file."""
    examples: list[Example] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # human-readable, with file line numbers
    without_output: int = 0                             # kept, but no usable output column/key


def load_examples_report(path: str | Path) -> ImportReport:
    """Parse a .csv / .jsonl / .json / .yaml file into an :class:`ImportReport`.

    Column/key names are matched flexibly (input|question|prompt…, good_output|
    output|answer…). A CSV with no recognizable header falls back to
    first-column=input, second-column=output. Reads with ``utf-8-sig`` so a
    byte-order mark (common in Excel/CRM exports) doesn't poison the first
    header/field. Malformed rows are skipped WITH a reason and file line number
    rather than aborting the whole import. Raises ValueError only when the file
    is missing/unreadable/unsupported, cannot be parsed at all (a CSV whose
    quoting breaks the reader mid-field), or yields no usable examples."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"No such file: {p}")
    try:
        # utf-8-sig strips a leading BOM; without it the BOM sticks to the first
        # header cell ("﻿question") so header detection fails and the header
        # row is imported as data (and lands in the fine-tune set).
        text = p.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {p}: {exc}") from exc
    suffix = p.suffix.lower()

    report = ImportReport()
    # (row_dict_or_None, file_line_number) pairs — a None row is already malformed.
    indexed: list[tuple[object, int]]  # a row dict, or None for one already malformed
    if suffix == ".csv":
        # A csv.Error is NOT a ValueError, so an unguarded parse escapes the CLI's
        # `except ValueError` and reaches the user as a raw traceback. It is a
        # whole-file failure, not a per-row one: the reader is mid-field when it
        # gives up, so there is no row to attribute it to.
        try:
            indexed = list(_parse_csv(text))
        except csv.Error as exc:
            raise ValueError(
                f"{p.name} could not be read as CSV: {exc}. The likeliest cause is an unbalanced "
                'double-quote (") in a cell — everything after it reads as one giant field. '
                'Escape it as "" and retry.'
            ) from exc
    elif suffix == ".jsonl":
        indexed = []
        for lineno, ln in enumerate(text.splitlines(), start=1):
            if not ln.strip():
                continue
            try:
                indexed.append((json.loads(ln), lineno))
            except json.JSONDecodeError as exc:
                # ONE bad line must not abort a real-world export — skip + report.
                report.skipped.append(f"{p.name}:{lineno} — invalid JSON ({exc.msg})")
    elif suffix == ".json":
        try:
            data = json.loads(text or "[]")
        except json.JSONDecodeError as exc:  # already a ValueError — message it clearly
            raise ValueError(f"{p.name} is not valid JSON: {exc.msg} (line {exc.lineno}, column "
                             f"{exc.colno}). Fix the file and retry.") from exc
        rows = data if isinstance(data, list) else [data]
        indexed = [(r, i) for i, r in enumerate(rows, start=1)]
    elif suffix in (".yaml", ".yml"):
        # A YAMLError is NOT a ValueError, so an unguarded parse escapes the CLI's
        # `except ValueError` and reaches the user as a pyyaml traceback — for a
        # hand-written file, the likeliest failure there is.
        try:
            data = yaml.safe_load(text) or []
        except yaml.YAMLError as exc:
            where = getattr(exc, "problem_mark", None)
            spot = f" (line {where.line + 1}, column {where.column + 1})" if where is not None else ""
            what = getattr(exc, "problem", None) or exc
            raise ValueError(f"{p.name} is not valid YAML: {what}{spot}. "
                             "Check for stray tabs or unclosed quotes, then retry.") from exc
        rows = data if isinstance(data, list) else [data]
        indexed = [(r, i) for i, r in enumerate(rows, start=1)]
    else:
        raise ValueError(f"Unsupported example file type {suffix!r}. Use .csv, .jsonl, .json, or .yaml.")

    for row, lineno in indexed:
        if not isinstance(row, dict):
            report.skipped.append(f"{p.name}:{lineno} — not a record (got {type(row).__name__})")
            continue
        ex = _row_to_example(row)
        if ex is None:
            report.skipped.append(
                f"{p.name}:{lineno} — no usable input "
                f"(need a column/key named one of {', '.join(_INPUT_KEYS)})"
            )
            continue
        if not is_str(ex.good_output):
            report.without_output += 1
        report.examples.append(ex)

    if not report.examples:
        raise ValueError(
            f"No usable examples in {p.name} — each row needs an input "
            f"(a column/key named one of {', '.join(_INPUT_KEYS)}) and ideally an output."
        )
    return report


def load_examples_file(path: str | Path) -> list[Example]:
    """Back-compat convenience: just the examples from :func:`load_examples_report`."""
    return load_examples_report(path).examples


def _parse_csv(text: str) -> list[tuple[dict, int]]:
    reader = csv.reader(io.StringIO(text))
    # Track the source line of each row (skipping blank lines) so a skip report
    # can point at the file. csv.reader collapses embedded newlines within a
    # quoted field, so this is the row's starting line — close enough to locate it.
    all_rows: list[tuple[list[str], int]] = []
    for lineno, r in enumerate(reader, start=1):
        if any(c.strip() for c in r):
            all_rows.append((r, lineno))
    if not all_rows:
        return []
    header = [c.strip().lower() for c in all_rows[0][0]]
    has_header = any(h in _INPUT_KEYS for h in header)
    if has_header:
        return [(dict(zip(header, r, strict=False)), ln) for r, ln in all_rows[1:]]
    # headerless: first col is input, second (if present) is output
    return [({"input": r[0], "good_output": r[1] if len(r) > 1 else None}, ln) for r, ln in all_rows]


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
    """Collapse examples that share an input. Returns the number removed.

    The LAST occurrence wins, kept in the position of the first (so order is
    stable). Later is better here: examples accumulate over time, and the newest
    one for an input is the most recently ratified — a correction absorbed from
    live feedback, or a `teach` judgment. First-wins would delete the correction
    and keep the answer a human already rejected. But a later occurrence can be an
    input-only row (a list of bare questions imported afterwards), which carries no
    ratified answer at all — so the newest occurrence WITH an output wins, and a
    bare input only survives when no occurrence of that input has one."""
    def _has_output(ex: Example) -> bool:
        return is_str(ex.good_output) or is_str(ex.bad_output)

    last: dict[str, Example] = {}
    for ex in spec.examples:
        prior = last.get(ex.input)
        if prior is None or _has_output(ex) or not _has_output(prior):
            last[ex.input] = ex
    seen: set[str] = set()
    kept: list[Example] = []
    for ex in spec.examples:
        if ex.input in seen:
            continue
        seen.add(ex.input)
        kept.append(last[ex.input])
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
    # Only human-authored / human-ratified rows are fine-tuning targets, so this
    # is the count that decides whether the Advanced tier can do anything. Counting
    # every row told owners they were ready while `calibrate train` refused them.
    trainable = sum(1 for e in spec.examples if is_str(e.good_output) and e.trainable)
    return {
        "total": total,
        "unique_inputs": unique,
        "duplicates": total - unique,
        "with_output": with_output,
        "trainable": trainable,
        "engine_written": with_output - trainable,
        "recommended": RECOMMENDED_EXAMPLES,
        # Measured in TRAINABLE rows: a project whose examples are all
        # compiler-written has nothing to fine-tune on, and saying otherwise sends
        # the owner to `calibrate train`, which then refuses them.
        "short_by": max(0, RECOMMENDED_EXAMPLES - trainable),
        "enough_to_finetune": trainable >= RECOMMENDED_EXAMPLES,
    }
