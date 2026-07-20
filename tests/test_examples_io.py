"""Tests for bulk example import + curation."""

import pytest

from ai_calibrator.examples_io import (
    dedup_examples, examples_status, load_examples_file, merge_examples,
)
from ai_calibrator.models import BehaviorSpec, Example


def test_csv_with_flexible_headers(tmp_path):
    f = tmp_path / "qa.csv"
    f.write_text("question,answer\nHow do I return?,Within 30 days.\nShipping?,US only.\n")
    exs = load_examples_file(f)
    assert [e.input for e in exs] == ["How do I return?", "Shipping?"]
    assert exs[0].good_output == "Within 30 days."


def test_csv_headerless_falls_back_to_first_two_columns(tmp_path):
    f = tmp_path / "qa.csv"
    f.write_text("What is your name?,I'm the support bot.\nBye,See you!\n")
    exs = load_examples_file(f)
    assert exs[0].input == "What is your name?" and exs[0].good_output == "I'm the support bot."


def test_csv_ragged_rows_tolerated(tmp_path):
    f = tmp_path / "qa.csv"
    f.write_text("input,good_output\nonly an input\nq2,a2\n")   # first row missing the output col
    exs = load_examples_file(f)
    assert exs[0].input == "only an input" and exs[0].good_output is None
    assert exs[1].good_output == "a2"


def test_jsonl_and_json_and_yaml(tmp_path):
    jl = tmp_path / "d.jsonl"
    jl.write_text('{"input":"a","output":"A"}\n{"input":"b","good_output":"B"}\n')
    assert [e.good_output for e in load_examples_file(jl)] == ["A", "B"]
    js = tmp_path / "d.json"
    js.write_text('[{"prompt":"a","response":"A"}]')          # flexible keys
    assert load_examples_file(js)[0].input == "a"
    ya = tmp_path / "d.yaml"
    ya.write_text("- input: a\n  answer: A\n")
    assert load_examples_file(ya)[0].good_output == "A"


def test_unsupported_and_empty_raise_friendly(tmp_path):
    bad = tmp_path / "d.txt"; bad.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported"):
        load_examples_file(bad)
    empty = tmp_path / "e.jsonl"; empty.write_text("\n\n")        # no rows → no usable examples
    with pytest.raises(ValueError, match="No usable examples"):
        load_examples_file(empty)
    with pytest.raises(ValueError, match="No such file"):
        load_examples_file(tmp_path / "missing.jsonl")


def test_merge_dedups_within_batch_and_against_existing():
    spec = BehaviorSpec(goal="g", examples=[Example(input="a", good_output="A")])
    added, skipped = merge_examples(spec, [Example(input="a"), Example(input="b"), Example(input="b")])
    assert added == 1 and skipped == 2                      # 'a' exists, second 'b' is a batch dup
    assert [e.input for e in spec.examples] == ["a", "b"]


def test_dedup_examples_keeps_first():
    spec = BehaviorSpec(goal="g", examples=[
        Example(input="a", good_output="first"), Example(input="a", good_output="second"),
        Example(input="b", good_output="B")])
    assert dedup_examples(spec) == 1
    assert [(e.input, e.good_output) for e in spec.examples] == [("a", "first"), ("b", "B")]


def test_examples_status_threshold_guidance():
    spec = BehaviorSpec(goal="g", examples=[Example(input=f"q{i}", good_output="x") for i in range(10)])
    st = examples_status(spec)
    assert st["unique_inputs"] == 10 and st["short_by"] == 40 and st["enough_to_finetune"] is False
    spec.examples = [Example(input=f"q{i}", good_output="x") for i in range(55)]
    assert examples_status(spec)["enough_to_finetune"] is True
