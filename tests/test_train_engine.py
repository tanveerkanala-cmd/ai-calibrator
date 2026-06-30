"""Engine-Trainer — logging, dataset assembly, and the agreement / prove gate."""

import json

import pytest
from typer.testing import CliRunner

from calibrator.cli import app
from calibrator.engine_log import LoggingEngine, wrap_engine
from calibrator.models import Project
from calibrator.store import save_project
from calibrator.train_engine import (
    agreement,
    assemble_role_dataset,
    export_engine_bundle,
    prove_engine,
    read_log,
)

runner = CliRunner()


class FakeEngine:
    def __init__(self, output):
        self.output = output
        self.name = "fake@test"

    def complete(self, prompt, *, system=None, schema=None):
        return self.output


# --- logging -----------------------------------------------------------------

def test_logging_engine_records_and_passes_through(tmp_path):
    inner = FakeEngine({"results": [{"criterion_id": "c1", "passed": True}]})
    eng = LoggingEngine(inner, "judge", tmp_path / "logs")
    out = eng.complete("the prompt", system="sys", schema={"type": "object"})
    assert out == inner.output  # passes through unchanged

    rec = json.loads((tmp_path / "logs" / "judge.jsonl").read_text().splitlines()[0])
    assert rec["role"] == "judge" and rec["prompt"] == "the prompt"
    assert rec["system"] == "sys" and rec["schema"] == {"type": "object"}
    assert rec["output"] == inner.output


def test_logging_engine_appends(tmp_path):
    eng = LoggingEngine(FakeEngine("x"), "extractor", tmp_path / "logs")
    eng.complete("a")
    eng.complete("b")
    assert len((tmp_path / "logs" / "extractor.jsonl").read_text().splitlines()) == 2


def test_wrap_engine_toggle(tmp_path):
    inner = FakeEngine("x")
    assert wrap_engine(inner, "judge", tmp_path, enabled=False) is inner
    wrapped = wrap_engine(inner, "judge", tmp_path, enabled=True)
    assert isinstance(wrapped, LoggingEngine) and wrapped.name == inner.name


# --- dataset assembly --------------------------------------------------------

def test_read_log_skips_malformed(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    (d / "judge.jsonl").write_text('{"role":"judge","prompt":"p","output":"o"}\nNOT JSON\n\n[1,2]\n')
    rows = read_log(tmp_path, "judge")
    assert len(rows) == 1 and rows[0]["prompt"] == "p"  # malformed + non-dict skipped


def test_assemble_role_dataset_dedups_and_serializes(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    recs = [
        {"role": "judge", "system": "S", "prompt": "p1", "schema": {"type": "object"},
         "output": {"results": [{"criterion_id": "c1", "passed": True}]}},
        {"role": "judge", "system": "S", "prompt": "p1", "schema": {"type": "object"},
         "output": {"results": [{"criterion_id": "c1", "passed": True}]}},  # exact dup
        {"role": "judge", "system": "S", "prompt": "p2", "output": "plain text"},
        {"role": "judge", "prompt": "", "output": "x"},  # empty prompt skipped
    ]
    (d / "judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows = assemble_role_dataset(tmp_path, "judge")
    assert len(rows) == 2  # dup + empty-prompt dropped
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert "results" in rows[0]["messages"][-1]["content"]   # structured target serialized
    assert rows[1]["messages"][-1]["content"] == "plain text"


def test_export_engine_bundle_writes_files(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    (d / "judge.jsonl").write_text(json.dumps({"role": "judge", "prompt": "p", "output": "o"}) + "\n")
    result = export_engine_bundle(tmp_path, "judge")
    assert result.examples == 1
    base = tmp_path / "trained-engines" / "judge"
    for fn in ["dataset.jsonl", "recipe.yaml", "train.py", "README.md"]:
        assert (base / fn).exists()


# --- agreement + prove gate --------------------------------------------------

def test_agreement_default_exact_match():
    assert agreement(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
    assert agreement([{"k": 1, "j": 2}], [{"j": 2, "k": 1}]) == 1.0  # order-insensitive


def test_agreement_judge_is_verdict_based():
    ref = [{"results": [{"criterion_id": "c1", "passed": True}, {"criterion_id": "c2", "passed": False}]}]
    same = [{"results": [{"criterion_id": "c1", "passed": True, "rationale": "diff wording"},
                         {"criterion_id": "c2", "passed": False}]}]
    assert agreement(ref, same, role="judge") == 1.0          # rationale ignored
    flipped = [{"results": [{"criterion_id": "c1", "passed": False}, {"criterion_id": "c2", "passed": False}]}]
    assert agreement(ref, flipped, role="judge") == 0.5       # one verdict differs


def test_agreement_penalizes_missing_outputs():
    assert agreement(["a", "b"], ["a"]) == 0.5  # denominator is len(reference)


def test_prove_engine_replays_and_gates(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    recs = [
        {"role": "judge", "prompt": "p1", "schema": {"type": "object"},
         "output": {"results": [{"criterion_id": "c1", "passed": True}]}},
        {"role": "judge", "prompt": "p2", "schema": {"type": "object"},
         "output": {"results": [{"criterion_id": "c1", "passed": False}]}},
    ]
    (d / "judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    class MatchingCandidate:
        name = "local@ollama"

        def complete(self, prompt, *, system=None, schema=None):
            return {"results": [{"criterion_id": "c1", "passed": prompt == "p1"}]}

    res = prove_engine(tmp_path, "judge", MatchingCandidate(), threshold=0.9)
    assert res.samples == 2 and res.agreement == 1.0 and res.passes is True

    class HalfCandidate:
        name = "local@ollama"

        def complete(self, prompt, *, system=None, schema=None):
            return {"results": [{"criterion_id": "c1", "passed": True}]}  # wrong on p2

    res2 = prove_engine(tmp_path, "judge", HalfCandidate(), threshold=0.9)
    assert res2.agreement == 0.5 and res2.passes is False


def test_prove_engine_no_logs_is_zero(tmp_path):
    class C:
        name = "c"

        def complete(self, prompt, *, system=None, schema=None):
            return "x"

    res = prove_engine(tmp_path, "judge", C(), threshold=0.5)
    assert res.samples == 0 and res.passes is False


# --- CLI ---------------------------------------------------------------------

def test_cli_log_toggle(tmp_path):
    save_project(Project(name="p", goal="g"), tmp_path)
    assert "OFF" in runner.invoke(app, ["log", str(tmp_path)]).output
    r = runner.invoke(app, ["log", str(tmp_path), "--on"])
    assert r.exit_code == 0 and "ON" in r.output
    from calibrator.store import load_project
    assert load_project(tmp_path).log_interactions is True


def test_cli_train_engine_validations(tmp_path):
    save_project(Project(name="p", goal="g"), tmp_path)
    # unknown role
    assert runner.invoke(app, ["train-engine", "bogus", str(tmp_path)]).exit_code == 1
    # no logs yet
    r = runner.invoke(app, ["train-engine", "judge", str(tmp_path)])
    assert r.exit_code == 1 and "No logged judge decisions" in r.output
    # --prove without --candidate
    r2 = runner.invoke(app, ["train-engine", "judge", str(tmp_path), "--prove"])
    assert r2.exit_code == 1 and "needs --candidate" in r2.output
