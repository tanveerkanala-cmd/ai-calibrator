"""Engine-Trainer — logging, dataset assembly, and the agreement / prove gate."""

import json
from pathlib import Path

import yaml

import pytest
from typer.testing import CliRunner

from ai_calibrator.cli import app
from ai_calibrator.engine_log import LoggingEngine, wrap_engine
from ai_calibrator.models import Project
from ai_calibrator.store import save_project
from ai_calibrator.train_engine import (
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

    rec = json.loads((tmp_path / "logs" / "judge.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["role"] == "judge" and rec["prompt"] == "the prompt"
    assert rec["system"] == "sys" and rec["schema"] == {"type": "object"}
    assert rec["output"] == inner.output


def test_logging_engine_appends(tmp_path):
    eng = LoggingEngine(FakeEngine("x"), "extractor", tmp_path / "logs")
    eng.complete("a")
    eng.complete("b")
    assert len((tmp_path / "logs" / "extractor.jsonl").read_text(encoding="utf-8").splitlines()) == 2


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


def _judge_log_prompt(i: int) -> str:
    return (f"INPUT:\nquestion {i}?\n\nAI OUTPUT:\nanswer {i}\n\n"
            "CRITERIA:\n- c1: cites the policy\n\nGrade each criterion.")


def _seed_judge_log(tmp_path, n=40, *, copies=1) -> dict:
    """`n` distinct logged judge questions, each logged `copies` time(s).

    Returns {prompt: the verdict the cloud judge logged for it}."""
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    truth: dict[str, bool] = {}
    lines = []
    for i in range(n):
        prompt, passed = _judge_log_prompt(i), i % 3 != 0
        truth[prompt] = passed
        for _ in range(copies):
            lines.append(json.dumps({
                "role": "judge", "system": "S", "prompt": prompt, "schema": {"type": "object"},
                "output": {"results": [{"criterion_id": "c1", "passed": passed,
                                        "score": 1.0 if passed else 0.0, "rationale": "cloud"}]}}))
    (logs / "judge.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    return truth


def _q_index(prompt: str) -> int:
    return int(prompt.split("question ", 1)[1].split("?", 1)[0])


def _dataset_prompts(tmp_path, role="judge") -> set:
    """The questions dataset.jsonl actually trains on."""
    f = tmp_path / "trained-engines" / role / "dataset.jsonl"
    return {json.loads(line)["messages"][-2]["content"]
            for line in f.read_text(encoding="utf-8").splitlines() if line.strip()}


class _Replayer:
    """A candidate judge that answers `verdict(prompt)` and records what it was asked."""

    name = "local@ollama"

    def __init__(self, verdict):
        self.verdict = verdict
        self.asked: list[str] = []

    def complete(self, prompt, *, system=None, schema=None):
        self.asked.append(prompt)
        return {"results": [{"criterion_id": "c1", "passed": self.verdict(prompt)}]}


def test_prove_engine_replays_and_gates(tmp_path):
    truth = _seed_judge_log(tmp_path)
    export_engine_bundle(tmp_path, "judge")

    res = prove_engine(tmp_path, "judge", _Replayer(lambda p: truth[p]), threshold=0.9)
    assert res.samples >= 3 and res.agreement == 1.0 and res.passes is True

    half = _Replayer(lambda p: truth[p] if _q_index(p) % 2 else not truth[p])
    res2 = prove_engine(tmp_path, "judge", half, threshold=0.9)
    assert res2.agreement < 0.9 and res2.passes is False


def test_prove_gate_fails_a_candidate_that_only_memorized_its_training_rows(tmp_path):
    """The gate decides whether a local model grades every future eval, so it has
    to be a gate a memorizer cannot clear: replayed on the rows it trained on, one
    reports perfect agreement having generalized nothing."""
    truth = _seed_judge_log(tmp_path)
    export_engine_bundle(tmp_path, "judge")
    trained = _dataset_prompts(tmp_path)

    memorizer = _Replayer(lambda p: truth[p] if p in trained else not truth[p])
    res = prove_engine(tmp_path, "judge", memorizer, threshold=0.9)

    assert res.samples >= 3                                   # a real population, not a token one
    assert not (set(memorizer.asked) & trained)               # nothing it was trained on
    assert res.agreement == 0.0 and res.passes is False

    # A candidate that answers the unseen questions too still clears the gate.
    generalizer = _Replayer(lambda p: truth[p])
    assert prove_engine(tmp_path, "judge", generalizer, threshold=0.9).passes is True


def test_prove_gate_refuses_when_nothing_can_be_held_out(tmp_path):
    """Too few logged rows to hold any back is "cannot judge", never a pass — the
    alternative is a gate that certifies a model on its own training set."""
    _seed_judge_log(tmp_path, n=4)
    export_engine_bundle(tmp_path, "judge")

    with pytest.raises(ValueError, match="did not train on"):
        prove_engine(tmp_path, "judge", _Replayer(lambda p: True), threshold=0.9)


def test_prove_gate_needs_the_bundle_that_says_what_was_trained_on(tmp_path):
    """dataset.jsonl records what the candidate saw; without it the gate cannot
    know which rows are unseen, and guessing would let training rows be scored."""
    _seed_judge_log(tmp_path)

    with pytest.raises(ValueError, match="bundle"):
        prove_engine(tmp_path, "judge", _Replayer(lambda p: True), threshold=0.9)


def test_prove_gate_asks_each_logged_question_once(tmp_path):
    """`judge_passes` self-consistency grades one criterion with several identical
    calls, and re-running `eval` logs an unchanged answer again — one question
    logged N times is one piece of evidence, not N."""
    truth = _seed_judge_log(tmp_path, n=40, copies=3)
    export_engine_bundle(tmp_path, "judge")

    rec = _Replayer(lambda p: truth[p])
    res = prove_engine(tmp_path, "judge", rec, threshold=0.9)

    assert len(rec.asked) == len(set(rec.asked))
    assert res.samples == len(rec.asked) and res.population == 40


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
    from ai_calibrator.store import load_project
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


# --- human ground truth from judge-check labels --------------------------------

def _seed_labeled_project(tmp_path):
    """Project + scorecard + one human label that CONTRADICTS the logged judge."""
    from ai_calibrator.eval import JUDGE_SYSTEM, judge_prompt, save_scorecard
    from ai_calibrator.judge_check import save_labels
    from ai_calibrator.models import BehaviorSpec, CriterionResult, EvalCriterion, Scorecard, TestCase, TestResult, Weight

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="cites the policy", weight=Weight.HIGH)])
    p.tests = [TestCase(id="t1", input="can I return this?", expects=["c1"])]
    save_project(p, tmp_path)

    # Stamp the content hash the way `run_eval` does, so this fixture is a run
    # THIS version recorded — a hand-built card with no hash is a pre-field
    # scorecard and silently takes the back-compat path instead.
    from ai_calibrator.models import test_input_hash
    card = Scorecard(run_id="run-0001", results=[TestResult(
        test_id="t1", output="the answer", input_hash=test_input_hash(p.tests[0]),
        criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])])   # judge said PASS
    save_scorecard(tmp_path, card)
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c1", "passed": False}])  # human: FAIL

    # the logged judge row asks the exact same single-criterion question (conflict)
    prompt = judge_prompt("can I return this?", "the answer", [("c1", "cites the policy")])
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "judge.jsonl").write_text(json.dumps({
        "role": "judge", "prompt": prompt, "system": JUDGE_SYSTEM,
        "output": {"results": [{"criterion_id": "c1", "passed": True, "score": 1.0, "rationale": "judge"}]},
    }) + "\n")
    return prompt


def test_human_judge_rows_built_from_labels(tmp_path):
    from ai_calibrator.train_engine import human_judge_rows

    expected_prompt = _seed_labeled_project(tmp_path)
    rows = human_judge_rows(tmp_path)
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    assert msgs[1]["content"] == expected_prompt            # byte-identical judge format
    target = json.loads(msgs[2]["content"])
    assert target["results"][0]["passed"] is False          # the HUMAN verdict, not the judge's


def test_export_bundle_ground_truth_overrides_conflicting_log_row(tmp_path):
    expected_prompt = _seed_labeled_project(tmp_path)
    result = export_engine_bundle(tmp_path, "judge")
    assert result.examples == 1 and result.human_examples == 1   # log row dropped, human row kept

    row = json.loads((tmp_path / "trained-engines" / "judge" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["messages"][1]["content"] == expected_prompt
    assert json.loads(row["messages"][2]["content"])["results"][0]["passed"] is False
    assert "ground-truth" in (tmp_path / "trained-engines" / "judge" / "README.md").read_text(encoding="utf-8")


def test_export_bundle_skips_stale_labels(tmp_path):
    """Labels for vanished tests/criteria are skipped, never mis-built."""
    from ai_calibrator.judge_check import save_labels

    _seed_labeled_project(tmp_path)
    save_labels(tmp_path, "run-0001", [
        {"test_id": "ghost", "criterion_id": "c1", "passed": True},   # test gone
        {"test_id": "t1", "criterion_id": "ghost", "passed": True},   # criterion gone
    ])
    result = export_engine_bundle(tmp_path, "judge")
    assert result.human_examples == 1  # still just the valid label


def test_human_judge_rows_skip_labels_whose_test_input_changed(tmp_path):
    """The saved run holds the ANSWER; the current suite holds the QUESTION.

    `compile` re-mints t1..tN, so joining them by id alone asks the model to
    grade an answer to a question that was never put to it — and then stamps a
    human's verdict on that invented pair as ground truth, which is the worst
    kind of training row: confidently mislabeled."""
    from ai_calibrator.models import TestCase
    from ai_calibrator.store import load_project, save_project
    from ai_calibrator.train_engine import human_judge_rows

    _seed_labeled_project(tmp_path)
    assert len(human_judge_rows(tmp_path)) == 1        # the honest pair, before the recompile

    # `compile` re-mints t1 onto a different question. The run's answer ("the
    # answer") was a reply to "can I return this?", not to this.
    p = load_project(tmp_path)
    p.tests = [TestCase(id="t1", input="what are your hours?", expects=["c1"])]
    save_project(p, tmp_path)

    rows = human_judge_rows(tmp_path)
    assert rows == []
    # Specifically: no row pairs the NEW question with the OLD answer.
    assert not any("what are your hours?" in m["content"]
                   for row in rows for m in row["messages"])


def _seed_labeled_provable_project(tmp_path, n=40):
    """`n` graded tests the judge passed and a human overturned, every call logged.

    Big enough for the split to hold rows back, so the gate has a population."""
    from ai_calibrator.compile import render_system_prompt
    from ai_calibrator.eval import judge_prompt, judge_system, save_scorecard
    from ai_calibrator.judge_check import save_labels
    from ai_calibrator.models import (BehaviorSpec, CriterionResult, EvalCriterion, Scorecard,
                                      TestCase, TestResult, Weight, test_input_hash)

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="cites the policy", weight=Weight.HIGH)])
    p.tests = [TestCase(id=f"t{i}", input=f"question {i}?", expects=["c1"]) for i in range(n)]
    save_project(p, tmp_path)

    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[
        TestResult(test_id=f"t{i}", output=f"answer {i}", input_hash=test_input_hash(p.tests[i]),
                   criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])
        for i in range(n)]))
    save_labels(tmp_path, "run-0001",
                [{"test_id": f"t{i}", "criterion_id": "c1", "passed": False} for i in range(n)])

    system = judge_system(render_system_prompt(p.spec))
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "judge.jsonl").write_text("".join(json.dumps({
        "role": "judge", "system": system, "schema": {"type": "object"},
        "prompt": judge_prompt(p.tests[i].input, f"answer {i}", [("c1", "cites the policy")]),
        "output": {"results": [{"criterion_id": "c1", "passed": True, "score": 1.0,
                                "rationale": "cloud"}]},
    }) + "\n" for i in range(n)), encoding="utf-8")


def test_prove_gate_scores_against_the_ground_truth_the_bundle_ships(tmp_path):
    """The bundle trains on the human's verdict, so the gate has to ask for the
    human's verdict: scored against the reading a human overturned, the candidate
    that learned the correction is failed for being right, and the one that
    reproduced the mistake is certified in its place."""
    _seed_labeled_provable_project(tmp_path)
    export_engine_bundle(tmp_path, "judge")

    corrected = prove_engine(tmp_path, "judge", _Replayer(lambda p: False), threshold=0.9)
    assert corrected.agreement == 1.0 and corrected.passes is True

    imitator = prove_engine(tmp_path, "judge", _Replayer(lambda p: True), threshold=0.9)
    assert imitator.agreement == 0.0 and imitator.passes is False


def test_newest_human_label_wins_over_the_one_it_retracted(tmp_path):
    """Re-labeling the same (test, criterion) is the owner CHANGING their verdict.
    Training on the first one stamps the reading they withdrew as ground truth,
    and `judge-check` always labels the newest run, so they cannot undo it."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.judge_check import save_labels
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult, test_input_hash
    from ai_calibrator.store import load_project
    from ai_calibrator.train_engine import human_judge_rows

    _seed_labeled_project(tmp_path)                      # run-0001: human says FAIL
    p = load_project(tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0002", results=[TestResult(
        test_id="t1", output="the answer", input_hash=test_input_hash(p.tests[0]),
        criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])]))
    save_labels(tmp_path, "run-0002", [{"test_id": "t1", "criterion_id": "c1", "passed": True}])

    rows = human_judge_rows(tmp_path)
    assert len(rows) == 1
    assert json.loads(rows[0]["messages"][2]["content"])["results"][0]["passed"] is True

    # …and the logged judge row is patched with the verdict that stands, not the
    # one it replaced.
    export_engine_bundle(tmp_path, "judge")
    row = json.loads((tmp_path / "trained-engines" / "judge" / "dataset.jsonl")
                     .read_text(encoding="utf-8").splitlines()[0])
    assert json.loads(row["messages"][-1]["content"])["results"][0]["passed"] is True


def test_human_judge_rows_still_built_from_a_pre_hash_scorecard(tmp_path):
    """Back-compat: a scorecard written before the field records None, which
    means "unknown", never "matches" — those keep pairing by id as they always
    did, so upgrading does not silently empty an existing training set."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult
    from ai_calibrator.train_engine import human_judge_rows

    _seed_labeled_project(tmp_path)
    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[TestResult(
        test_id="t1", output="the answer",                        # no input_hash
        criteria=[CriterionResult(criterion_id="c1", passed=True, score=1.0)])]))

    assert len(human_judge_rows(tmp_path)) == 1




def test_agreement_survives_unhashable_criterion_id():
    """A judge output with a non-string criterion_id must not crash the gate.

    The verdict map keys on that id, so an unhashable one raised TypeError and
    aborted the whole prove-it comparison."""
    ref = [{"results": [{"criterion_id": "c1", "passed": True}]}]
    cand = [{"results": [{"criterion_id": ["c1"], "passed": True},
                         {"criterion_id": "c1", "passed": True}]}]
    assert agreement(ref, cand, role="judge") == 1.0




def test_engine_bundle_install_line_matches_the_trainer_it_ships(tmp_path):
    """Both bundles ship the same train.py, so the Engine-Trainer README has to
    name the same floors: SFTConfig(max_length=...) needs trl 1.x."""
    d = tmp_path / "logs"
    d.mkdir()
    (d / "judge.jsonl").write_text(json.dumps({"role": "judge", "prompt": "p", "output": "o"}) + "\n")
    export_engine_bundle(tmp_path, "judge")
    readme = (tmp_path / "trained-engines" / "judge" / "README.md").read_text(encoding="utf-8")
    assert '"trl>=1.0"' in readme and '"transformers>=4.56.2"' in readme




def test_train_engine_names_the_roles_that_actually_have_logs(tmp_path):
    """extractor/interviewer/predictor are trainable, but nothing wraps them, so
    "run eval and retry" would send the owner after data that never appears."""
    from ai_calibrator.train_engine import LOGGED_ROLES

    assert LOGGED_ROLES == {"judge", "compiler"}
    save_project(Project(name="p", goal="g"), tmp_path)
    r = runner.invoke(app, ["train-engine", "extractor", str(tmp_path)])
    assert r.exit_code == 1
    assert "Nothing records the extractor role" in r.output
    assert "then retry" not in r.output  # no false promise of data


def test_engine_bundle_install_line_matches_the_declared_trl_floor(tmp_path):
    """The bundle's own install line must not name a trl the emitted train.py
    rejects: it passes SFTConfig(max_length=...), which needs trl>=1.0."""
    save_project(Project(name="p", goal="g"), tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "judge.jsonl").write_text(
        json.dumps({"role": "judge", "prompt": "p", "output": "o"}) + "\n", encoding="utf-8")
    result = export_engine_bundle(tmp_path, "judge")
    readme = (Path(result.bundle_dir) / "README.md").read_text(encoding="utf-8")
    assert "trl>=1.0" in readme and "trl>=0.9" not in readme
    assert "pyyaml" in readme  # train.py reads recipe.yaml at run time


def test_regenerating_an_engine_bundle_keeps_hand_edited_hyperparameters(tmp_path):
    """The engine bundle writes its own recipe.yaml, so it needs the same
    protection the fine-tune bundle has: a re-export after `judge-check` labels
    land must not silently revert the knobs the user tuned for their GPU."""
    log = tmp_path / "logs"
    log.mkdir(parents=True)
    (log / "judge.jsonl").write_text(
        "".join(json.dumps({"system": "s", "prompt": f"q{i}", "output": f"a{i}"}) + "\n"
                for i in range(20)), encoding="utf-8")

    export_engine_bundle(tmp_path, "judge")
    recipe_file = tmp_path / "trained-engines" / "judge" / "recipe.yaml"
    tuned = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
    tuned["learning_rate"] = 7e-6
    tuned["lora_r"] = 64
    recipe_file.write_text(yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")

    export_engine_bundle(tmp_path, "judge")

    after = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
    assert after["learning_rate"] == 7e-6 and after["lora_r"] == 64


def test_one_bad_byte_does_not_brick_the_whole_tier(tmp_path):
    """A truncated multi-byte character — a kill mid-flush, a half-restored
    backup — made read_text raise. Every entrance to the Engine-Trainer goes
    through here, so the CLI answered with a traceback and the API with a 500
    until someone hand-repaired the file, with every good record behind the bad
    byte unreachable. `flywheel.read_feedback_lines` was hardened for this."""
    from ai_calibrator.train_engine import read_log

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    good = b'{"system": "s", "prompt": "q1", "output": "a1"}\n'
    bad = b'{"system": "s", "prompt": "caf\xc3", "output": "a2"}\n'
    (logs / "judge.jsonl").write_bytes(good + bad + good.replace(b"q1", b"q3"))

    rows = read_log(tmp_path, "judge")

    assert [r["prompt"] for r in rows] == ["q1", "caf�", "q3"]
