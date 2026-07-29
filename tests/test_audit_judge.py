"""Judge ground truth, engine truncation limits, key detection, training floors.

These pin behavior that only shows up at the seams: a human correction meeting a
multi-criterion judge call, an output budget meeting a provider's own ceiling, a
real API key meeting a placeholder heuristic, and a generated trainer meeting the
library version it was written for.
"""

import json

import pytest

from ai_calibrator.judge_check import judge_agreement, load_labels, save_labels
from ai_calibrator.models import (
    BehaviorSpec,
    Check,
    CriterionResult,
    EvalCriterion,
    Project,
    Scorecard,
)
from ai_calibrator.models import TestCase as Case
from ai_calibrator.models import TestResult as Result
from ai_calibrator.store import save_project
from ai_calibrator.train_engine import export_engine_bundle, human_judge_rows


# --- human ground truth vs. the judge call it corrects -----------------------

def _seed_two_criterion_project(tmp_path, *, check=None, with_log=True):
    """Project whose one test is graded on TWO criteria, both passed by the judge.

    Returns the prompt the logged judge row asked (all criteria in one call, the
    way `eval` grades)."""
    from ai_calibrator.eval import JUDGE_SYSTEM, judge_prompt, save_scorecard

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="is polite"),
        EvalCriterion(id="c2", description="cites a policy", check=check),
    ])
    p.tests = [Case(id="t1", input="can I return this?", expects=["c1", "c2"])]
    save_project(p, tmp_path)

    save_scorecard(tmp_path, Scorecard(run_id="run-0001", results=[Result(
        test_id="t1", output="the answer", criteria=[
            CriterionResult(criterion_id="c1", passed=True, score=1.0),
            CriterionResult(criterion_id="c2", passed=True, score=1.0)])]))

    prompt = judge_prompt("can I return this?", "the answer",
                          [("c1", "is polite"), ("c2", "cites a policy")])
    if with_log:
        (tmp_path / "logs").mkdir(exist_ok=True)
        (tmp_path / "logs" / "judge.jsonl").write_text(json.dumps({
            "role": "judge", "prompt": prompt, "system": JUDGE_SYSTEM,
            "output": {"results": [
                {"criterion_id": "c1", "passed": True, "score": 1.0, "rationale": "polite"},
                {"criterion_id": "c2", "passed": True, "score": 1.0, "rationale": "cites policy"}]},
        }) + "\n", encoding="utf-8")
    return prompt


def _dataset(tmp_path):
    text = (tmp_path / "trained-engines" / "judge" / "dataset.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _verdicts(row):
    return {r["criterion_id"]: r["passed"]
            for r in json.loads(row["messages"][-1]["content"])["results"]}


def test_ground_truth_overrides_the_verdict_it_corrects_in_a_multi_criterion_row(tmp_path):
    """The judge grades every criterion of a test in ONE call, so a correction has
    to rewrite that call's verdict — appending a second row leaves the dataset
    teaching both the human's answer and the one they overturned."""
    prompt = _seed_two_criterion_project(tmp_path)
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c2", "passed": False}])

    result = export_engine_bundle(tmp_path, "judge")
    rows = _dataset(tmp_path)
    assert [row["messages"][-2]["content"] for row in rows] == [prompt]  # the shape eval emits
    assert _verdicts(rows[0]) == {"c1": True, "c2": False}               # c1 undisputed, c2 corrected
    assert result.examples == 1 and result.human_examples == 1


def test_ground_truth_for_an_unlogged_question_becomes_its_own_row(tmp_path):
    """A label the logs cannot answer still has to reach the dataset."""
    _seed_two_criterion_project(tmp_path, with_log=False)
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c2", "passed": False}])

    result = export_engine_bundle(tmp_path, "judge")
    rows = _dataset(tmp_path)
    assert result.examples == 1 and result.human_examples == 1
    assert _verdicts(rows[0]) == {"c2": False}
    assert "- c2: cites a policy" in rows[0]["messages"][-2]["content"]


def test_a_code_checked_criterion_gets_no_new_row_but_still_corrects_its_logged_one(tmp_path):
    """A criterion with a deterministic check is graded by code, so the judge is
    never asked it again and no standalone row should teach it.

    The logged call from BEFORE the check was attached still exists in the
    dataset, though, and the human overturned it — dropping the label outright
    would let attaching a check silently restore the judgment they overruled."""
    _seed_two_criterion_project(tmp_path, check=Check(kind="contains", value="policy"))
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c2", "passed": False}])

    assert human_judge_rows(tmp_path) == []          # nothing new invented

    result = export_engine_bundle(tmp_path, "judge")
    assert result.human_examples == 1                # the logged row was corrected
    rows = _dataset(tmp_path)
    assert _verdicts(rows[0]) == {"c1": True, "c2": False}   # c1 untouched, c2 overturned


def test_a_code_checked_criterion_with_no_logged_call_adds_nothing(tmp_path):
    """With nothing to correct, the label produces no training data at all."""
    _seed_two_criterion_project(tmp_path, check=Check(kind="contains", value="policy"),
                                with_log=False)
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c2", "passed": False}])

    assert human_judge_rows(tmp_path) == []
    assert export_engine_bundle(tmp_path, "judge").human_examples == 0


# --- a verdict is a boolean, whoever supplied it -----------------------------

def _passed_card():
    return Scorecard(run_id="r", results=[Result(test_id="t1", output="o", criteria=[
        CriterionResult(criterion_id="c1", passed=True, rationale="ok")])])


@pytest.mark.parametrize("verdict", ["false", "no", "FAIL"])
def test_a_string_verdict_in_a_label_reads_as_a_fail(verdict):
    """`bool("false")` is True, so a verdict that arrives as a string flipped a
    human FAIL into agreement — in the safest-looking direction."""
    ag = judge_agreement(_passed_card(), [{"test_id": "t1", "criterion_id": "c1", "passed": verdict}])
    assert (ag.total, ag.agreed) == (1, 0)
    assert ag.disagreements[0]["human"] is False
    assert ag.unreliable_criteria() == ["c1"]


def test_a_string_verdict_is_persisted_as_a_boolean(tmp_path):
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c1", "passed": "false"}])
    assert load_labels(tmp_path, "run-0001") == [
        {"test_id": "t1", "criterion_id": "c1", "passed": False}]


def test_a_string_verdict_trains_the_verdict_it_states(tmp_path):
    _seed_two_criterion_project(tmp_path, with_log=False)
    save_labels(tmp_path, "run-0001", [{"test_id": "t1", "criterion_id": "c2", "passed": "false"}])

    target = json.loads(human_judge_rows(tmp_path)[0]["messages"][-1]["content"])
    assert target["results"][0]["passed"] is False and target["results"][0]["score"] == 0.0


# --- output budgets a provider will actually accept --------------------------

def _anthropic_engine(max_tokens, messages):
    """The adapter without __init__ — the anthropic SDK is an optional extra."""
    from ai_calibrator.engines.anthropic import AnthropicEngine

    eng = AnthropicEngine.__new__(AnthropicEngine)
    eng.name, eng.model, eng.max_tokens = "claude-x@anthropic", "claude-x", max_tokens
    eng._client = type("Client", (), {"messages": messages})()
    return eng


class _Truncated:
    """A response that stopped because it ran out of output budget."""

    class Block:
        type = "text"
        text = "partial"

    stop_reason = "max_tokens"
    content = [Block()]

    def create(self, **kwargs):
        return self


def test_the_nonstreaming_cap_is_the_one_the_sdk_enforces():
    """The SDK refuses a non-streaming request whose max_tokens implies more than
    10 minutes of generation, so the cap has to sit exactly under that line."""
    from ai_calibrator.engines.anthropic import MAX_NONSTREAMING_TOKENS

    assert 3600 * MAX_NONSTREAMING_TOKENS / 128_000 <= 600
    assert 3600 * (MAX_NONSTREAMING_TOKENS + 1) / 128_000 > 600


def test_max_tokens_knob_cannot_exceed_the_nonstreaming_cap(monkeypatch):
    """The truncation message told users to set 32000, which the SDK rejects
    outright — the remedy the tool prints must leave it able to make a call."""
    from ai_calibrator.engines.anthropic import MAX_NONSTREAMING_TOKENS, _default_max_tokens

    monkeypatch.setenv("CALIBRATOR_ANTHROPIC_MAX_TOKENS", "32000")
    assert _default_max_tokens() == MAX_NONSTREAMING_TOKENS


def test_truncation_advice_names_a_limit_the_sdk_accepts():
    from ai_calibrator.engines.anthropic import DEFAULT_MAX_TOKENS, MAX_NONSTREAMING_TOKENS

    eng = _anthropic_engine(DEFAULT_MAX_TOKENS, _Truncated())
    with pytest.raises(RuntimeError) as exc:
        eng.complete("hi")
    suggested = int(str(exc.value).split("CALIBRATOR_ANTHROPIC_MAX_TOKENS=")[1].split()[0])
    assert DEFAULT_MAX_TOKENS < suggested <= MAX_NONSTREAMING_TOKENS


def test_truncation_at_the_cap_advises_splitting_instead_of_raising_it():
    from ai_calibrator.engines.anthropic import MAX_NONSTREAMING_TOKENS

    eng = _anthropic_engine(MAX_NONSTREAMING_TOKENS, _Truncated())
    with pytest.raises(RuntimeError, match="split the input"):
        eng.complete("hi")


def test_a_rejected_output_budget_is_reported_as_an_engine_failure():
    """The SDK validates max_tokens BEFORE sending, with a bare ValueError — which
    escaped as a raw traceback and was reported as bad user input, not as an
    engine failure."""
    from ai_calibrator.engines.base import EngineError

    class Messages:
        def create(self, **kwargs):
            raise ValueError("Streaming is required for operations that may take "
                             "longer than 10 minutes.")

    with pytest.raises(EngineError, match="CALIBRATOR_ANTHROPIC_MAX_TOKENS"):
        _anthropic_engine(16000, Messages()).complete("hi")


# --- a cut-off answer is an error, not an answer -----------------------------

def _openai_engine(finish_reason, content, calls=None):
    from ai_calibrator.engines.openai import OpenAIEngine

    class Msg:
        refusal = None

    class Choice:
        message = Msg()

    Msg.content = content
    Choice.finish_reason = finish_reason
    resp = type("Resp", (), {"choices": [Choice()]})()

    class Completions:
        @staticmethod
        def create(**kwargs):
            if calls is not None:
                calls.append(kwargs)
            return resp

    eng = OpenAIEngine.__new__(OpenAIEngine)
    eng.name, eng.model = "gpt-x@openai", "gpt-x"
    eng._client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})()})()
    return eng


def test_openai_truncated_answer_is_an_error():
    """A length-truncated reply was handed back as a finished answer, so the judge
    graded — and the scorecard certified — half a sentence."""
    with pytest.raises(RuntimeError, match="truncated"):
        _openai_engine("length", "Here are your options. Diversify across ind").complete("hi")


def test_openai_truncated_answer_is_an_error_before_a_second_call():
    """On the schema path the cut JSON failed to parse, buying a second billed call
    and a "bind a stronger model" diagnosis that cannot fix an output budget."""
    calls = []
    eng = _openai_engine("length", '{"results": [{"criterion_id": "c1", "pas', calls)
    with pytest.raises(RuntimeError, match="truncated"):
        eng.complete("hi", schema={"type": "object"})
    assert len(calls) == 1


@pytest.mark.parametrize("finish_reason", [None, "stop", "tool_calls"])
def test_openai_returns_the_content_for_any_other_finish_reason(finish_reason):
    """OpenAI-compatible endpoints omit or repurpose the field; only "length"
    means the answer was cut."""
    assert _openai_engine(finish_reason, "a whole answer").complete("hi") == "a whole answer"


def test_ollama_truncated_answer_is_an_error(monkeypatch):
    import ai_calibrator.engines.ollama as ollama_mod
    from ai_calibrator.engines.ollama import OllamaEngine

    resp = type("Resp", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {"message": {"content": "half an ans"}, "done_reason": "length"},
    })()
    monkeypatch.setattr(ollama_mod.httpx, "post", lambda *a, **k: resp)
    with pytest.raises(RuntimeError, match="truncated"):
        OllamaEngine("gemma").complete("hi")


def test_ollama_returns_the_content_when_it_stopped_normally(monkeypatch):
    import ai_calibrator.engines.ollama as ollama_mod
    from ai_calibrator.engines.ollama import OllamaEngine

    resp = type("Resp", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: {"message": {"content": "a whole answer"}, "done_reason": "stop"},
    })()
    monkeypatch.setattr(ollama_mod.httpx, "post", lambda *a, **k: resp)
    assert OllamaEngine("gemma").complete("hi") == "a whole answer"


# --- a real credential must not be called a placeholder ----------------------

def test_a_key_with_a_hyphen_near_its_end_is_configured(monkeypatch):
    """Key bodies are base64url, an alphabet that contains "-", so judging only the
    last hyphen-delimited segment called ~9% of working keys placeholders."""
    from ai_calibrator.auth import anthropic_status, openai_status

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "a" * 86 + "-bcdefAA")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-" + "b" * 40 + "-cdefg")
    assert anthropic_status().configured is True
    assert openai_status().configured is True


@pytest.mark.parametrize("key", ["sk-ant-...", "sk-...", "<your-key>", "sk-ant-"])
def test_documented_placeholders_are_still_rejected(monkeypatch, key):
    from ai_calibrator.auth import anthropic_status

    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    assert anthropic_status().configured is False


# --- the generated trainer and the floors it is installed against ------------

def test_every_generated_install_line_names_the_transformers_the_trainer_needs(tmp_path):
    """`from_pretrained(dtype=...)` is only accepted from transformers 4.56 — on an
    older one the kwarg reaches the model constructor and raises TypeError, after
    the multi-GB base model has already downloaded."""
    from pathlib import Path

    from ai_calibrator.finetune import export_finetune, recommend_recipe, render_train_py

    assert "dtype=" in render_train_py(recommend_recipe(40))

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g")
    export_finetune(p, project_dir=tmp_path)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "judge.jsonl").write_text(
        json.dumps({"role": "judge", "prompt": "p", "output": "o"}) + "\n", encoding="utf-8")
    export_engine_bundle(tmp_path, "judge")

    written = [tmp_path / "finetune" / "README.md", tmp_path / "finetune" / "train.py",
               tmp_path / "finetune" / "merge.py",
               tmp_path / "trained-engines" / "judge" / "README.md",
               Path(__file__).resolve().parents[1] / "pyproject.toml"]
    for f in written:
        text = f.read_text(encoding="utf-8")
        assert "transformers>=" in text, f
        assert "transformers>=4.56.2" in text, f


def test_a_non_budget_value_error_does_not_advise_lowering_the_cap():
    """The SDK also raises ValueError when a proxy answers with a content type it
    cannot parse. Telling that operator to lower their token budget sends them
    after the wrong problem."""
    import pytest as _pytest

    from ai_calibrator.engines.base import EngineError

    class _Proxy:
        def create(self, **kwargs):
            raise ValueError("Expected JSON response, got text/html")

    eng = _anthropic_engine(16000, _Proxy())
    with _pytest.raises(EngineError) as exc:
        eng.complete("hi")

    assert "text/html" in str(exc.value)
    assert "CALIBRATOR_ANTHROPIC_MAX_TOKENS" not in str(exc.value)


def test_an_over_ceiling_env_override_is_clamped_out_loud(monkeypatch, capsys):
    """Silently honouring a smaller number than the operator set makes the next
    truncation inexplicable."""
    from ai_calibrator.engines.anthropic import MAX_NONSTREAMING_TOKENS, _default_max_tokens

    monkeypatch.setenv("CALIBRATOR_ANTHROPIC_MAX_TOKENS", "64000")
    assert _default_max_tokens() == MAX_NONSTREAMING_TOKENS
    assert "64000" in capsys.readouterr().err
