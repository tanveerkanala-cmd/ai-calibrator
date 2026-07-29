"""Edge inputs on the ingest / import / export paths.

What a materials scan is allowed to read, what survives extraction, what a
malformed import does to the CLI, and what the promptfoo export promises about
parity with `calibrate eval`. Deterministic — no engine, no network.
"""

from __future__ import annotations

import csv
import zipfile

import pytest
import yaml
from typer.testing import CliRunner

from ai_calibrator.cli import app
from ai_calibrator.examples_io import load_examples_report
from ai_calibrator.flywheel import absorb_feedback, append_feedback, read_feedback
from ai_calibrator.ingest import extract_gaps, parse_materials
from ai_calibrator.interop import to_promptfoo
from ai_calibrator.models import (
    BehaviorSpec,
    Check,
    EvalCriterion,
    Example,
    Project,
    TestCase,
    Weight,
)
from ai_calibrator.parsing import read_document

runner = CliRunner()


class FakeEngine:
    """Returns a canned structured payload — the extractor without a model."""

    name = "fake@test"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, prompt, *, system=None, schema=None):
        return self.payload


# --- materials scan --------------------------------------------------------

def test_ingest_skips_hidden_directories(tmp_path):
    """`rglob` yields a hidden directory's children as paths of their own, so a
    leaf-name test skips `.git` and then ingests `.git/config` — a remote URL
    with a token in it — into the spec, the index and every served prompt."""
    materials = tmp_path / "materials"
    (materials / ".git").mkdir(parents=True)
    (materials / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://user:s3cr3t-token@example.com/acme/private.git\n')
    (materials / ".env").write_text("API_KEY=s3cr3t-token")
    (materials / "policy.md").write_text("Refunds within 30 days.")

    docs, skipped = parse_materials(materials)

    assert [p.name for p, _ in docs] == ["policy.md"]
    assert not any("s3cr3t-token" in text for _, text in docs)
    assert skipped == []          # excluded by policy, not reported as a parse failure


def test_ingest_reads_a_source_dir_that_is_itself_hidden(tmp_path):
    """Only components BELOW the root are hidden-filtered: a materials dir living
    under a dotted path (`--source ~/.config/handbook`) still ingests."""
    materials = tmp_path / ".config" / "handbook"
    materials.mkdir(parents=True)
    (materials / "policy.md").write_text("Refunds within 30 days.")

    docs, _skipped = parse_materials(materials)

    assert [p.name for p, _ in docs] == ["policy.md"]


def test_binary_materials_are_reported_as_skipped(tmp_path):
    """A spreadsheet or an image decoded as UTF-8 is dense mojibake, which passes
    the corpus's non-empty check: it counts as ingested, its real content never
    reaches the extractor, and it can fill the extraction window on its own."""
    materials = tmp_path / "materials"
    materials.mkdir()
    with zipfile.ZipFile(materials / "prices.xlsx", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/worksheets/sheet1.xml", "<x>" + "cell " * 200 + "</x>")
    (materials / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8)
    (materials / "policy.md").write_text("Refunds within 30 days.")

    docs, skipped = parse_materials(materials)

    assert [p.name for p, _ in docs] == ["policy.md"]
    assert sorted(rel for rel, _ in skipped) == ["logo.png", "prices.xlsx"]
    assert all(reason.strip() for _, reason in skipped)   # the CLI prints these


def test_plain_text_materials_are_read_whatever_the_suffix(tmp_path):
    """Content decides, not the extension — an allowlist would silently drop a
    .jsonl export, a .tex draft, or an extensionless README."""
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "faq.jsonl").write_text('{"q": "returns?", "a": "within 30 days"}\n')
    (materials / "README").write_text("House style: plain English, no jargon.")

    docs, skipped = parse_materials(materials)

    assert sorted(p.name for p, _ in docs) == ["README", "faq.jsonl"]
    assert skipped == []
    assert "plain English" in dict((p.name, t) for p, t in docs)["README"]


def test_byte_order_marks_do_not_leak_into_material_text(tmp_path):
    """Notepad's "Unicode" .txt is UTF-16; read as UTF-8 it becomes NUL-interleaved
    characters neither the extractor nor the embedder can match on."""
    utf16 = tmp_path / "notes.txt"
    utf16.write_bytes("Refunds within 30 days.".encode("utf-16"))
    utf8_bom = tmp_path / "voice.md"
    utf8_bom.write_bytes(b"\xef\xbb\xbfWarm and plain.")

    assert read_document(utf16) == "Refunds within 30 days."
    assert read_document(utf8_bom) == "Warm and plain."


# --- extraction ------------------------------------------------------------

def test_extract_gaps_keeps_long_and_templated_facts():
    """Facts are whole sentences the materials state, not gap labels: a compound
    policy runs past a phrase-length ceiling, and a rule about a response macro
    quotes the placeholder. Grading them as labels deleted both, silently."""
    long_fact = ("Refunds are processed within 30 days of receipt provided the item is unworn, in "
                 "its original packaging, and accompanied by proof of purchase; items bought during "
                 "a promotional sale are final and cannot be returned or exchanged.")
    assert len(long_fact) > 200
    engine = FakeEngine({"facts": [long_fact,
                                   "Emails must open with 'Hi {first_name},'",
                                   "Escalate to a human agent within 4 hours --- no exceptions.",
                                   "We ship internationally."],
                         "gaps": []})

    facts, _gaps, _analyzed = extract_gaps("g", "support_assistant", [("policy.md", "t")], engine)

    assert facts == [long_fact,
                     "Emails must open with 'Hi {first_name},'",
                     "Escalate to a human agent within 4 hours --- no exceptions.",
                     "We ship internationally."]


def test_extract_gaps_still_drops_a_dumped_json_blob_as_a_fact():
    """The true-positive direction stays pinned: scaffolding a small local model
    leaked must never reach the facts."""
    engine = FakeEngine({"facts": ["We ship internationally.",
                                   '```jsonall-important-fields: [fact, gap]'],
                         "gaps": []})

    facts, _gaps, _analyzed = extract_gaps("g", "assistant", [("f.md", "t")], engine)

    assert facts == ["We ship internationally."]


# --- example import --------------------------------------------------------

def _broken_quoting_csv(path):
    """A CSV export with one unbalanced double-quote: the reader opens a field at
    the stray quote and swallows the rest of the file into it."""
    rows = ["input,good_output"]
    for i in range(4000):
        rows.append('Do you price match?,"He said hello and left' if i == 7
                    else f"question {i}?,answer {i} with some ordinary padding text here")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_unreadable_csv_import_is_a_friendly_error(tmp_path):
    """csv.Error is not a ValueError, so it escapes the CLI's handler."""
    f = _broken_quoting_csv(tmp_path / "tickets.csv")

    with pytest.raises(ValueError) as exc:
        load_examples_report(f)

    assert not isinstance(exc.value, csv.Error)
    assert "tickets.csv" in str(exc.value) and "double-quote" in str(exc.value)


def test_examples_import_of_unreadable_csv_prints_a_message_not_a_traceback(tmp_path):
    (tmp_path / "project.yaml").write_text(
        yaml.safe_dump({"name": "p", "goal": "g", "spec": {"goal": "g"}}))
    f = _broken_quoting_csv(tmp_path / "tickets.csv")

    result = runner.invoke(app, ["examples", str(tmp_path), "--import", str(f)])

    # Under CliRunner an escaping exception is captured rather than printed, so
    # checking the output for a traceback alone would pass while broken.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert "Traceback (most recent call last)" not in result.output
    assert "tickets.csv" in result.output


# --- flywheel --------------------------------------------------------------

def _feedback_project():
    p = Project(name="p", goal="answer return questions")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="on-policy", weight=Weight.HIGH)])
    return p


def test_absorb_keeps_feedback_pending_when_the_commit_fails(tmp_path):
    """The inbox is the only copy: emptying it before the caller has persisted the
    project it was folded into loses the verdict for good — the archive is an
    audit trail nothing reads back."""
    project = _feedback_project()
    append_feedback(tmp_path, {"turns": ["Can I return after 40 days?"], "output": "Sure, any time!",
                               "verdict": "down", "correction": "No — the window is 30 days."})

    def failing_save():
        raise OSError("No space left on device")

    with pytest.raises(OSError):
        absorb_feedback(project, tmp_path, commit=failing_save)

    assert len(read_feedback(tmp_path)) == 1     # still absorbable on the next run


def test_absorb_commits_before_emptying_the_inbox(tmp_path):
    project = _feedback_project()
    append_feedback(tmp_path, {"turns": ["q"], "output": "a", "verdict": "up"})
    pending_during_commit = []

    def save():
        pending_during_commit.append(len(read_feedback(tmp_path)))

    result = absorb_feedback(project, tmp_path, commit=save)

    assert result.ups == 1 and result.examples_added == 1
    assert pending_during_commit == [1]          # records outlive the save that folds them
    assert read_feedback(tmp_path) == []         # then, and only then, the inbox empties


# --- spec diff -------------------------------------------------------------

def test_diff_flags_a_retracted_training_example():
    """`finetune.assemble_dataset` builds the training set from spec.examples, so
    the same answer moving from golden to flagged changes the shipped model. Under
    "no behavior change" that retraction would ship unreviewed."""
    from ai_calibrator.specdiff import diff_dict, diff_specs

    answer = "Yes, refunds are unlimited."
    before = BehaviorSpec(goal="g", examples=[
        Example(input="Can I return after 40 days?", good_output=answer, source="human")])
    after = BehaviorSpec(goal="g", examples=[
        Example(input="Can I return after 40 days?", bad_output=answer,
                why="invented policy", source="human_ratified")])

    d = diff_specs(before, after)

    assert d.changed
    assert len(d.examples_added) == 1 and len(d.examples_removed) == 1
    assert answer in d.examples_removed[0] and answer in d.examples_added[0]
    assert diff_dict(d)["examples"]["removed"] == d.examples_removed
    assert not diff_specs(before, before.model_copy(deep=True)).changed


# --- promptfoo export ------------------------------------------------------

def _checked_project():
    p = Project(name="p", goal="answer return questions")
    p.spec = BehaviorSpec(goal="answer return questions", eval_criteria=[
        EvalCriterion(id="brief", description="is brief", weight=Weight.HIGH,
                      check=Check(kind="max_chars", value="200")),
        EvalCriterion(id="nolegal", description="does not promise a guarantee", weight=Weight.HIGH,
                      check=Check(kind="not_contains", value="guarantee")),
        EvalCriterion(id="tone", description="is friendly", weight=Weight.LOW)])
    p.tests = [TestCase(id="t1", input="refund?", expects=["brief", "nolegal", "tone"])]
    return p


def test_promptfoo_export_keeps_deterministic_checks_deterministic():
    """`calibrate eval` grades a criterion with a check by code and never calls the
    judge. Exporting it as an llm-rubric hands a hard fail to a grader that can
    talk itself out of it, and drops the operand (the limit, the banned term)."""
    cfg = yaml.safe_load(to_promptfoo(_checked_project()))

    asserts = cfg["tests"][0]["assert"]
    assert {"type": "not-icontains", "value": "guarantee"} in asserts
    assert {"type": "javascript", "value": "output.length <= 200"} in asserts
    assert {"type": "llm-rubric", "value": "is friendly"} in asserts   # no check → judged


def test_promptfoo_export_notes_a_check_it_cannot_express():
    """A downgrade to the judge changes what the pass rate means, so it is
    announced — the same rule the multi-turn omission already follows."""
    p = _checked_project()
    p.spec.eval_criteria[0].check = Check(kind="max_chars", value="not a number")

    out = to_promptfoo(p)

    assert out.startswith("# NOTE:") and "brief" in out.splitlines()[0]
    asserts = yaml.safe_load(out)["tests"][0]["assert"]
    assert {"type": "llm-rubric", "value": "is brief"} in asserts


def test_promptfoo_prompt_preserves_template_syntax_in_the_spec():
    """promptfoo renders prompts through Nunjucks — the only reason `{{input}}`
    substitutes. A response macro quoted in the spec would render away too, so
    promptfoo would grade a prompt `calibrate eval` never scored."""
    p = _checked_project()
    p.spec.format = ('Use the macro exactly: "Hi {{first_name}}, your order {{order_id}} ships '
                     '{% if rush %}today{% endif %}."')

    prompt = yaml.safe_load(to_promptfoo(p))["prompts"][0]

    body = prompt.split("{% raw %}", 1)[1].split("{% endraw %}", 1)[0]
    assert "{{first_name}}" in body and "{% if rush %}" in body
    # …and the one construct that must still render sits outside the raw block.
    assert prompt.rsplit("{% endraw %}", 1)[1].strip() == "{{input}}"


# --- generated-artifact tokens ---------------------------------------------

def test_safe_token_rejects_a_leading_dash():
    """The token is interpolated into command lines a human is told to run
    ("ollama pull <token>"), where "-rf" reads as an option, not a model."""
    from ai_calibrator.coerce import safe_token

    for bad in ("-rf", "--output-dir", "-"):
        with pytest.raises(ValueError, match="must not start with"):
            safe_token(bad, "base model")
    assert safe_token("Qwen/Qwen2.5-7B-Instruct", "base model") == "Qwen/Qwen2.5-7B-Instruct"


# --- the retrieval index never outlives its source --------------------------

class _NoEngine:
    """extract_gaps is never reached on an empty corpus; fail loudly if it is."""

    def complete(self, prompt, system=None, schema=None):  # pragma: no cover - guard
        raise AssertionError("the engine must not be called for an empty corpus")


def test_no_index_still_drops_an_index_whose_materials_are_gone(tmp_path, monkeypatch):
    """`--no-index` skips the expensive rebuild. It cannot mean "keep serving text
    whose source file was deleted" — that index feeds every eval and run."""
    from ai_calibrator import ingest as ingest_mod

    dropped, built = [], []
    monkeypatch.setattr(ingest_mod.rag, "drop_index", lambda d: dropped.append(d) or True)
    monkeypatch.setattr(ingest_mod.rag, "build_index", lambda d, c: built.append(d) or len(c))

    empty = tmp_path / "materials"
    empty.mkdir()
    project = Project(name="p", goal="g")
    result = ingest_mod.ingest_project(project, empty, _NoEngine(),
                                       project_dir=tmp_path, build_index=False)

    assert dropped == [tmp_path]
    assert built == []
    assert result.indexed == 0


def test_no_index_leaves_a_live_index_alone_when_materials_remain(tmp_path, monkeypatch):
    """The other half of the contract: with materials still there, --no-index
    rebuilds nothing and destroys nothing."""
    from ai_calibrator import ingest as ingest_mod

    dropped, built = [], []
    monkeypatch.setattr(ingest_mod.rag, "drop_index", lambda d: dropped.append(d) or True)
    monkeypatch.setattr(ingest_mod.rag, "build_index", lambda d, c: built.append(d) or len(c))
    monkeypatch.setattr(ingest_mod, "extract_gaps", lambda *a, **k: ([], [], 1))

    mats = tmp_path / "materials"
    mats.mkdir()
    (mats / "faq.txt").write_text("Refunds are issued within 30 days.", encoding="utf-8")
    project = Project(name="p", goal="g")
    result = ingest_mod.ingest_project(project, mats, _NoEngine(),
                                       project_dir=tmp_path, build_index=False)

    assert dropped == [] and built == []
    assert result.indexed is None
