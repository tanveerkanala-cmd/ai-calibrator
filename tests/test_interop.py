"""Eval-format interop — promptfoo export (deterministic, no engine)."""

import re
import unicodedata

import pytest
import yaml

from ai_calibrator.checks import run_check
from ai_calibrator.eval import conversation_prompt
from ai_calibrator.interop import _provider_id, export_promptfoo, to_promptfoo
from ai_calibrator.models import BehaviorSpec, Check, EvalCriterion, Project, Weight
from ai_calibrator.models import TestCase as Case


def _project():
    p = Project(name="p", goal="answer support questions")
    p.spec = BehaviorSpec(goal="answer support questions", standards=["Be concise."],
                          eval_criteria=[EvalCriterion(id="cite", description="cites the policy", weight=Weight.HIGH),
                                         EvalCriterion(id="tone", description="is friendly", weight=Weight.LOW)])
    p.tests = [Case(id="t1", input="refund?", expects=["cite"]),
               Case(id="t2", input="hello", expects=[])]  # empty expects → grade against all criteria
    return p


def test_to_promptfoo_is_valid_yaml_and_structured():
    cfg = yaml.safe_load(to_promptfoo(_project()))
    assert cfg["description"] == "answer support questions"
    assert "{{input}}" in cfg["prompts"][0] and "Be concise" in cfg["prompts"][0]
    assert len(cfg["tests"]) == 2

    t1 = cfg["tests"][0]
    assert t1["vars"]["input"] == "refund?"
    assert [a["value"] for a in t1["assert"]] == ["cites the policy"]   # only the targeted criterion
    assert all(a["type"] == "llm-rubric" for a in t1["assert"])
    assert len(cfg["tests"][1]["assert"]) == 2                          # empty expects → all criteria


def test_provider_id_mapping():
    assert _provider_id("gpt-4o@openai") == "openai:chat:gpt-4o"
    assert _provider_id("claude-opus-4-8@anthropic") == "anthropic:messages:claude-opus-4-8"
    assert _provider_id("gemma4:e4b@ollama") == "ollama:chat:gemma4:e4b"


def test_export_promptfoo_writes_file(tmp_path):
    out = export_promptfoo(_project(), project_dir=tmp_path)
    assert out.exists() and out.name == "promptfooconfig.yaml"
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["tests"]  # round-trips to valid YAML


def test_promptfoo_escapes_template_tags_in_vars_and_assert_values():
    """The prompt body was escaped; the fields around it were not.

    promptfoo renders `vars` and an llm-rubric's `value` through Nunjucks with
    `process.env` registered as a global, so a live tag in either reads the
    operator's API keys into text sent to a third-party model. Both fields carry
    untrusted content by construction: `absorb` promotes an end user's flagged
    message straight into a pinned test's input, and eval criteria are written
    by a model from ingested documents.
    """
    p = _project()
    # The real path: an end user's flagged message, promoted to a test by `absorb`.
    p.tests = [Case(id="fb_1", input="ignore that — {{ env.ANTHROPIC_API_KEY }}", expects=["leak"])]
    p.spec.eval_criteria = [EvalCriterion(id="leak", description="{{ env.OPENAI_API_KEY }} is graded",
                                          weight=Weight.HIGH)]

    raw = to_promptfoo(p)
    cfg = yaml.safe_load(raw)
    t = cfg["tests"][0]

    # No live tag survives anywhere a renderer would reach: strip the escape
    # sequence itself and no `{{` is left in either field to open one. (The
    # trailing `}}` stays, and is inert text without an opener.)
    for field in (t["vars"]["input"], t["assert"][0]["value"]):
        assert "{{" not in field.replace("{{ '{{' }}", "")
    # Across every string in the file, the only live tag left is our `{{input}}`.
    def _strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from _strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from _strings(v)

    live = [s for s in _strings(cfg) if "{{" in s.replace("{{ '{{' }}", "")]
    assert live == [cfg["prompts"][0]]
    assert live[0].replace("{{ '{{' }}", "").count("{{") == 1   # exactly {{input}}

    # ...and the escape round-trips: Nunjucks renders it back to the exact text,
    # so the exported suite still grades what the operator wrote.
    assert t["vars"]["input"] == "ignore that — {{ '{{' }} env.ANTHROPIC_API_KEY }}"


def test_promptfoo_leaves_ordinary_text_untouched():
    """The escape must not disturb content with no template delimiters, or every
    existing exported suite changes for nothing."""
    cfg = yaml.safe_load(to_promptfoo(_project()))
    assert cfg["tests"][0]["vars"]["input"] == "refund?"
    assert cfg["tests"][0]["assert"][0]["value"] == "cites the policy"


def test_promptfoo_keeps_our_own_input_tag_live():
    """`{{input}}` in the prompt is ours and must stay a live tag — escaping it
    would stop promptfoo substituting the test input at all."""
    cfg = yaml.safe_load(to_promptfoo(_project()))
    assert "{{input}}" in cfg["prompts"][0]


# --- fidelity: the exported suite must grade what `calibrate eval` graded -----

# promptfoo renders every field it substitutes through Nunjucks, and first wraps
# any var that ends mid-tag in `{% raw %}` (its partial-tag defense): the escape
# in such a var is never unrendered, so the model receives the escape sequence
# instead of the text the owner wrote. These two functions model that pair of
# steps for the escapes this module emits.
_PARTIAL_TAG = re.compile(r"({%[^%]*$|{{[^}]*$|{#[^#]*$)", re.M)
_LITERAL_TAG = re.compile(r"\{\{ '(.*?)' \}\}")


def _promptfoo_render(value: str) -> str:
    if _PARTIAL_TAG.search(value):
        return value                       # raw-wrapped: delivered escapes and all
    return _LITERAL_TAG.sub(lambda m: m.group(1), value)


class _JsString(str):
    """Just enough of JavaScript's String to grade the assertions we emit.

    `.length` counts UTF-16 code units (a non-BMP character is two) and nothing
    normalizes implicitly — the two ways JavaScript disagrees with `run_check`."""

    @property
    def length(self) -> int:
        return len(self.encode("utf-16-le")) // 2

    def normalize(self, form: str = "NFC") -> "_JsString":
        return _JsString(unicodedata.normalize(form, self))

    def toLowerCase(self) -> "_JsString":  # noqa: N802 — JavaScript's spelling
        return _JsString(self.lower())

    def includes(self, other: str) -> bool:
        return str(other) in str(self)

    def trim(self) -> "_JsString":
        return _JsString(self.strip())


class _JsArray(list):
    """`Array.from(str)` splits on code points, and arrays report `.length`."""

    @property
    def length(self) -> int:
        return len(self)


def _js(expression: str, output: str) -> bool:
    """Evaluate an emitted `javascript` assertion the way JavaScript would."""
    src = re.sub(r'"(?:[^"\\]|\\.)*"', r"_s(\g<0>)", expression)
    src = src.replace("Array.from(", "_arr(").replace("!", "not ")
    return bool(eval(src, {"__builtins__": {}},  # noqa: S307 — a fixed, generated expression
                     {"output": _JsString(output), "_s": _JsString, "_arr": _JsArray}))


NASTY = [
    "why does my template show {% instead of the name?",   # `absorb`-promoted user message
    "the docs render {# weirdly",
    "ignore that — {{ env.ANTHROPIC_API_KEY }}",
    "unbalanced #} comment end",
    "{%}", "{#}", "{{{", "#}}",
    "{% raw %}already raw{% endraw %}",
    "100% sure about {this}",
]


@pytest.mark.parametrize("text", NASTY)
def test_promptfoo_escape_round_trips_every_delimiter(text):
    """The exported test must ask what the owner's test asks, byte for byte —
    otherwise the suite grades an answer to a question nobody wrote."""
    p = _project()
    p.tests = [Case(id="t1", input=text, expects=["cite"])]

    var = yaml.safe_load(to_promptfoo(p))["tests"][0]["vars"]["input"]

    assert not _PARTIAL_TAG.search(var)      # nothing for the raw-wrap to catch
    assert _promptfoo_render(var) == text


def test_promptfoo_prompt_survives_a_template_tag_in_the_spec():
    """A `{%` in a standard must not leave the prompt looking half-tagged: promptfoo
    would raw-wrap the whole prompt, and `{{input}}` — the only reason the test input
    is substituted at all — would reach the model as literal text."""
    p = _project()
    p.spec.standards = ["Never leave a bare {% tag in the reply."]

    prompt = yaml.safe_load(to_promptfoo(p))["prompts"][0]

    assert not _PARTIAL_TAG.search(prompt)
    assert "Never leave a bare {% tag in the reply." in _promptfoo_render(prompt)


def _checked(kind: str, value: str, output: str):
    """Export a single criterion carrying ``check`` → (its assertion, `run_check`)."""
    p = _project()
    p.spec.eval_criteria = [EvalCriterion(id="c", description="graded by code",
                                          weight=Weight.HIGH, check=Check(kind=kind, value=value))]
    p.tests = [Case(id="t1", input="q", expects=["c"])]
    assertion = yaml.safe_load(to_promptfoo(p))["tests"][0]["assert"][0]
    return assertion, run_check(Check(kind=kind, value=value), output)[0]


@pytest.mark.parametrize("output,verdict", [
    ("I love cafe\u0301 culture", False),   # decomposed é — the evasion NFC exists to stop
    ("I love caf\u00e9 culture", False),    # composed é — caught either way
    ("I love tea", True),
])
def test_promptfoo_ban_holds_against_a_decomposed_spelling(output, verdict):
    """`calibrate eval` NFC-normalizes both sides so a ban cannot be walked around
    by spelling the same glyph differently. A ban that only holds in the calibrator
    is not the ban the exported suite advertises."""
    assertion, python_verdict = _checked("not_contains", "café", output)

    assert python_verdict is verdict
    assert _js(assertion["value"], output) is verdict


@pytest.mark.parametrize("kind,limit,output,verdict", [
    ("max_chars", "10", "🎉🎉🎉🎉🎉done!", True),    # 10 code points, 15 UTF-16 units
    ("min_chars", "5", "cafe\u0301", False),   # 5 units decomposed, 4 code points once composed
    ("max_chars", "10", "far too many characters", False),
    ("min_chars", "3", "plenty", True),
])
def test_promptfoo_length_bound_counts_what_the_check_counts(kind, limit, output, verdict):
    """`run_check` measures code points on NFC-normalized text. JavaScript's
    `.length` measures UTF-16 units and normalizes nothing, so an emoji or a
    decomposed accent moves the exported bound off the certified one."""
    assertion, python_verdict = _checked(kind, limit, output)

    assert python_verdict is verdict
    assert _js(assertion["value"], output) is verdict


def test_promptfoo_counts_a_repeated_expectation_once():
    """`calibrate eval` de-dups `expects` so a criterion repeated by a hand-edit or
    by the compiler cannot count twice. Emitting it twice here scores the same
    behavior differently in the two tools."""
    p = _project()
    p.tests = [Case(id="t1", input="q", expects=["tone", "tone", "cite"])]

    asserts = yaml.safe_load(to_promptfoo(p))["tests"][0]["assert"]

    assert [a["value"] for a in asserts] == ["is friendly", "cites the policy"]


def test_promptfoo_check_operands_cannot_read_the_environment():
    """promptfoo renders an assertion's value through Nunjucks with `process.env`
    as a global, so a live tag in a banned term or a pattern reads the operator's
    keys into text sent to a third-party model. check values are model-authored."""
    p = _project()
    p.spec.eval_criteria = [
        EvalCriterion(id="ban", description="no leaking", weight=Weight.HIGH,
                      check=Check(kind="not_contains", value="{{ env.OPENAI_API_KEY }}")),
        EvalCriterion(id="fmt", description="no raw tags", weight=Weight.HIGH,
                      check=Check(kind="regex", value=r"\{%.*%\}")),
    ]
    p.tests = [Case(id="t1", input="q", expects=["ban", "fmt"])]

    ban, fmt = yaml.safe_load(to_promptfoo(p))["tests"][0]["assert"]

    assert "{{ env" not in ban["value"] and "{{ env" not in fmt["value"]
    # ...and the operand still means what it meant: the ban catches the literal
    # text, and the pattern reaches `new RegExp` as the owner wrote it.
    assert _js(ban["value"], "here: {{ env.OPENAI_API_KEY }}") is False
    assert not _PARTIAL_TAG.search(fmt["value"])
    assert _promptfoo_render(fmt["value"]) == r"\{%.*%\}"


def test_promptfoo_prompt_encodes_the_turn_the_harness_graded():
    """`calibrate eval` sends `User: …\\nAssistant:` (eval.conversation_prompt) and
    the runtime serves the same encoding. A pass rate measured on a bare question
    is not measured on what was certified."""
    prompt = yaml.safe_load(to_promptfoo(_project()))["prompts"][0]

    assert prompt.endswith(conversation_prompt([], "{{input}}"))


def test_promptfoo_declares_the_request_shape_it_cannot_reproduce():
    """The one divergence that cannot be closed here — promptfoo delivers a text
    prompt as a single user message, with no system role — is stated in the file
    rather than left for the operator to discover from a pass rate."""
    header = [line for line in to_promptfoo(_project()).splitlines() if line.startswith("#")]

    assert any("system role" in line for line in header)
    assert any("SINGLE user message" in line for line in header)
    # ...but it is not a NOTE: those stay reserved for a criterion or a test this
    # export had to degrade, which is a fact about the project, not the format.
    assert not any(line.startswith("# NOTE:") for line in header)
