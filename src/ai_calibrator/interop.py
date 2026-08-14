"""Eval-format interop — export the spec's tests + rubric to promptfoo.

Anti-lock-in: lets power users run calibrator's generated suite inside promptfoo
(a popular eval harness) instead of only `calibrate eval`. Deterministic — no
engine. The provider-agnostic system prompt becomes a promptfoo prompt, each test
an `input` var, and each expected eval criterion an assertion — a code-graded
`check` maps to a deterministic assertion that grades it exactly as
:mod:`.checks` does, everything else to `llm-rubric`. Edit the emitted
`providers:` line to point at whatever model you want to grade.

The bargain of this module: whatever it emits must grade the same behavior the
same way `calibrate eval` does, and where the format cannot carry something
(a multi-turn test, a check promptfoo has no equivalent for, the system/user
split) the file says so in a header comment. A number measured here is only
worth reading if it means what the certificate means.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import yaml

from .compile import render_system_prompt
from .engines.base import parse_engine_spec
from .eval import conversation_prompt
from .models import Check, EvalCriterion, Project
from .store import atomic_write_text


def _provider_id(spec: str) -> str:
    """Map a ``model@provider`` spec to a promptfoo provider id (best effort)."""
    model, provider = parse_engine_spec(spec)
    return {
        "anthropic": f"anthropic:messages:{model}",
        "openai": f"openai:chat:{model}",
        "ollama": f"ollama:chat:{model}",
    }.get(provider, spec)


# Constructs Python's `regex`/`re` accept that JavaScript's RegExp does not read
# the same way. Conservative by design: a pattern exports only when NONE of these
# appear in it, so the failure mode is an honest downgrade to llm-rubric, never a
# rule that quietly means something different on the other side.
_PY_ONLY_REGEX = (
    "(?P", "(?#", "(?>", "(?(",      # named groups, comments, atomic, conditionals
    "\\A", "\\Z", "\\z",          # JS has ^ and $ only
    "(?i", "(?m", "(?s", "(?x",      # inline flags — JS takes flags as an argument
    "\\p{", "\\P{",                 # Unicode properties need the /u flag JS is not given
    "*+", "++", "?+", "}+",          # possessive quantifiers
    "\\h", "\\R", "\\K",           # `regex` module extensions
)


def _portable_regex(pattern: str) -> bool:
    """True if this pattern means the same thing to Python and to JavaScript."""
    return not any(token in pattern for token in _PY_ONLY_REGEX)


def _js_literal(value: str) -> str:
    """``value`` as a JavaScript string literal no template renderer will touch.

    promptfoo renders an assertion's ``value`` through Nunjucks before evaluating
    it, with ``process.env`` registered as a global — so a `{{ … }}` inside a
    banned term reads the operator's API keys, and a bare `{%` or `#}` throws in
    the lexer and takes the whole config with it. check values are model-authored
    (the compiler) or owner-authored (`add-check`), so neither is trustworthy
    input. `\\u007b`/`\\u0023` are the same characters to JavaScript and open
    nothing to the lexer. NFC here for the same reason :mod:`.checks` normalizes:
    the comparison must not turn on which spelling of a glyph was stored."""
    return (json.dumps(unicodedata.normalize("NFC", value))
            .replace("{", "\\u007b").replace("#", "\\u0023"))


def _check_assert(check: Check) -> dict | None:
    """The promptfoo assertion equivalent to a deterministic ``check``, or None.

    `calibrate eval` grades a criterion carrying a check by code and never asks
    the judge (see :mod:`.checks`), so exporting it as an `llm-rubric` hands a
    hard pass/fail to a grader that can talk itself out of it — and drops the
    operand (the banned term, the length limit) from the file entirely.

    The equivalence has to hold on the exact text a model emits, not just on
    ASCII: `run_check` compares NFC-normalized text and counts code points, and
    promptfoo's own assertions do neither (`icontains` is a plain
    ``toLowerCase().includes()``; JavaScript's ``.length`` counts UTF-16 units).
    Where that gap is reachable the rule is written out as JavaScript that
    mirrors `run_check` — an exported ban must not be walkable around by
    spelling é decomposed, and a length bound must not move because the answer
    contains an emoji."""
    kind, value = check.kind, check.value
    if kind in ("contains", "not_contains"):
        # Both sides normalized and lowercased, exactly as `run_check` does it.
        carries = f"output.normalize('NFC').toLowerCase().includes({_js_literal(value)}.toLowerCase())"
        return {"type": "javascript",
                "value": carries if kind == "contains" else f"!{carries}"}
    if kind == "regex":
        # promptfoo's regex assertion is `new RegExp(value)` — JavaScript, not
        # Python. The dialects agree on a common subset and diverge outside it
        # (named groups, \\A, \\Z, inline flags, atomic groups): JS either throws
        # or, worse, matches something else, so the exported suite would grade a
        # different rule than `calibrate eval` does. Export only what both read
        # identically; anything else takes the honest downgrade the caller already
        # implements (llm-rubric, plus a NOTE naming the criterion).
        # The pattern is escaped because promptfoo renders assertion values
        # through Nunjucks: a `\\{%.*%\\}` rule (the natural way to ban a leaked
        # template tag) is otherwise read as a block tag and the config throws.
        return {"type": "regex", "value": _nunjucks_literal(value)} if _portable_regex(value) else None
    if kind == "non_empty":
        return {"type": "javascript", "value": "output.trim().length > 0"}
    if kind in ("max_chars", "min_chars"):
        try:
            limit = int(value.strip())
        except ValueError:
            return None  # not a usable limit; run_check fails it, promptfoo can't express it
        if limit < 0:
            return None
        # `run_check` measures code points on NFC-normalized text. `output.length`
        # is UTF-16 units on whatever arrived, which is larger for every non-BMP
        # character and every decomposed accent — the exported bound would fail
        # answers the certified one passes (and pass short ones it fails).
        return {"type": "javascript",
                "value": f"Array.from(output.normalize('NFC')).length "
                         f"{'<=' if kind == 'max_chars' else '>='} {limit}"}
    return None


# Each delimiter becomes a Nunjucks expression that renders back to the exact two
# characters. The escapes carry no delimiter of their own: promptfoo wraps any
# value matching /({%[^%]*$|{{[^}]*$|{#[^#]*$)/m in `{% raw %}` before rendering
# it (its defense against a half-written tag), and a raw-wrapped value is
# delivered ESCAPE AND ALL — so `{{ '{%' }}`, which contains a `{%` with no
# closing `%` on the line, would reach the model instead of the `{%` it stands
# for. Nothing here emits a bare `{`, `%` or `#` next to a neighbouring character
# either, so two escapes side by side cannot form a new tag between them.
_ESCAPES = {
    "{{": "{{ '{{' }}",
    "{%": "{{ '{' }}{{ '%' }}",
    "{#": "{{ '{' }}{{ '#' }}",
    "#}": "{{ '#' }}{{ '}' }}",
}


def _nunjucks_literal(text: str) -> str:
    """Escape ``text`` so promptfoo renders it back byte for byte.

    promptfoo runs Nunjucks over every field it substitutes into — the prompt,
    each test's ``vars``, and an assertion's ``value`` — and registers
    ``process.env`` as a template global. So a `{{ ... }}` reaching ANY of those
    fields is not a rendering quirk: it reads the operator's environment,
    including their API keys, into text sent to a third-party model. The content
    here is untrusted by construction — ingested documents, model-authored
    criteria, and end-user feedback promoted to a test by `absorb`.

    Escaping the delimiters beats wrapping in `{% raw %}`: a raw block has a
    terminator to guess, Nunjucks accepts every spelling of it, and a value
    containing one would close the block early and get the remainder EXECUTED.
    An escaped delimiter has no terminator.

    `#}` is escaped too because Nunjucks' lexer throws "unexpected end of
    comment" on a bare one, which would make promptfoo refuse the file.

    One left-to-right pass, never repeated ``str.replace`` calls: replacing in
    passes lets an earlier escape's output be rewritten by a later one, and lets
    a delimiter the input never contained (`{` + `%` from two separate escapes)
    appear in the result.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        escaped = _ESCAPES.get(text[i:i + 2])
        if escaped is None:
            out.append(text[i])
            i += 1
        else:
            out.append(escaped)
            i += 2
    return "".join(out)


def to_promptfoo(project: Project) -> str:
    """Render a promptfoo config (YAML) from the project's spec + tests."""
    spec = project.spec
    if spec is None:
        raise ValueError("No spec — run `calibrate compile` (or `import`) first.")
    crit: dict[str, EvalCriterion] = {c.id: c for c in spec.eval_criteria}

    tests = []
    omitted: list[str] = []
    downgraded: list[str] = []   # checks promptfoo can't express → judged here instead

    def _assertion(c: EvalCriterion) -> dict:
        native = _check_assert(c.check) if c.check is not None else None
        if c.check is not None and native is None and c.id not in downgraded:
            downgraded.append(c.id)
        # Every field promptfoo substitutes into is rendered through Nunjucks —
        # the prompt, `vars`, and an assertion's `value`, deterministic ones
        # included — so each carries its operand escaped (`_check_assert` for the
        # checks, here for a rubric).
        return native or {"type": "llm-rubric", "value": _nunjucks_literal(c.description)}

    for t in project.tests:
        # A multi-turn test's later turns are what the eval harness actually sends.
        # Exporting only the first turn would silently turn it into a different
        # (single-turn) test carrying the same assertions, so the promptfoo pass
        # rate would not mean what the calibrator's does. Omit and say so.
        if t.follow_ups:
            omitted.append(t.id)
            continue
        # De-dup while preserving order, exactly as `run_eval` does: a repeated id
        # in `expects` (a hand-edited spec, or a compiler that listed one twice)
        # would otherwise emit the same assertion twice and score that criterion
        # twice, so the two tools would disagree about identical behavior.
        targeted = list(dict.fromkeys(cid for cid in (t.expects or list(crit)) if cid in crit))
        asserts = [_assertion(crit[cid]) for cid in targeted]
        if not asserts:  # no criteria → grade against the goal so the test still runs
            asserts = [{"type": "llm-rubric",
                        "value": _nunjucks_literal(f"Satisfies the goal: {project.goal}")}]
        tests.append({
            "description": t.id + (f" — {t.notes}" if t.notes else ""),
            # The test input is the most reliably attacker-reachable field in the
            # file: `absorb` promotes an end user's flagged message straight into
            # a pinned test, and this is where it lands.
            "vars": {"input": _nunjucks_literal(t.input)},
            "assert": asserts,
        })

    # `{{input}}` below is OURS and must stay a live tag — it is the only reason
    # promptfoo substitutes the test input at all. Everything the spec carries is
    # escaped: see `_nunjucks_literal`. ("Hi {{first_name}}" from a support macro
    # would otherwise render to nothing, and promptfoo would grade a prompt
    # `calibrate eval` never scored.)
    body = _nunjucks_literal(render_system_prompt(spec))
    # The user turn is encoded the way the harness encodes it (and the way
    # `calibrate run` serves it), so the answer graded here is an answer to the
    # question the certified pass rate was earned on. The system/user SPLIT
    # cannot be reproduced — a text prompt reaches the provider as one user
    # message — which is what the header comment below declares.
    turn = conversation_prompt([], "{{input}}")

    config = {
        "description": project.goal,
        "prompts": [body + "\n\n" + turn],
        "providers": [_provider_id(project.engines.subject)],
        "defaultTest": {"options": {"provider": _provider_id(project.engines.judge)}},
        "tests": tests,
    }
    out = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100)
    # Stated on every export, unlike the NOTEs below: this one is a property of the
    # format rather than of this project, and an unannounced difference in request
    # shape is what would quietly make a pass rate measured here not mean what the
    # certificate says. A NOTE stays reserved for "something about YOUR export was
    # degraded", so it keeps its signal.
    out = ("# Prompt shape: promptfoo delivers this as a SINGLE user message — the behavior and\n"
           "# the question in one turn, with no system role. `calibrate eval` sends the behavior\n"
           "# as a real system message, so a pass rate measured here is measured on a slightly\n"
           "# different request. The user turn itself is encoded exactly as the harness encodes it.\n"
           ) + out
    if downgraded:
        out = (f"# NOTE: {len(downgraded)} criterion/criteria carry a deterministic check promptfoo "
               f"cannot express ({', '.join(downgraded[:10])}"
               + (", …" if len(downgraded) > 10 else "") + ");\n"
               "# they are graded by the judge here and by code in `calibrate eval`.\n") + out
    if omitted:
        out = (f"# NOTE: {len(omitted)} multi-turn test(s) omitted — this export is single-turn "
               f"only ({', '.join(omitted[:10])}"
               + (", …" if len(omitted) > 10 else "") + ").\n"
               "# The pass rate here therefore covers fewer tests than `calibrate eval`.\n") + out
    return out


def export_promptfoo(project: Project, *, project_dir: str | Path) -> Path:
    """Write ``<project>/promptfooconfig.yaml`` (atomically)."""
    return atomic_write_text(Path(project_dir) / "promptfooconfig.yaml", to_promptfoo(project))
