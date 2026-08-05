"""Eval-format interop — export the spec's tests + rubric to promptfoo.

Anti-lock-in: lets power users run calibrator's generated suite inside promptfoo
(a popular eval harness) instead of only `calibrate eval`. Deterministic — no
engine. The provider-agnostic system prompt becomes a promptfoo prompt, each test
an `input` var, and each expected eval criterion an assertion — a code-graded
`check` maps to promptfoo's own deterministic assertion for it, everything else
to `llm-rubric`. Edit the emitted `providers:` line to point at whatever model
you want to grade.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .compile import render_system_prompt
from .engines.base import parse_engine_spec
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


def _check_assert(check: Check) -> dict | None:
    """The promptfoo assertion equivalent to a deterministic ``check``, or None.

    `calibrate eval` grades a criterion carrying a check by code and never asks
    the judge (see :mod:`.checks`), so exporting it as an `llm-rubric` hands a
    hard pass/fail to a grader that can talk itself out of it — and drops the
    operand (the banned term, the length limit) from the file entirely.
    contains/not_contains are case-insensitive here, hence promptfoo's i-forms."""
    kind, value = check.kind, check.value
    if kind == "contains":
        return {"type": "icontains", "value": value}
    if kind == "not_contains":
        return {"type": "not-icontains", "value": value}
    if kind == "regex":
        # promptfoo's regex assertion is `new RegExp(value)` — JavaScript, not
        # Python. The dialects agree on a common subset and diverge outside it
        # (named groups, \\A, \\Z, inline flags, atomic groups): JS either throws
        # or, worse, matches something else, so the exported suite would grade a
        # different rule than `calibrate eval` does. Export only what both read
        # identically; anything else takes the honest downgrade the caller already
        # implements (llm-rubric, plus a NOTE naming the criterion).
        return {"type": "regex", "value": value} if _portable_regex(value) else None
    if kind == "non_empty":
        return {"type": "javascript", "value": "output.trim().length > 0"}
    if kind in ("max_chars", "min_chars"):
        try:
            limit = int(value.strip())
        except ValueError:
            return None  # not a usable limit; run_check fails it, promptfoo can't express it
        if limit < 0:
            return None
        return {"type": "javascript",
                "value": f"output.length {'<=' if kind == 'max_chars' else '>='} {limit}"}
    return None


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

    `#}` must be replaced LAST — the three openers above emit `}}` and never
    `#}`, so escaping it after them is safe while escaping it first is not.
    It has to be escaped at all because Nunjucks' lexer throws "unexpected end
    of comment" on a bare `#}`, which would make promptfoo refuse the file.
    """
    return (text
            .replace("{{", "{{ '{{' }}")
            .replace("{%", "{{ '{%' }}")
            .replace("{#", "{{ '{#' }}")
            .replace("#}", "{{ '#}' }}"))


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
        # Native assertions are left as authored. promptfoo renders the prompt,
        # `vars`, and an llm-rubric's value through Nunjucks — those are escaped
        # below. Whether it also renders a deterministic assertion's value is
        # not established here, and escaping one that is NOT rendered would make
        # the check search for the escape sequence instead of the owner's text.
        return native or {"type": "llm-rubric", "value": _nunjucks_literal(c.description)}

    for t in project.tests:
        # A multi-turn test's later turns are what the eval harness actually sends.
        # Exporting only the first turn would silently turn it into a different
        # (single-turn) test carrying the same assertions, so the promptfoo pass
        # rate would not mean what the calibrator's does. Omit and say so.
        if t.follow_ups:
            omitted.append(t.id)
            continue
        targeted = [cid for cid in (t.expects or list(crit)) if cid in crit]
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

    config = {
        "description": project.goal,
        "prompts": [body + "\n\n{{input}}"],
        "providers": [_provider_id(project.engines.subject)],
        "defaultTest": {"options": {"provider": _provider_id(project.engines.judge)}},
        "tests": tests,
    }
    out = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100)
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
