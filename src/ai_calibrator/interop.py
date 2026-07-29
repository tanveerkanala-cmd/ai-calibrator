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
        return {"type": "regex", "value": value}
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
        return native or {"type": "llm-rubric", "value": c.description}

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
            asserts = [{"type": "llm-rubric", "value": f"Satisfies the goal: {project.goal}"}]
        tests.append({
            "description": t.id + (f" — {t.notes}" if t.notes else ""),
            "vars": {"input": t.input},
            "assert": asserts,
        })

    # promptfoo renders prompts through Nunjucks — that is the only reason
    # `{{input}}` substitutes at all. Anything else the spec happens to contain
    # ("Hi {{first_name}}" from a support macro, a `{% if %}` block) would render
    # too, silently blanking it, so promptfoo would grade a prompt `calibrate
    # eval` never scored.
    #
    # Escape the three delimiters rather than wrapping the body in `{% raw %}`.
    # A raw block has a terminator to guess, and Nunjucks accepts every spelling
    # of it — `{%endraw%}`, `{%   endraw   %}`, tabs — so a spec containing one
    # closes the block early and the rest of the spec is EXECUTED as a template.
    # promptfoo registers process.env as a template global, so that is not merely
    # a rendering bug: it can read the operator's API keys into a prompt that is
    # then sent to a third-party model. Escaped delimiters have no terminator and
    # render back to the spec text byte for byte.
    #
    # `#}` needs escaping too, and must be replaced LAST. Nunjucks' lexer throws
    # "unexpected end of comment" on a `#}` found in template text — a spec that
    # merely mentions one would make promptfoo refuse to lex the prompt at all.
    # Last, because the three openers above emit `}}`, never `#}`, so escaping it
    # first would leave the openers' output untouched but escaping it after is
    # safe; reordering breaks that.
    body = (render_system_prompt(spec)
            .replace("{{", "{{ '{{' }}")
            .replace("{%", "{{ '{%' }}")
            .replace("{#", "{{ '{#' }}")
            .replace("#}", "{{ '#}' }}"))

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
