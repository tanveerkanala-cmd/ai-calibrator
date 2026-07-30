# Contributing

Thanks for considering it. This project has one mission — turn a non-technical
owner's knowledge and standards into a **tested, reliable** AI configuration —
and every change should serve it. Features that make the *testing* more
trustworthy are core; adjacent tooling (ops dashboards, generic LLM telemetry)
is out of scope.

## Dev setup

```bash
git clone https://github.com/tanveerkanala-cmd/ai-calibrator.git && cd ai-calibrator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[all,dev]'    # every engine + web UI + docs + the test/lint tools
```

The **full** suite needs `[all,dev]`: several test modules `importorskip` when
their optional library is absent, so on `[dev]` alone the entire RAG layer (plus
a few OpenAI / DOCX tests) is silently skipped rather than run. `[dev]` is fine
for a quick inner-loop run — just know it doesn't cover retrieval. (`[all]` pulls
a multi-GB ML stack via `rag`; skip it only if you're not touching those paths.)

No API keys are needed for development: the entire test suite runs against
deterministic fake engines. To exercise a real model locally, install
[Ollama](https://ollama.com) and point a role at `<model>@ollama`.

## Before you open a PR

```bash
ruff check .              # must be clean (pyflakes, bugbear, syntax)
mypy                      # must be clean — configured in pyproject.toml
pytest -q                 # must be all green — and runs in a few seconds
```

These three are what CI enforces (CI also runs the suite once with `[all,dev]`
so the retrieval/OpenAI/DOCX paths are exercised on every push, and once under
`LC_ALL=C` so an encoding assumption cannot hide behind a UTF-8 locale).

`mypy` is a gate, not advice. It reports zero errors today, and the value of
that is not tidiness: the checks it makes are exactly the ones that catch an
unguarded `project.spec` or a `None` reaching a formatter, and those cannot be
seen at all in a report that is already 76 errors long. If a genuinely
impossible case is the only way to satisfy it, narrow it with a real guard
rather than a cast — every one of those guards in this tree turned out to be a
reachable path, or is documented as unreachable where it is written.

Ground rules the codebase already follows — please keep them true:

- **Tests with teeth.** Every behavior change ships with a test that fails
  without the fix. Tests never call a network engine — use fake engines
  (see `tests/test_eval.py` for the pattern).
- **Every feature has four surfaces.** Core function (importable, engine-agnostic)
  → CLI command → API endpoint → docs entry in `docs/USAGE.md`. Docs drift is
  treated as a bug.
- **Friendly failures.** A user mistake (bad YAML, missing file, wrong flag) must
  produce an actionable message, never a traceback. The API returns 4xx, not 500,
  on bad input.
- **Durability.** All artifact writes go through `store.atomic_write_text` /
  `save_project`; project mutations hold `project_lock`.
- **Engine output is hostile.** Anything an LLM returns is coerced defensively
  (`coerce.as_list` / `as_str` …) — never trust shapes.
- **Don't invent claims.** README/docs state only what's built and verified.

## Project layout

`src/ai_calibrator/` — the package (src-layout). `models.py` holds the data
contracts (pydantic); `compile.py` / `eval.py` the core pipeline; one module per
verify feature (`lint.py`, `coverage.py`, `drift.py`, `snapshot.py`,
`judge_check.py`, `ci.py`, …); `cli.py` (typer) and `api.py` (FastAPI) are thin
wrappers over Core. Tests mirror modules in `tests/`.

## Reporting bugs

Open an issue with the minimal reproduction (the CLI command or Core call and
the observed vs expected output). For anything security-sensitive, see
[`SECURITY.md`](SECURITY.md).
