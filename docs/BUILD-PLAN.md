# v0 Build Plan (historical)

> **Status: completed.** This is the original v0 roadmap, kept as a record of
> how the project was sequenced and why. Everything below — including the
> items marked "later" (cloud adapters, the API, the web UI) and the v1
> Advanced tier — has since shipped. For what the tool does *today*, see
> [`USAGE.md`](USAGE.md).

Companion to `ARCHITECTURE.md`. This is the *map*: what to build, in what order,
and how the pieces fit. v0 = **Guided mode**, local-first, Core + CLI (UI later).

## Scope of v0

The full Guided loop, runnable from a terminal:

```
init → ingest → interview → compile → eval → refine → export
```

Local engine (Ollama) by default; cloud engines are opt-in adapters added later.
No fine-tuning in v0 (that's the Advanced tier, v1). Desktop UI sits on top of
the Core once the Core works.

## Repo layout

```
ai-calibrator/
  pyproject.toml
  README.md
  docs/            ARCHITECTURE.md, BUILD-PLAN.md
  src/calibrator/
    __init__.py
    models.py        # data contracts (the spine — everything imports this)
    engines/
      base.py        # Engine interface + role registry + factory
      ollama.py      # local default engine
      # anthropic.py / openai.py — opt-in cloud adapters (later)
    store.py         # load/save a Project to a project directory   [M1]
    ingest.py        # parse + index materials, extract facts/gaps   [M1]
    interview.py     # gap-driven adaptive questions + ratify        [M2]
    compile.py       # answers + materials -> BehaviorSpec + artifacts[M3]
    eval.py          # deterministic checks + LLM-judge + scorecard  [M4]
    pipeline.py      # orchestrates the stages                       [M4]
    cli.py           # `calibrate` entrypoint
  tests/
  # later: api/ (FastAPI), desktop/ (Tauri + React)
```

## The spine: data contracts (`models.py`)

Everything flows through typed models so stages stay decoupled and the project
is just serializable data on disk:

`Project` → holds `Material[]`, `Gap[]`, `InterviewItem[]`, `BehaviorSpec`,
`TestCase[]`, `EngineBinding`. `BehaviorSpec` is the source of truth; artifacts
(system prompt, RAG config, rubric, tests) compile from it. `Scorecard` holds
`TestResult[] → CriterionResult[]`.

## Milestones (each ends with something runnable)

- **M0 — Foundation.** Repo, `pyproject`, `models.py`, the
  `Engine` interface + `OllamaEngine`, a runnable `cli.py` (`init`, `status`,
  stubs). *Done when:* `calibrate init` creates a project and `calibrate status`
  reads it; `OllamaEngine.complete()` talks to a local model.
- **M1 — Ingest.** `store.py` (load/save Project) + `ingest.py`: parse docs,
  embed into LanceDB, run the Extractor to produce facts + a gap list. *Done
  when:* `calibrate ingest ./docs` populates `materials` + `gaps`.
- **M2 — Interview.** Gap-driven question generation + propose-and-ratify; only
  high-information questions; stores rationale (teach-while-scaffolding). *Done
  when:* `calibrate interview` fills `interview[]` from the gaps.
- **M3 — Compile.** Synthesize the `BehaviorSpec`, then compile system prompt +
  RAG config + rubric + test cases. *Done when:* `calibrate compile` writes the
  artifact bundle.
- **M4 — Eval + Refine loop.** Deterministic checks + LLM-as-judge → `Scorecard`;
  diagnose failures → propose fixes → re-run. *Done when:* `calibrate eval`
  produces a scorecard and the loop improves the pass rate.
- **M5 — Export.** Emit a portable bundle + an Ollama `Modelfile` so the result
  runs in the local-first runtime.
- **M6 — Local API + Desktop UI.** FastAPI over the Core; Tauri + React shell.
- **v1 — Advanced tier.** The fine-tuning toolchain (§3.1).

## Build order rationale

M0–M1 are coherence-critical and were built sequentially — the data contracts
and store shape everything downstream. From M2 onward the modules are
independent (interview, compile, eval each sit behind the same contracts), so
they could be developed and reviewed in parallel.

## Testing

`pytest`. Each module gets unit tests against the contracts; engines are mocked
so tests don't need a live model. An end-to-end test runs the whole loop against
a tiny fixture project with a stub engine.
