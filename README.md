# AI Calibrator

[![ci](https://github.com/tanveerkanala-cmd/ai-calibrator/actions/workflows/ci.yml/badge.svg)](https://github.com/tanveerkanala-cmd/ai-calibrator/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-calibrator)](https://pypi.org/project/ai-calibrator/)

Turn your knowledge and standards into a **tested, reliable AI** — without
writing prompts, code, or datasets. You bring your materials and answer
questions; the tool builds, tests, and *proves* an AI that behaves the way you
want.

Think of it like onboarding a brilliant new hire: it **interviews** you about
how you want the job done, writes the **playbook**, then **quizzes itself**
against that playbook until it reliably passes.

## How it works

1. **State the goal** — what should this AI do?
2. **Upload your materials** — docs, examples, policies. The tool indexes them
   and finds the *gaps* they don't cover.
3. **Answer a short interview** — only about the gaps; it drafts likely answers
   for you to approve or correct.
4. **It compiles** a behavior spec → system prompt + knowledge lookup (RAG) +
   an eval rubric + test cases.
5. **It tests and scores** the AI against your standards, fixes failures, and
   loops until it passes.
6. **You get a finished, runnable AI** plus the saved spec and tests.

**Guided mode** (default) does all of this with configuration — runs on any
machine. **Advanced mode** (opt-in, technical users) adds a fine-tuning
toolchain on top, gated on actually beating the configured baseline.

Bring-your-own-key: the engine defaults to **Claude** via *your own* API key,
but works equally with **OpenAI** (`<model>@openai`, incl. OpenAI-compatible
endpoints) or a **local Ollama** model (no key / offline). The app runs on your
machine and **no secrets ship in this repo.**

## Status

Alpha (`v0.0.1`) — but the whole pipeline is built and tested:
the Guided loop (`init → ingest → interview → compile → eval → export`, CLI +
local web UI), a deep verification surface (spec lint, deterministic checks,
LLM-judge with self-consistency + human judge calibration, coverage, red-team,
drift, golden snapshots, weighted scoring — composed into one `calibrate ci`
gate), and the Advanced tier (fine-tuning + Engine-Trainer with prove-it gates).
The test suite runs engine-free with fakes; the pipeline is also verified
end-to-end against a real local model via Ollama.

Build plan: `docs/BUILD-PLAN.md` · Architecture: `docs/ARCHITECTURE.md` ·
Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate   # required on stock macOS / modern Debian (PEP 668)
pip install 'ai-calibrator[cloud]'       # or '[all]' for every engine + the web UI (note: [all] pulls a multi-GB ML stack via [rag])
export ANTHROPIC_API_KEY=<your-key>      # your own key; nothing is stored in the repo
calibrate --help
calibrate init my-support-ai --goal "Answer customer product questions in our voice."
calibrate status my-support-ai
```

The engine defaults to **Claude** (cloud, bring-your-own key). To use **OpenAI**,
set `OPENAI_API_KEY` and bind roles to `<model>@openai`. To run **locally** with
no key, install [Ollama](https://ollama.com), pull a model (e.g.
`ollama pull qwen2.5:7b`), and point every role at it:

```bash
calibrate engines my-support-ai --all qwen2.5:7b@ollama
```

`--all` points the **judge** at that model too, so it grades its own answers.
That works, and it is often the only option locally — but the pass rate is then
one model's opinion of itself, and the failures it cannot see are exactly the
ones it cannot report. `calibrate lint` says so, and there are two ways to earn
a stronger number: bind the judge to a different model
(`calibrate engines my-support-ai judge <model@provider>`), or make the criteria
that matter deterministic with `calibrate add-check`, which does not consult a
judge at all.

**📖 Full walkthrough:** [`docs/USAGE.md`](docs/USAGE.md) — engine setup for
Claude / OpenAI / local, the step-by-step workflow, and current build status.
License: MIT.
