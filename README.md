# AI Calibrator

[![ci](https://github.com/tanveerkanala-cmd/ai-calibrator/actions/workflows/ci.yml/badge.svg)](https://github.com/tanveerkanala-cmd/ai-calibrator/actions/workflows/ci.yml)

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

Alpha (`0.0.1`, no releases yet) — but the whole pipeline is built and tested:
the Guided loop (`init → ingest → interview → compile → eval → export`, CLI +
local web UI), a deep verification surface (spec lint, deterministic checks,
LLM-judge with self-consistency + human judge calibration, coverage, red-team,
drift, golden snapshots, weighted scoring — composed into one `calibrate ci`
gate), and the Advanced tier (fine-tuning + Engine-Trainer with prove-it gates).
The test suite runs engine-free with fakes; the pipeline is also verified
end-to-end against a real local model via Ollama.

Roadmap: `docs/BUILD-PLAN.md` · Architecture: `docs/ARCHITECTURE.md` ·
Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Quickstart

```bash
# from a clone of this repo:
pip install '.[cloud]'                   # or '.[all]' for every engine + the web UI (note: [all] pulls a multi-GB ML stack via [rag])
export ANTHROPIC_API_KEY=sk-ant-...      # your own key; nothing is stored in the repo
calibrate --help
calibrate init my-support-ai --goal "Answer customer product questions in our voice."
calibrate status my-support-ai
```

The engine defaults to **Claude** (cloud, bring-your-own key). To use **OpenAI**,
set `OPENAI_API_KEY` and bind roles to `<model>@openai`; to run **locally** with
no key, install [Ollama](https://ollama.com) (`ollama pull qwen2.5:14b`) and use
`…@ollama`.

**📖 Full walkthrough:** [`docs/USAGE.md`](docs/USAGE.md) — engine setup for
Claude / OpenAI / local, the step-by-step workflow, and current build status.
License: MIT.
