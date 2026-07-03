# Using the AI Calibrator

A step-by-step walkthrough: install it, pick an engine (Claude, OpenAI, or
local), and run the workflow. For the *why* behind the design, see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for the build roadmap, see
[`BUILD-PLAN.md`](BUILD-PLAN.md).

> **Build status (v0 — early).** Commands marked **✅** run today. Commands
> marked **🔜** are scaffolded and currently print which milestone delivers them.
> This guide describes the full intended workflow so you understand the product,
> and flags exactly what's live right now.

---

## 1. What it does (in a minute)

You bring your knowledge and standards; the tool **interviews** you, writes a
behavior spec, then compiles **and tests** an AI configuration that behaves the
way you want — no prompt-writing, no datasets. Like onboarding a new hire: it
interviews you, writes the playbook, then quizzes itself until it passes.

Two depths:
- **Guided mode** (default) — configure an AI via system prompt + retrieval
  (RAG) + evals. Runs on any machine.
- **Advanced mode** (opt-in, technical users) — adds a fine-tuning toolchain.
  (🔜 v1.)

---

## 2. Install

```bash
git clone <your-repo-url> ai-calibrator
cd ai-calibrator
pip install -e '.[cloud]'        # the [cloud] extra adds the Anthropic + OpenAI SDKs
calibrate --help
```

Requires Python 3.10+. (If you'll only ever run local models, plain
`pip install -e .` is enough.)

---

## 3. Choose your engine

The **engine** is the LLM that powers the tool's intelligence — asking
questions, mining your docs, grading outputs. It is pluggable **per role**
(interviewer, predictor, extractor, compiler, judge). Pick one:

### Option A — Claude (default)
Two ways to authenticate:
```bash
calibrate login claude              # (a) browser login — NO API key (uses the `ant` CLI / OAuth)
export ANTHROPIC_API_KEY=sk-ant-... # (b) or an API key
```
Reasoning roles default to `claude-opus-4-8`; the high-volume judge to
`claude-haiku-4-5`; the **subject** — the AI being configured and tested — to
`claude-sonnet-4-6`. Run `calibrate auth` to see what's configured and
`calibrate engines` to see each role's binding (§5 shows how to change them).

### Option B — OpenAI
```bash
export OPENAI_API_KEY=sk-...
# optional — Azure OpenAI or any OpenAI-compatible server:
# export OPENAI_BASE_URL=https://your-endpoint/v1
```
> OpenAI's API is **key-based** — there is no supported "sign in with ChatGPT"
> for third-party tools, so you use a key (from platform.openai.com).
Then point roles at OpenAI models — e.g. `gpt-4o` for reasoning, `gpt-4o-mini`
for the judge (or any current model you have access to). See §5 for setting
bindings on a project.

### Option C — Local (no key, offline, private)
Install [Ollama](https://ollama.com), pull a model, and bind roles to
`<model>@ollama`:
```bash
ollama pull qwen2.5:14b
ollama serve
```
No API key, no per-use cost, fully private. Comfortable on a 12 GB+ GPU.

> You can **mix** engines per role — e.g. Claude for the interviewer, a cheap
> local model for the judge. (A per-role CLI command is 🔜; for now set bindings
> in `project.yaml` — see §5.)

---

## 4. The workflow

Each command maps to one pipeline stage. Run them in order on your project.

### `calibrate init` ✅
```bash
calibrate init my-support-ai \
  --goal "Answer customer product questions in our brand voice, never making medical claims." \
  --task-type support_assistant
```
Creates a project folder with `project.yaml` and an empty `materials/` directory.

### `calibrate import` ✅ — already have a system prompt? Start here.
If you already wrote a system prompt and just want it **tested**, skip the
interview entirely:
```bash
calibrate import my-ai --prompt ./system_prompt.txt \
  --goal "Help customers with bookstore questions" --task-type support_assistant
```
It **reverse-engineers** the behavior spec your prompt implicitly encodes
(standards, never-rules, edge cases, refusal policy, and measurable eval
criteria), generates a test suite, and saves the original prompt for the record.
The result is a normal project — run `calibrate eval`, `coverage`, `redteam`,
`report`, or `drift` on it immediately. (`--engine model@provider` picks the
engine for extraction and the created project; default is the standard binding.)

### `calibrate status` ✅
```bash
calibrate status my-support-ai
```
Shows the goal and a checklist of how far the project has progressed.

### `calibrate engines` ✅
```bash
calibrate engines my-support-ai
```
Shows which engine powers each role.

### `calibrate ingest [--source DIR] [--no-index]` ✅
Drop your materials — product docs, past replies, policies, FAQs — into
`materials/`, then ingest. The tool parses and indexes them, and works out the
**gaps**: the things your materials *don't* settle (tone? refusal policy? edge
cases?). Needs a configured engine (see §3).

### `calibrate interview [--accept-drafts] [--regenerate]` ✅
Fills the gaps. It generates one targeted question per gap with a **drafted
answer** and a short *why*, then walks you through them — press Enter to accept
the draft or type a correction (propose-and-ratify). `--accept-drafts` takes the
drafts non-interactively. This is where the judgment that lives only in your head
gets captured.

### `calibrate compile` ✅
Turns your answers + materials into the **behavior spec** (the source of truth),
then compiles the artifact bundle into `<project>/build/`: `spec.yaml`,
`system_prompt.txt`, `rag.config.yaml`, `rubric.yaml`, and `tests.jsonl`. Needs a
configured engine (see §3).

### `calibrate eval [--refine] [--rounds N] [--threshold 0.8] [--judge-passes N]` ✅
Runs each test on the configured AI (the `subject` engine), grades each output
against the rubric with an LLM judge (plus a deterministic empty-output guard),
and saves a scorecard under `<project>/evals/`. `--refine` loops: it diagnoses
failures, adds standards to the spec, and re-runs until the pass rate clears
`--threshold`. `--judge-passes N` (self-consistency) grades each criterion with
`N` independent judge calls and majority-votes — then flags the verdicts the
judge was **split** on, so you can spot-check where the noisy LLM-judge is
unreliable. Tests can also be **multi-turn conversations** (a test's `follow_ups`
are subsequent user turns); the subject answers each in context and the judge
grades the whole exchange. This testing step is what makes the result reliable
instead of guesswork.

Alongside the pass rate you get a **weighted score** — criteria count high=3 /
medium=2 / low=1, so it says how much of what *matters* passed (a test that
missed only a low-weight criterion scores 0.85, not 0). Pass/fail itself stays
strict: any failing criterion fails its test. Failures are listed highest-weight
first, tagged `[high]` / `[medium]` / `[low]`, and each verdict records the
weight it was graded under, so old scorecards stay honest even after you reweight
the spec.

### `calibrate export [--name NAME]` ✅
Packages the calibrated config into `<project>/export/`: the system prompt,
spec/rubric/tests, an Ollama **`Modelfile`** (`ollama create … -f Modelfile`
then `ollama run`), a zero-dependency `run.py`, and a README. Fully
deterministic — no engine needed. The behavior lives in the provider-agnostic
system prompt, so it runs on any model.

---

## 4a. Beyond the core loop — confidence, coverage & tuning

These build on a compiled project to make calibration *smarter, more trustworthy,
and lower-friction*. All are live (✅).

### `calibrate teach [--n 5]`
Calibrate **by example** instead of (or alongside) the interview. The tool shows
sample outputs on real inputs; you approve or reject each with an optional
one-line reason, and it **infers your standards** from the pattern — folding them
into the spec and recording each judgment as a golden example. Ideal when you can
*recognize* good output but struggle to *articulate* the rules. Can even bootstrap
a spec from scratch (judge the raw model first, build from your verdicts).

### `calibrate lint [--deep]` (no engine, unless `--deep`)
Lints the **spec itself** for quality problems before you waste an eval run: no
measurable criteria, criteria nothing tests, vague/unfalsifiable standards,
duplicates, a missing refusal policy. `--deep` adds an engine pass that flags
**self-contradictions** ("be concise" vs "always explain in depth"). Exits
non-zero on errors (CI-friendly).

### `calibrate judge-check [--sample N]` (no engine)
Calibrates the **judge** — the eval is only as trustworthy as the LLM doing the
grading. Confirm or correct a sample of the judge's verdicts from the latest run;
it reports how often the judge agreed with you, overall and **per criterion**, and
flags the criteria where the judge is unreliable (too subjective — reword them, or
grade with `eval --judge-passes`). The §9 "calibrate the judge" mitigation, made
concrete. Your answers are saved to `evals/<run>/human-labels.json` — they're an
asset: `calibrate train-engine judge` uses them as **ground truth** (§6b).

### `calibrate ci [--threshold 0.8] [--tolerance 0] [--baseline RUN] [--judge-passes N] [--json]`
The whole verification surface as **one gate** for pipelines and cron:
**lint → eval → drift → snapshot**, cheap-to-expensive. Lint errors stop the gate
before any engine call is spent; a stage that can't run yet (no baseline run, no
pinned golden) reports *skip* — never a silent pass. `--baseline RUN` drifts
against a blessed run instead of the previous one; `--json` prints a
machine-readable result. Exit codes: `0` gate passed, `1` couldn't gate
(spec/engine problems), `2` the AI failed the gate.
```bash
calibrate ci my-ai --threshold 0.9 --tolerance 0.05   # e.g. nightly, or on every spec change
```

### `calibrate add-check <path> <criterion> <kind> <value>` (no engine)
Attach a **deterministic check** to a criterion so it's graded exactly by code —
not the noisy LLM judge — for objectively-verifiable behavior. Kinds: `contains`,
`not_contains`, `regex`, `max_chars`, `min_chars`, `non_empty`. e.g.
`calibrate add-check my-ai cites contains "30-day"`. This is §9's layer-1
(deterministic checks): the reliability floor under the judge.

### `calibrate examples-to-tests` (no engine)
Turns the spec's good/bad **examples** into regression tests (each example's input
becomes a test graded against all criteria) — §9's "golden examples as regression
anchors", so the exact cases you cared about stay pinned in the suite.

### `calibrate coverage` (no engine — instant)
"Test coverage, but for behavior." Shows which eval criteria have a **targeted
test** and which don't, flags HIGH-weight criteria with no test, and warns when a
spec is under-measured. No model calls.

### `calibrate redteam [--max-probes 12] [--add-tests]`
Adversarially tries to make your configured AI **break its own rules** — crafting
attacks (social engineering, edge cases, false authority) against each standard /
never-rule / edge case, running them, and reporting what broke. `--add-tests`
promotes confirmed violations into the suite as regressions, so
`calibrate eval --refine` is then forced to fix them.

### `calibrate rightsize [--models a@p,b@p,…] [--threshold 0.8]`
Runs your existing tests across several models (default: the Claude tier ladder)
and recommends the **cheapest model that still meets your pass bar** — e.g.
"Haiku passes 94% at ~1/20th Opus's cost."

### `calibrate drift [--baseline RUN] [--tolerance 0]`
Re-runs the suite and flags **behavior drift** vs a baseline scorecard (default:
the latest): the pass-rate delta and exactly which tests flipped pass↔fail.
**CI-friendly** — exits code 2 when behavior regresses, so you can gate a deploy
or catch a provider's silent model update.

### `calibrate diff <before> <after>` (no engine — instant)
Shows how the behavior **spec** changed between two projects — standards,
never-rules, edge cases, and criteria added / removed / changed. (`drift`
compares scorecards; `diff` compares the specs themselves.) Great for reviewing
the effect of a refine, teach, or merge before you ship it.

### `calibrate snapshot [--check]` (no engine — instant)
Golden-output snapshot testing for AI. `calibrate snapshot` pins the latest run's
outputs as a golden; `calibrate snapshot --check` flags any test whose **output
text changed** since (exit 2 on change). Catches tone shifts and semantic drift
that pass/fail grading is too coarse to notice — run it after an `eval` to see
not just *whether* the score moved but *what the answers became*.

### `calibrate report` (no engine — instant)
Generates a shareable **calibration report** (`calibration-report.md`) — the AI's
"nutrition label": a **Calibration Confidence** score (coverage × pass rate), the
spec at a glance, coverage gaps, the latest eval's weak spots, and provenance (the
ratified answers the spec was built from). For showing stakeholders, at a glance,
how trustworthy the configured AI is.

### `calibrate export-evals [--format promptfoo]` (no engine)
Exports the generated test suite + rubric as a **promptfoo** config
(`promptfooconfig.yaml`) — the provider-agnostic system prompt becomes a prompt,
each test an input, each criterion an `llm-rubric` assertion. Run your
calibrator suite inside promptfoo (`promptfoo eval -c promptfooconfig.yaml`)
instead of being locked into `calibrate eval`. Anti-lock-in.

---

## 4b. Drive it from the web UI (instead of the CLI)

`calibrate serve` starts a local API + web UI:
```bash
pip install -e '.[api]'
calibrate serve                 # → http://127.0.0.1:8765
calibrate serve --port 9000     # if 8765 is taken
```
Flags: `--port PORT` (default `8765`), `--projects DIR` (where projects live;
default `~/.ai-calibrator/projects`), and `--host HOST` (default `127.0.0.1`,
i.e. reachable only from your machine — binding anything else prints a warning,
since the API has no authentication).

Open that URL to create a project, upload materials, and run
ingest → interview → compile → eval → export from the browser — the same Guided
loop, with a UI and a scorecard view. The **Teach, Coverage, Report, Red-team,
Rightsize, and Drift** actions (§4a) are surfaced there too. A native desktop
wrapper (Tauri) around this same UI is a packaging step on the roadmap.

## 5. Your project on disk

A project is just plain, git-friendly files:

```
my-support-ai/
  project.yaml      # goal, task type, engine bindings, spec, tests — the whole project
  materials/        # the documents you upload
```

To choose or mix engines today, edit the `engines:` block in `project.yaml`
(a per-role CLI command is 🔜):

```yaml
engines:
  interviewer: gpt-4o@openai
  predictor:   gpt-4o@openai
  extractor:   gpt-4o@openai
  compiler:    gpt-4o@openai
  judge:       gpt-4o-mini@openai      # cheap/fast model for the high-volume role
  subject:     gpt-4o@openai           # the model your CONFIGURED AI runs on (evaluated in M4)
```

Each value is a `model@provider` string. Providers: `anthropic`, `openai`,
`ollama`. Mix freely — e.g. `claude-opus-4-8@anthropic` for the compiler and
`qwen2.5:14b@ollama` for the judge.

---

## 6. Advanced mode — fine-tuning ✅ (opt-in, technical users)

For technical users, when evals show configuration alone isn't enough:
```bash
calibrate finetune                 # → <project>/finetune/ : dataset.jsonl, recipe.yaml, train.py, README
calibrate finetune --base mistralai/Mistral-7B-Instruct-v0.3   # pick the open base model
```
It assembles a chat-format dataset from your spec's examples (human-authored /
corrected — never the model's own output), recommends a LoRA recipe for the base
model (`--base`, a Hugging Face model id; default `Qwen/Qwen2.5-7B-Instruct`),
and emits a runnable training script. You train on a GPU (local ~16 GB+, or a rented cloud
GPU), then run the **prove-it gate** — keep the fine-tune *only* if it beats your
configured baseline on the same evals:
```bash
calibrate finetune --gate --baseline <run-id> --candidate <run-id>
```
Non-technical users never see any of this. Details in
[`ARCHITECTURE.md`](ARCHITECTURE.md) §3.1.

---

## 6b. Engine-Trainer — run the tool on your own models (autonomy) ✅

The tool can calibrate *itself*. Turn on local logging and every decision your
cloud engines make for a role is recorded — that log is a labeled dataset to
fine-tune a small **local** model that reproduces the role. Localize one role at a
time until the tool runs privately and free on your own engines.

```bash
calibrate log --on              # opt-in; logs to <project>/logs/<role>.jsonl (stays local)
calibrate eval                  # run normally — the judge's decisions are now logged
calibrate train-engine judge    # → <project>/trained-engines/judge/ : dataset, LoRA recipe, train.py, README
```
(`--base <hf-model-id>` picks the open base model for the LoRA recipe, as in
`calibrate finetune`.)

For the **judge** role the dataset is upgraded with **human ground truth**: every
verdict you confirmed or corrected in `calibrate judge-check` becomes a training
row of what the judge *should* have said — and where a logged row asks the exact
same question, the human answer replaces it. Imitating the cloud judge copies its
mistakes; your labels train past them. (The bundle README shows the split.)

Train that bundle on a GPU (see its README), serve the result (e.g. `ollama
create`), then **prove it matches** before trusting it:

```bash
calibrate train-engine judge --prove --candidate my-judge@ollama --threshold 0.9
```

This replays your logged inputs through the local engine and reports how often it
**agrees** with the cloud engine (for the judge, that's per-criterion pass/fail
agreement — rationale wording is ignored). Swap it into `engines.judge` in
`project.yaml` **only** once agreement clears your threshold; otherwise keep the
cloud engine. Repeat per role (`judge`, `compiler`, `extractor`, `interviewer`).

Logging is **off by default** and never leaves your machine — the path to a fully
local, private, autonomous tool, on your terms.

---

## 6c. Multi-stakeholder calibration — merge & reconcile (org use) ✅

In an org, one AI answers to several voices — legal, sales, support, brand — whose
standards contradict. Let each stakeholder calibrate their **own** project, then
merge them into one, reconciling the conflicts explicitly instead of silently
averaging them:

```bash
calibrate merge org-ai --from legal-ai --from sales-ai --from support-ai
```

The tool flags every pair of rules that directly conflict (e.g. legal's "always
add a disclaimer" vs sales' "keep it punchy, no disclaimers"), shows who
contributed each, and asks you to **rule**: keep A, keep B, or write a merged
rule (with a rationale). The result is a unified spec — everyone's
non-conflicting rules plus your rulings — and an audit trail in
`org-ai/reconciliation.yaml`. Use `--report-only` to just preview the conflicts.

Then `calibrate compile` the merged project to regenerate its tests and rubric.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not reach Ollama at ...` | Run `ollama serve`, and `ollama pull <model>` for the model in your bindings. |
| Anthropic / OpenAI auth error | Make sure `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` is exported **in the same shell** you run `calibrate` from. |
| `needs the 'anthropic'/'openai' package` | Install the cloud extra: `pip install -e '.[cloud]'`. |
| Want zero setup / no key | Use a local Ollama model (Option C) for every role. |

---

*Questions about the design, the tiers, or the roadmap? See
[`ARCHITECTURE.md`](ARCHITECTURE.md) and [`BUILD-PLAN.md`](BUILD-PLAN.md).*
