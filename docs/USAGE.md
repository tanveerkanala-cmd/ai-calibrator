# Using the AI Calibrator

A step-by-step walkthrough: install it, pick an engine (Claude, OpenAI, or
local), and run the workflow. For the *why* behind the design, see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for the build roadmap, see
[`BUILD-PLAN.md`](BUILD-PLAN.md).

> **Status (v0 — early).** Every command in this guide works today. Expect rough
> edges — this is a young tool.

---

## 1. What it does (in a minute)

You bring your knowledge and standards; the tool **interviews** you, writes a
behavior spec, then compiles **and tests** an AI configuration that behaves the
way you want — no prompt-writing, no datasets. Like onboarding a new hire: it
interviews you, writes the playbook, then quizzes itself until it passes.

Two depths:
- **Guided mode** (default) — configure an AI via system prompt + retrieval
  (RAG) + evals. Runs on any machine.
- **Advanced mode** (opt-in, technical users) — adds a fine-tuning toolchain
  (§6).

---

## 2. Install

```bash
git clone https://github.com/tanveerkanala-cmd/ai-calibrator.git
cd ai-calibrator
python3 -m venv .venv && source .venv/bin/activate   # required on stock macOS / modern Debian (PEP 668)
pip install -e '.[cloud]'        # the [cloud] extra adds the Anthropic + OpenAI SDKs
calibrate --help
```

Requires Python 3.10+. The virtualenv isn't optional on a system Python that
enforces [PEP 668](https://peps.python.org/pep-0668/) (Homebrew Python, recent
Debian/Ubuntu) — a bare `pip install` there fails with
`externally-managed-environment`. (If you'll only ever run local models, plain
`pip install -e .` is enough.)

The optional extras, mix as needed:

| Extra | Adds |
|-------|------|
| `cloud` | Anthropic + OpenAI SDKs (skip if you only use local Ollama) |
| `api` | the web UI + `calibrate serve` / `run` servers |
| `docs` | PDF / DOCX ingestion (`.pdf`, `.docx` materials) |
| `rag` | local retrieval index — **pulls a multi-GB ML stack** (sentence-transformers → PyTorch) |
| `train` | the fine-tuning stack (PyTorch + transformers/trl/peft) — install with `-e '.[train]'`; **excluded from `all`** |
| `all` | `cloud` + `api` + `docs` + `rag` (so also multi-GB, via `rag`; **not** `train`) |

`pip install -e '.[cloud,api,docs]'` is the common no-GPU combination.

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

Claude replies are capped at 16000 output tokens. A long compile or export can
hit that ceiling — the error says so; raise it with
`CALIBRATOR_ANTHROPIC_MAX_TOKENS=32000` (tokens) in the environment.

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
ollama pull qwen2.5:7b
ollama serve
```
No API key, no per-use cost, fully private. Comfortable on a 12 GB+ GPU.

On a slower machine (or a busy shared model) a big extraction can exceed the
default 120s request timeout — raise it with `CALIBRATOR_OLLAMA_TIMEOUT=300`
(seconds) in the environment.

> You can **mix** engines per role — e.g. Claude for the interviewer, a cheap
> local model for the judge — with `calibrate engines` (see §5), or by editing
> `project.yaml` directly.

---

## 4. The workflow

Each command maps to one pipeline stage. Run them in order on your project.

### `calibrate init`
```bash
calibrate init my-support-ai \
  --goal "Answer customer product questions in our brand voice, never making medical claims." \
  --task-type support_assistant
```
Creates a project folder with `project.yaml` and an empty `materials/` directory.

### `calibrate import` — already have a system prompt? Start here.
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

### `calibrate status`
```bash
calibrate status my-support-ai
```
Shows the goal and a checklist of how far the project has progressed.

### `calibrate engines [ROLE MODEL] [--all MODEL]` (no engine)
```bash
calibrate engines my-support-ai                              # show every role's binding
calibrate engines my-support-ai subject gpt-4o-mini@openai   # rebind one role
calibrate engines my-support-ai --all qwen2.5:7b@ollama      # point every role at one model
```
Shows — or sets — which engine powers each role. `model@provider` uses
`anthropic` / `openai` / `ollama` (bare `model` defaults to local Ollama). The
binding is validated (known provider, non-empty model) without contacting the
provider, so it never needs a key just to configure. (Also `PUT
/api/projects/<name>/engines`.)

### `calibrate ingest [--source DIR] [--no-index]`
Drop your materials — product docs, past replies, policies, FAQs — into
`materials/`, then ingest. The tool parses and indexes them, and works out the
**gaps**: the things your materials *don't* settle (tone? refusal policy? edge
cases?). Needs a configured engine (see §3).

With the `rag` extra installed (`pip install -e '.[rag]'`), ingest builds a local
vector index (`knowledge.lancedb`) over your materials. **`calibrate eval` and
`calibrate run` then retrieve** the most relevant chunks for each question and
prepend them to the AI's context — so your scorecard reflects the RAG-augmented
AI you actually serve, not a prompt-only version. Without the extra (or with
`--no-index`), ingest still works and eval/run run prompt-only. The
`rag.config.yaml` in the export bundle describes the same index for your own
deployment.

### `calibrate interview [--accept-drafts] [--regenerate]`
Fills the gaps. It generates one targeted question per gap with a **drafted
answer** and a short *why*, then walks you through them — press Enter to accept
the draft or type a correction (propose-and-ratify). `--accept-drafts` takes the
drafts non-interactively. This is where the judgment that lives only in your head
gets captured.

`--regenerate` re-drafts questions from the current gaps — for example after
re-ingesting new materials. **Answers you already gave are never re-asked or
overwritten**: a gap you've answered is carried through untouched, and an answer
whose gap has since disappeared is kept too. Only unanswered drafts regenerate.

### `calibrate compile`
Turns your answers + materials into the **behavior spec** (the source of truth),
then compiles the artifact bundle into `<project>/build/`: `spec.yaml`,
`system_prompt.txt`, `rag.config.yaml`, `rubric.yaml`, and `tests.jsonl`. Needs a
configured engine (see §3).

### `calibrate eval [--refine] [--rounds N] [--threshold 0.8] [--judge-passes N]`
Runs each test on the configured AI (the `subject` engine), grades each output
against the rubric with an LLM judge (an answer that is empty fails every
criterion outright, before any grading layer runs),
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

### `calibrate export [--name NAME]`
Packages the calibrated config into `<project>/export/`: the system prompt,
spec/rubric/tests, an Ollama **`Modelfile`** (`ollama create … -f Modelfile`
then `ollama run`), a zero-dependency `run.py`, and a README. Fully
deterministic — no engine needed. The behavior lives in the provider-agnostic
system prompt, so it runs on any model.

---

## 4a. Beyond the core loop — confidence, coverage & tuning

These build on a compiled project to make calibration *smarter, more trustworthy,
and lower-friction*. All are live.

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
grade with `eval --judge-passes`). The "calibrate the judge" mitigation, made
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
`calibrate add-check my-ai cites contains "30-day"`. This is the layer-1
(deterministic-check) reliability floor under the judge.

### `calibrate examples-to-tests` (no engine)
Turns the spec's good/bad **examples** into regression tests (each example's input
becomes a test graded against all criteria) — "golden examples as regression
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
`calibrate eval --refine` is then forced to fix them. A run that produces **no
probes** (a spec with no concrete rules to attack, or a generator that returned
nothing usable) is reported as a warning, never as a hold — nothing was attacked,
so nothing held.

### `calibrate rightsize [--models a@p,b@p,…] [--threshold 0.8]`
Runs your existing tests across several models (default: the Claude tier ladder)
and recommends the **cheapest model that still meets your pass bar** — e.g.
"Haiku passes 94% at ~1/20th Opus's cost."

### `calibrate drift [--baseline RUN] [--tolerance 0]`
Re-runs the suite and flags **behavior drift** vs a baseline scorecard (default:
the latest **full** run): the pass-rate delta and exactly which tests flipped
pass↔fail. **CI-friendly** — exits code 2 when behavior regresses, so you can gate
a deploy or catch a provider's silent model update. Partial runs (interrupted, or
`--max-tests`) are never used as the baseline and are refused if you pin one:
comparing across two different test sets would hide every regression on a test the
baseline never ran.

### `calibrate diff <before> <after>` (no engine — instant)
Shows how the behavior **spec** changed between two projects — the goal, persona,
format and refusal policy, plus standards, never-rules, edge cases, and criteria
added / removed / changed (including a retargeted deterministic check). (`drift`
compares scorecards; `diff` compares the specs themselves.) Great for reviewing
the effect of a refine, teach, or merge before you ship it.

### `calibrate snapshot [--check]` (no engine — instant)
Golden-output snapshot testing for AI. `calibrate snapshot` pins the latest run's
outputs as a golden; `calibrate snapshot --check` flags any test whose **output
text changed** since (exit 2 on change). Catches tone shifts and semantic drift
that pass/fail grading is too coarse to notice — run it after an `eval` to see
not just *whether* the score moved but *what the answers became*.

### `calibrate report [--html] [--badge]` (no engine — instant)
Generates a shareable **calibration report** (`calibration-report.md`) — the AI's
"nutrition label": a **Calibration Confidence** score (coverage × pass rate), the
spec at a glance, coverage gaps, the latest eval's weak spots, and provenance (the
ratified answers the spec was built from). For showing stakeholders, at a glance,
how trustworthy the configured AI is.

`--html` also writes a single-file **calibration certificate**
(`calibration-report.html`) you can publish next to your bot. `--badge` writes
`badge.json` in the shields.io *endpoint* format — embed
`https://img.shields.io/endpoint?url=<public URL of badge.json>` in a README and
the project wears its calibration the way a repo wears CI:
**calibrated | 97% · 12 tests**. Colors are honest: green only for a *passing*
gate that certifies the *current* spec/subject; orange for ungated or stale; red
for a failing gate. The numbers are honest too: a green badge reports the run the
gate actually certified, and neither the badge nor the certificate ever headlines a
partial (`--max-tests` or interrupted) run. (The API also serves it live: `GET /api/projects/<name>/badge`.)

### `calibrate export-evals [--format promptfoo]` (no engine)
Exports the generated test suite + rubric as a **promptfoo** config
(`promptfooconfig.yaml`) — the provider-agnostic system prompt becomes a prompt,
each test an input, each criterion an `llm-rubric` assertion. Run your
calibrator suite inside promptfoo (`promptfoo eval -c promptfooconfig.yaml`)
instead of being locked into `calibrate eval`. Anti-lock-in.

---

## 4a-bis. Serve the calibrated AI itself — `calibrate run`

The export bundle is a file; this is the **live** thing. `calibrate run` serves
your calibrated AI as an **OpenAI-compatible endpoint** — point any chat UI, SDK,
or tool that speaks the OpenAI protocol at it and you're done wiring:

```bash
calibrate ci my-ai          # certify first
calibrate run my-ai         # → http://127.0.0.1:8600/v1  (model name = project name)
```

Three properties make it more than a proxy:

- **The boot gate.** It checks the last `ci` verdict before serving: a **failing
  gate refuses to boot** (exit 2; `--force` to override), a stale gate (the spec
  or subject changed since certification) or a missing one serves with a loud
  UNCERTIFIED warning, and a passing gate prints its certificate. An AI that
  can't prove it follows your rules shouldn't quietly pretend it does.
- **What you tested is what you serve.** The system prompt is compiled from your
  spec, and live conversations are transcript-encoded with the *same function*
  the eval harness uses for multi-turn tests.
- **`--guard`** re-runs the spec's deterministic checks on every **live** answer:
  a violating answer is retried once; still-failing responses are returned but
  flagged (`x-calibrate-guard: failed:<criteria>` header) and logged to
  `logs/guard.jsonl` — the tests never stop running. It can only enforce criteria
  that carry a check, and only `calibrate add-check` creates one — so on a project
  without any, `run --guard` warns at boot and `GET /` reports
  `"guard": "inactive"` rather than claiming an enforcement that isn't happening.

`GET /` self-describes the certification; `GET /v1/models` lists the project;
client `system` messages are ignored by design (the calibrated spec is the
authority). Flags: `--host` (default `127.0.0.1`; no auth — keep it local),
`--port` (default `8600`), `--guard`, `--force`. Streaming (`"stream": true`)
is supported, so standard chat UIs work unchanged.

Scope, honestly: **plain text chat**. Text content-parts
(`[{"type":"text",…}]`) are accepted; function/tool-calling and image/audio
content are **rejected with a clear 400** rather than silently dropped from the
context — a lost tool result would corrupt the conversation invisibly.

### The flywheel: `POST /v1/feedback` + `calibrate absorb`

The runtime is also the capture point for **learning from real use**. Thumbs-up
or thumbs-down any live answer:

```bash
curl -s http://127.0.0.1:8600/v1/feedback -H "Content-Type: application/json" -d '{
  "completion_id": "chatcmpl-…",          # from the completion response
  "verdict": "down",                       # or "up"
  "correction": "No — the window is 30 days.",
  "reason": "invented policy"
}'
```
(or pass `"input"`/`"turns"` + `"output"` explicitly, e.g. after a restart).
Feedback lands durably in `logs/feedback.jsonl`. Then:

```bash
calibrate absorb    # (no engine)
```

folds every record into the project: the exchange becomes a spec **example**
(down → `bad_output`, with your correction as `good_output`; up → `good_output`)
— the same asset that feeds fine-tuning — and the conversation becomes a
**pinned regression test** (`fb_1`, `fb_2`, …; multi-turn feedback keeps its
follow-ups), so the exact exchange someone flagged can never silently regress.
Absorbing changes the certification fingerprint, so the gate goes **stale**
until `calibrate ci` re-proves the AI against the suite that now includes what
it just learned. Use → flag → absorb → re-certify: the AI gets measurably more
reliable the more it's used, with receipts.

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

**API reference.** Every screen action is a REST endpoint; the full, interactive
reference is served live at **`/docs`** (Swagger UI) with the raw schema at
**`/openapi.json`** — e.g. `http://127.0.0.1:8765/docs`. Projects are addressed
by name (`/api/projects/<name>/…`); create with `POST /api/projects`, remove with
`DELETE /api/projects/<name>`. An upstream engine failure returns `502`/`504`
(a bad request `400`); a project busy with another operation returns `423`.

**Same-origin guard.** The server accepts requests only from an allowed `Host`
and, for mutating requests, a same-origin (or no) `Origin` — a browser page on a
*different* origin gets `403`, and an unrecognized `Host` gets `400`. This blocks
DNS-rebinding and CSRF from a malicious local web page. If you build a browser
front-end on another port, call the API from a same-origin proxy (or bind the
server to that origin), not cross-origin.

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

Set engine bindings with `calibrate engines <role> <model@provider>` (or
`--all <model@provider>` for every role at once). You can also edit the
`engines:` block in `project.yaml` directly:

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
`qwen2.5:7b@ollama` for the judge.

---

## 6. Advanced mode — fine-tuning (opt-in, technical users)

> **Prerequisite:** Advanced mode builds on a *compiled* project — examples
> attach to the behavior spec. Complete the Guided loop first
> (`init → ingest → interview → compile`, §4) before importing examples or
> fine-tuning. (`calibrate examples --import` on a project with no spec fails
> with a message pointing you back here.)

**First, the data.** A fine-tune is only as good as its examples, and most owners
already have some (past replies, an FAQ, a spreadsheet). Collect + curate them
(the first argument is the project directory):
```bash
calibrate examples my-support-ai                          # review: how many you have, how far from a solid fine-tune
calibrate examples my-support-ai --import support-qa.csv  # bulk-import input/output pairs (.csv/.jsonl/.json/.yaml)
calibrate examples my-support-ai --dedup                  # drop duplicate inputs
```
Column/key names are matched flexibly (`input`/`question`/`prompt`…,
`good_output`/`output`/`answer`…); a UTF-8 BOM and messy rows are handled, and
malformed rows are skipped with a per-line report rather than aborting the import.
Examples also grow from `calibrate teach` and captured eval corrections. Rule of
thumb: ~50+ before a fine-tune tends to beat the prompt+RAG baseline.

For technical users, when evals show configuration alone isn't enough:
```bash
calibrate finetune my-support-ai                 # → <project>/finetune/ : dataset.jsonl, recipe.yaml, train.py, merge.py, README
calibrate finetune my-support-ai --base mistralai/Mistral-7B-Instruct-v0.3   # pick the open base model
```
It assembles a chat-format dataset from your spec's examples (human-authored /
corrected — never the model's own output), recommends a LoRA recipe for the base
model (`--base`, a Hugging Face model id; default `Qwen/Qwen2.5-7B-Instruct`),
and emits a **device-aware** training script (CUDA / Apple-Silicon MPS / CPU)
plus a `merge.py` that folds the trained adapter back into the base for serving.

**One-command run** — build the bundle, install the training stack (with your
OK), and train, in a single step:
```bash
calibrate train my-support-ai                    # detects your hardware, offers to install torch/transformers/…, then trains
calibrate train my-support-ai --base Qwen/Qwen2.5-3B-Instruct    # a smaller base fits a 10–12 GB GPU or an M-series Mac
calibrate train my-support-ai --epochs 1 --max-steps 20          # bound the work (a fast smoke run)
calibrate train my-support-ai --qlora            # load the base in 4-bit (CUDA + bitsandbytes) so a 7B fits a consumer card
```
`--epochs` / `--max-steps` are baked into the generated `train.py`. The other
hyperparameters (`learning_rate`, `lora_r`, `lora_alpha`, `lora_dropout`,
`max_seq_len`) are read from `recipe.yaml` at run time, so editing that file
before `python train.py` genuinely changes the run. The command prints an
estimated step count before it starts. (Or install once —
`pip install -e '.[train]'` — and run `python train.py` yourself.)

**Serve it, then gate it.** The adapter is a LoRA delta — `python merge.py`
writes a merged model to `finetune/merged/`, which you serve either via
`ollama create` or an OpenAI-compatible endpoint (`transformers serve`), then
bind as the project's `subject` (the bundle README spells out both). Run the
**prove-it gate** — keep the fine-tune *only* if it beats your configured
baseline on the same evals (both scorecards must be full runs graded by the same
judge; the gate warns otherwise). **The gate scores only held-out tests.** The
dataset is built from `spec.examples`, and `examples-to-tests` / `absorb` turn
those same examples into `ex_*` / `fb_*` tests — so the gate detects which graded
tests were also training prompts, excludes them, and decides on the rest. If every
graded test was a training prompt it refuses to judge (exit 2) rather than pass a
fine-tune that may only have memorized:
```bash
calibrate finetune my-support-ai --gate --baseline <run-id> --candidate <run-id>
```
Exit codes: `0` accept, `2` a clean reject, `1` an error (e.g. an unreadable run id).
Fitting the hardware: a fp16 LoRA of a 7B wants ~16 GB VRAM; below that use
`--qlora` (CUDA) or a smaller `--base`. On Apple Silicon a 0.5–3B base trains on
the MPS GPU comfortably; the 7B needs ~24 GB+ unified memory.
Non-technical users never see any of this. Details in
[`ARCHITECTURE.md`](ARCHITECTURE.md) §3.1.

---

## 6b. Engine-Trainer — run the tool on your own models (autonomy)

The tool can calibrate *itself*. Turn on local logging and the decisions your
cloud engines make get recorded — today that means the **judge** (during `eval`
and `ci`) and the **compiler** (during `eval --refine`); those logs are a labeled
dataset to fine-tune a small **local** model that reproduces the role. Localize
one role at a time until the tool runs privately and free on your own engines.

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

## 6c. Multi-stakeholder calibration — merge & reconcile (org use)

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
