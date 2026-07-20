# AI Calibrator — Architecture

**Status:** v0.2. The four gating decisions (§13) are RESOLVED:
local-first desktop · LLM/text (expandable) · configure-only v0 · pluggable
engines, **cloud (Claude) default, BYO key** + local (Ollama) opt-in.
Per-role model assignments: §5.1.
**Last structural review:** 2026-06-28

---

## 0. How to read this doc

Two kinds of inline callout are used throughout:

> ❓ **DECISION** — a fork where your answer changes the architecture. The big
> four are consolidated in §13 and are being resolved now.

> 💡 **IMPROVE** — a place where I think the design can be pushed further than
> the original idea. These are proposals, not commitments.

---

## 1. What this is

A guided system that converts a person's **domain knowledge and judgment** into
a **reliable, tested AI configuration** — without requiring them to write
prompts, design evaluations, or format datasets.

It works in four movements:

1. **Ingest** the user's materials (docs, examples, policies, workflows).
2. **Interview** them adaptively to extract the tacit knowledge the materials
   *don't* contain — their standards, tone, edge-case rulings, hard limits.
3. **Compile** answers + extracted knowledge into artifacts: a behavior spec,
   system prompt, RAG configuration, eval rubric, test cases, and — only when
   proven necessary — a fine-tuning dataset and training recipe.
4. **Evaluate & refine** the resulting AI against the spec in a loop until it
   measurably behaves.

**Thesis (the load-bearing idea):** the irreducible value is (a) harvesting the
information that exists *only in the user's head*, and (b) *proving* the AI
behaves. The interview is the on-ramp; **the eval loop is the actual product.**
A behavior spec is easy to produce; a behavior spec the model demonstrably
*obeys* is the hard, valuable thing — and the only thing competitors can't clone
in a weekend.

> 💡 **IMPROVE:** treat the tool as a *compiler for AI behavior*. The user edits
> one human-readable source of truth (the Behavior Spec); everything else
> (prompt, RAG, evals, dataset) is a build artifact regenerated from it. This is
> "behavior-as-code" — versionable, diffable, and re-buildable when the base
> model changes. See §4.

### 1.1 "Does it train your AI?" — yes, two ways

"Training" an AI to act the way you want has two levers — same as getting a new
hire to do the job your way:

- **Configure (default):** give it a precise playbook (system prompt) + a binder
  of your materials to consult (RAG). A capable model with clear instructions and
  the right references behaves to your standard *most of the time* — no rewiring.
  Fast, cheap, runs on your hardware.
- **Fine-tune (when needed):** bake the behavior into the model's weights.
  Powerful but slow/expensive, needs cloud GPUs, and is overkill for most goals.
  Reserved for when configuration *provably* can't get there.

The tool delivers the same end result you pictured — *an AI that behaves to your
exact standards* — and **decides which lever you actually need** (the fine-tune
gate, §3), defaulting to the efficient one. v0 ships the Configure lever; the
Fine-tune lever lands in v1.x. Either way, the promise is intact.

### 1.2 Two experience modes (and two *different* kinds of "tier")

One product, both audiences:

- **Guided mode (default — everyday users).** Interview → spec → system prompt +
  RAG + evals. You get an AI that behaves to your standards and never see the
  word "fine-tune." This is the whole experience for non-technical users.
- **Advanced mode (opt-in — technical users).** Unlocks the fine-tuning
  toolchain (§3.1): dataset generation, training recipes, LoRA configs, and a
  prove-it-beat-the-baseline gate. A deliberate step-up, never forced.

Don't conflate this with the **hardware tiers** (§5.1) — they're orthogonal:

| Axis | Decides… | Example |
|------|----------|---------|
| **Experience mode** | which *capabilities* are exposed | a non-tech user on a workstation still wants Guided |
| **Hardware tier** | which *engine* runs + where training happens | a technical user on a laptop gets Advanced, but the fine-tune routes to cloud |

*What you want to do* (mode) is independent of *what your machine can do*
(hardware tier). The tool picks an engine from the hardware tier and exposes
capabilities from the chosen mode.

---

## 2. Design principles

1. **Human supplies judgment; the system supplies structure.** The AI is
   excellent at *asking* the right questions and terrible at *being* the person
   whose standards are measured. Honor that asymmetry.
2. **Extract before you ask.** Mine the uploaded materials first. Only ask about
   *gaps* the materials don't answer. The richer the upload, the fewer questions.
3. **Propose-and-ratify.** The system drafts the likely answer ("teams like
   yours usually say X — right for you?"); the human confirms or corrects.
   Recognition is far cheaper than recall. This collapses 200 questions into a
   few dozen real decisions.
4. **Ask only high-information questions.** Use value-of-information / active
   learning: skip anything the system can already predict confidently; ask the
   uncertain, high-impact ones.
5. **Teach while scaffolding.** Explain *why* each decision matters as it's
   asked. The user comes out understanding RAG/evals/fine-tuning — not dependent
   on a black box whose failures they can't diagnose.
6. **Measure everything.** Nothing is "done" without an eval run. This forcing
   function is the single biggest reason the output beats hand-rolled configs.
7. **Configuration before training.** Always try prompt + RAG first. Fine-tune
   only when evals *prove* prompt/RAG can't get there.
8. **Model-agnostic & pluggable.** Every intelligent step is an `Engine` call
   behind an interface — swap cloud / local / your-own-fine-tuned model. (§5)
9. **Everything is a versioned artifact.** Spec as YAML, configs as files, evals
   as data. Git-friendly, reproducible, shareable.
10. **Hardware-adaptive.** Assume *nothing* about the user's machine. Detect
    capabilities and recommend an engine; degrade gracefully from strong-local →
    modest-local → cloud-key → no-GPU so *any* user gets a working setup. The
    3080 Ti is one tier among many, not a design assumption. (§5.1)

---

## 3. The core pipeline

```
[0 Goal] → [1 Ingest] → [2 Interview] → [3 Compile] → ⟨FT GATE⟩ → [4 Evaluate] → [5 Refine] ↺ → [6 Export]
                ↑__________________________________ refine loop _________________________________|
```

**Stage 0 — Goal.** One sentence: "what should this AI do?" + task type
(assistant / classifier / extractor / writer / agent / support / etc.). Sets the
shape of every later stage.

**Stage 1 — Ingest.** Parse uploads → chunk → embed → summarize. Run an
**Extractor** pass that pulls candidate facts, preferences, rules, and examples
out of the materials, and — critically — produces a **gap list**: dimensions the
materials *don't* settle (tone? refusal policy? format? edge cases?).

**Stage 2 — Interview.** The **Interviewer** generates adaptive questions
targeting the gap list + high-uncertainty dimensions. Propose-and-ratify UX.
Dimensions covered: goal/scope, persona & tone, quality standards, do/don't
rules, edge-case rulings, output format, refusal & escalation boundaries,
examples of good vs bad. Loop is active-learning driven (principle 4).

**Stage 3 — Compile.** The **Compiler** synthesizes the interview + extracted
knowledge into the artifact bundle (§4). The Behavior Spec is produced first;
the rest derive from it.

**⟨Fine-tune Gate⟩ — the "unknown unknowns" centerpiece.** Before any training,
an explicit, *explained* decision: does this goal need fine-tuning at all?
Heuristic the tool walks the user through:
- Failures are **missing knowledge** → RAG, not fine-tuning.
- Failures are **format/style ignored despite a clear prompt**, *and* volume is
  high / latency-sensitive → fine-tuning may help.
- Failures are **capability/reasoning** → bigger/different base model, not a
  small fine-tune.
- Default verdict: **don't fine-tune.** Most goals are solved by prompt + RAG.
- **Mode-aware:** in *Guided* mode the gate runs quietly and just keeps
  optimizing the config; in *Advanced* mode, if it judges fine-tuning would help,
  the tool *offers to run it* (§3.1).

**Stage 4 — Evaluate.** Run the candidate config against the test cases; grade
with the rubric (deterministic checks + LLM-as-judge). Emit a scorecard with
per-criterion pass rates and a list of failures. (§9)

**Stage 5 — Refine.** Diagnose each failure → propose a targeted change (prompt
edit, more retrieval, new test case, or "this is the fine-tune case") → re-run.
Loop until the pass threshold is met. Every run is diffed against the last so
regressions are caught.

**Stage 6 — Export / Deploy.** Emit the compiled bundle + a runtime adapter:
a hosted API endpoint, a local Ollama `Modelfile`, a portable prompt+RAG package,
or training scripts for the cloud.

### 3.1 Advanced tier — the fine-tuning toolchain

When a technical user opts into Advanced mode *and* the gate says fine-tuning
would actually help, the tool does as much of the work as the hardware allows.
Key insight: **the dataset is a byproduct of Guided mode** — we don't conjure
training data, we harvest what the interview and eval loop already produced.

1. **Dataset assembly.** Build the set from: the spec's examples, the interview's
   ratified answers, and — most valuable — the **human-corrected outputs from the
   eval/refine loop** (the cases where you said "wrong; here's right"). Format for
   the chosen trainer (chat JSONL, etc.). *Never self-distill* — the model writing
   both prompt and ideal answer teaches nothing; corrections are the signal.
2. **Approach recommendation.** LoRA vs full fine-tune, base model, and
   hyperparameters — fit to the user's hardware tier (§5.1). Default: LoRA.
3. **Runnable recipe.** Emit a training script wrapping standard OSS toolkits
   (`unsloth` / `axolotl` / `LLaMA-Factory` / PEFT). Capable GPU → run locally;
   otherwise → a **cloud recipe** (RunPod/Vast template + script) the user runs on
   their own rented GPU and account.
4. **Prove-it gate.** Score the fine-tuned model on the **same held-out eval
   harness** (§9). **Only accept a fine-tune that measurably beats the prompt+RAG
   baseline** — otherwise the tool tells you to stay on configuration. This is
   what stops impressive-looking fine-tunes that don't actually help.
5. **Export.** Ship the adapter/weights + how to run them (e.g. an Ollama
   `Modelfile`), so the result drops back into the same local-first runtime.

Non-technical users never see any of this — it is the step-up tier.

---

## 4. The Behavior Spec (central artifact)

The single source of truth. A structured doc (YAML front-matter + prose
sections) that everything else compiles from.

```yaml
goal: "Answer customer billing questions accurately and warmly."
task_type: support_assistant
persona:
  voice: "warm, concise, never condescending"
  reading_level: "8th grade"
standards:
  - "Never invent a refund policy; cite the policy doc or escalate."
  - "Always confirm the account before discussing specifics."
do_not:
  - "Promise timelines we don't control."
edge_cases:
  - situation: "Customer is angry and threatens chargeback"
    ruling: "Acknowledge, de-escalate, escalate to human within 1 reply."
format:
  default: "≤120 words, no markdown tables"
refusal_policy: "Decline legal/medical advice; redirect to a human."
knowledge_sources: [billing_policy.pdf, faq.md]
eval_criteria:                 # each becomes a rubric item + test cases
  - id: cites_policy
    description: "Refund claims are backed by a cited policy line."
    weight: high
examples:
  - input: "..."
    good_output: "..."
    bad_output: "..."
    why: "..."
```

Compilation targets derived from the spec:
`system_prompt.txt` · `rag.config.yaml` · `rubric.yaml` · `tests.jsonl` ·
`finetune.jsonl` (optional) · `train.recipe.yaml` (optional).

> 💡 **IMPROVE — drift handling = recurring value.** Because the spec is the
> source and artifacts are regenerated, the tool can *re-compile and re-eval
> when the base model updates*, flagging behavior drift. That converts a
> "configure once" tool into something with an ongoing reason to exist.

---

## 5. Pluggable Engine architecture  ← answers "can I add my own fine-tuned LLM?"

Every intelligent step is a **role** behind one interface. Roles:

| Role | Job | Good fine-tune target? |
|------|-----|------------------------|
| **Extractor** | mine facts/rules/gaps from uploads | medium |
| **Interviewer** | generate adaptive questions | **high** — domain-specific |
| **Predictor** | draft likely answers to ratify | **high** |
| **Compiler** | synthesize spec → artifacts | medium |
| **Judge** | grade outputs against rubric | **high** — your standards |
| *(Subject)* | the model being *configured* — separate from engines | n/a |

```
interface Engine {
  id: string
  capabilities: Role[]
  complete(prompt, schema?) -> structured result
}
```

Providers implementing `Engine`: `AnthropicEngine` (cloud — **the default**,
BYO key), `OpenAIEngine` (cloud — opt-in, BYO key; also any OpenAI-compatible
endpoint via `OPENAI_BASE_URL`), `OllamaEngine` (local — opt-in, no key),
`CustomFineTuneEngine` (your own model — endpoint or local weights).

**Cloud default, BYO key — still zero secrets in the repo.** The user supplies
their *own* `ANTHROPIC_API_KEY` (env var / OS keychain); the repo ships no keys.
Three paths:

1. **Cloud (default).** Claude powers every role — best quality, runs on any
   hardware, nothing to install. Needs the user's own key. (One tradeoff: it
   doesn't work key-free out of the box — see path 2.)
2. **Local (opt-in).** Point any role at `<model>@ollama` to run a local open
   model instead — no key, no cost, fully private/offline. *The AI you're
   configuring can also be a local model, so the whole loop can be cloud-free end
   to end if the user wants.*
3. **Your own fine-tuned engine (later).** Once the tool has logged enough
   interactions to train one (roadmap below), swap it in via `CustomFineTuneEngine`.

Roles → providers in config (cloud defaults shown; picks justified in §5.1):

```yaml
engines:
  interviewer: claude-opus-4-8@anthropic      # reasoning roles → Opus
  predictor:   claude-opus-4-8@anthropic
  compiler:    claude-opus-4-8@anthropic
  judge:       claude-haiku-4-5@anthropic      # high-volume → cheap/fast Haiku
  subject:     claude-sonnet-4-6@anthropic     # the model the CONFIGURED AI runs on
  # use OpenAI instead — point any role at an OpenAI model:
  # judge:     gpt-4o-mini@openai
  # run locally with no key — point any role at an Ollama model:
  # judge:     qwen2.5:14b@ollama
```

### 5.1 Engine selection — hardware-adaptive (capability tiers)

The tool assumes **nothing** about the user's machine. On first run it detects
OS / RAM / GPU VRAM and recommends an engine, degrading gracefully so *everyone*
gets a working setup:

| Tier | Hardware | Default engine | Fine-tune (v1.x) |
|------|----------|----------------|------------------|
| **0** | No/old GPU, or "I prefer cloud" | BYO cloud key (Claude/GPT), or a tiny 1–3B local model | cloud only |
| **1** | 8–16 GB VRAM (e.g. a 3080 Ti) | local quantized 7–14B via Ollama | cloud |
| **2** | 24 GB VRAM | local 14–32B; small local LoRA possible | local (small) or cloud |
| **3** | 48 GB+ / multi-GPU | large local models; local fine-tuning | local |

Rules:
- The recommendation is always **overridable** (pick a bigger/smaller model, or
  force cloud).
- A user with **no GPU and no key** is told their options plainly (add a key, or
  install a small local model) — never a broken state.
- **Training routes by capability:** if the machine can't fine-tune, the tool
  emits a *cloud training recipe* the user runs on their own rented GPU. It never
  silently assumes local training is possible.

**Cloud-path model picks (opt-in)** — grounded in the current Claude API reference:

| Role | Model | Price /1M (in/out) | Why |
|------|-------|--------------------|-----|
| interviewer / predictor / extractor / compiler | `claude-opus-4-8` | $5 / $25 | Reasoning-heavy, lower-volume — quality matters most for question generation and spec synthesis |
| judge | `claude-haiku-4-5` | $1 / $5 | High-volume (runs per test case) and more constrained — cheap + fast wins |

`claude-sonnet-4-6` ($3 / $15) is the natural middle option where Opus is more
than a role needs. All bindings are overridable. Two cost levers are built into
the adapter: **structured outputs** (`output_config.format` + a JSON schema, so
compiler/judge return validated JSON directly) and **prompt-cache the system
prompt** — the judge reuses the same rubric + spec across every test case, so
caching it cuts repeated input cost ~90% (activates once the cached prefix
passes the model's ~4K-token minimum).

**OpenAI (and OpenAI-compatible) products** work the same way — bind any role to
`<model>@openai` (e.g. `gpt-4o` for reasoning roles, `gpt-4o-mini` for the judge,
or any current model you have access to). Set `OPENAI_API_KEY`; set
`OPENAI_BASE_URL` to target Azure OpenAI or any OpenAI-compatible server. Same
structured-output mechanism (JSON-schema response format).

**Autonomization roadmap (the self-bootstrapping loop):**
1. Run on cloud engines; the tool logs every interview, prediction, and judgment.
2. That log *is* a labeled dataset for each role.
3. Fine-tune a small local model on, say, the Judge role for your domain.
4. Swap it in via `CustomFineTuneEngine`. Repeat per role.
5. End state: the tool runs largely on your own local, autonomous engines —
   private, free to run, and specialized to your standards.

> 💡 **IMPROVE:** ship an explicit **"Engine Trainer"** mode later: it takes the
> tool's own interaction logs and produces a fine-tuned engine + an eval proving
> the local model matches the cloud one before you trust it. The tool calibrates
> *itself* using its own machinery. (Dogfooding as a feature.)

---

## 6. Data model & storage

A **Project** is a directory (git-friendly):

```
my-project/
  goal.yaml
  materials/                # uploaded source files
  knowledge.db              # local vector index (chunks + embeddings)
  interview.jsonl           # transcript: questions, drafts, ratified answers
  spec.yaml                 # the Behavior Spec — source of truth
  build/                    # compiled artifacts (system_prompt, rubric, tests…)
  evals/
    run-0001/ scorecard.json failures.jsonl
    run-0002/ ...
  engines.yaml              # role → provider bindings
```

Everything is plain files → diffable, versionable, shareable as a git repo.
Vector store: local (LanceDB / sqlite-vss / Chroma) or pgvector if hosted.

---

## 7. Deployment & UI — RESOLVED: local-first desktop app

**Decision:** ship as a **downloadable, open-source desktop app**. It runs on
the user's machine and their API key lives in the OS keychain — the repo ships
no secrets. By default it calls Claude with the user's key; pointing roles at a
local Ollama model keeps everything on-device for privacy / offline use.

- A **desktop shell** (§8) wraps a localhost backend + a web UI.
- The same core is exposed as a **CLI**, so the whole pipeline is usable from a
  terminal before the GUI is polished — and scriptable.
- No mandatory server, no account, no secrets in the repo.

*Alternatives considered (self-host web / hosted SaaS / CLI-only) lost on cost,
data-custody, or accessibility trade-offs — see §13.*

---

## 8. Tech stack — RESOLVED

Layered so the UI stays a thin, replaceable shell:

**1. Calibration Core** (Python) — the whole pipeline (§3), spec compiler (§4),
eval runner (§9), and engine adapters (§5). Pure library, no UI. Python because
the doc-parsing / embedding / vector-search / local-model (Ollama) / training-
handoff ecosystem lives there.
- Doc parsing: `unstructured` / `pypdf` / `python-docx`.
- Embeddings + vector store: `sentence-transformers` (local) → **LanceDB**
  (embedded, file-based, no server).
- Engine adapters: `OllamaEngine` (default), plus optional cloud SDKs (§5).

**2. Local API** — **FastAPI** wrapping the Core, bound to localhost only.

**3. CLI** — a thin Typer front-end over the same Core. Usable today, before any
GUI. (The "core-library-first" principle — the UI choice stays reversible.)

**4. Desktop shell** — **Tauri** (small binaries; Rust shell spawns the Python
backend as a sidecar) + **React/TypeScript** UI. *Electron* is the fallback if
Tauri's Python-sidecar packaging proves fiddly.

> **v0 status:** the API (FastAPI) and a served **web UI** (static, bundled in
> `calibrator/web/`, launched by `calibrate serve`) are built and tested — they
> drive the full Guided loop in a browser today. The Tauri native packaging
> wraps this same localhost UI and is the remaining step.

```
[ React/TS UI ] ⇄ [ FastAPI @ localhost ] ⇄ [ Calibration Core (Py) ] ⇄ [ Engine: Ollama (default) / cloud / your model ]
        └────────── Tauri shell ──────────┘            └── LanceDB + file store ──┘
```

> 💡 **IMPROVE:** keep the Core importable and CLI-runnable in isolation. If the
> desktop story ever stalls, you still have a working tool.

---

## 9. Eval system design

The heart of the product. Three layers:

1. **Deterministic checks** — structure/JSON-schema validity, required/forbidden
   terms, length, citation presence. Cheap, exact, no LLM needed.
2. **LLM-as-judge** — score each output against each rubric criterion (0–1 +
   rationale). Used for subjective standards (tone, helpfulness, judgment).
3. **Golden examples** — the good/bad pairs from the interview become
   regression anchors.

**Judge reliability (a real risk):** the judge has its own error. Mitigations:
calibrate the judge against a small human-labeled set before trusting it; use
multiple judge passes or perspectives on high-stakes criteria; sample outputs
for human spot-check; prefer deterministic checks wherever a criterion can be
made objective.

**Scorecard:** the headline metric in v0 is a **binary pass rate** over gradeable
tests. Per-criterion 0–1 scores and criterion `weight`s are recorded (in
`scorecard.json` / `rubric.yaml`) for inspection and future weighted scoring, but
do not yet affect the pass rate. Plus a ranked failure list feeding Stage 5.

> 💡 **IMPROVE:** emit a **"Calibration Confidence"** score — how well-specified
> *and* how well-passing the AI is — so the user sees, concretely, "your AI is
> 86% calibrated; the gaps are tone-on-edge-cases." Makes invisible quality
> visible, which was the core sales/trust problem.

> 💡 **IMPROVE — interop:** also export evals in `promptfoo` / standard formats
> so power users plug into existing harnesses instead of being locked in.

---

## 10. Phasing / roadmap

- **v0 (Guided mode — the core):** Goal → Ingest → Interview →
  Compile (spec + system prompt + RAG) → Evaluate → Refine. Local engine default
  (pluggable). Ships the whole loop for the common case, on any hardware tier.
- **v1 (Advanced mode):** the fine-tuning toolchain (§3.1) — dataset assembly,
  LoRA recipes (local or cloud by tier), prove-it eval gate. Opt-in step-up.
- **v2:** Engine Trainer (autonomization §5); drift detection; multi-stakeholder
  calibration; modality expansion if chosen.

---

## 11. Improvement ideas (consolidated 💡)

1. Behavior-as-code compiler with regenerate-on-model-change (drift = recurring value).
2. Self-bootstrapping engine autonomy via the tool's own interaction logs.
3. Calibration Confidence score (makes quality visible).
4. Active-learning question selection (minimize human effort).
5. Adversarial/counterfactual test generation for robustness.
6. Multi-stakeholder calibration with conflict reconciliation (org use).
7. Eval-format interop (promptfoo/OpenAI-evals export).
8. Core-library-first design so the UI choice stays reversible.

---

## 12. Risks & honest constraints

- **Spec→adherence gap.** The hard, unsolved problem. We *mitigate* it with the
  eval loop; we don't *solve* it. Set expectations accordingly.
- **Judge error.** LLM-as-judge can be wrong; see §9 mitigations.
- **Fine-tuning dataset quality.** Never let the model write both prompts *and*
  ideal answers (self-distillation teaches nothing new). Require human-corrected
  examples. Keep fine-tuning behind the gate.
- **Scope creep.** Multimodal + full SaaS + autonomy all at once would sink a
  young project. v0 must stay tight.

---

## 13. Open decisions — RESOLVED 2026-06-28

1. **Deployment / UI** → **Local-first desktop app** (Tauri + React over a Python
   core; CLI from the same core). (§7, §8)
2. **AI scope for v1** → **LLM/text, expandable** — concrete text artifacts now,
   modality-agnostic abstractions so image/video can plug in later. (§1, §3)
3. **v1 depth** → **Both levers, as two experience modes** (§1.2). *Guided mode*
   (configure: prompt + RAG + evals) is the v0 foundation and the default for
   everyone. *Advanced mode* (the full fine-tuning toolchain, §3.1) is an opt-in
   step-up for technical users — built as completely as feasible, fed by the data
   Guided mode already gathers, gated on beating the config baseline. Sequencing:
   Guided first (v0), Advanced next (v1). (§1.2, §3.1, §5.1, §10)
4. **Engine default** → **Cloud (Claude), BYO key.** Chosen for output quality
   and zero-install simplicity — the engine's quality drives the calibrated AI's
   quality. The user supplies their own `ANTHROPIC_API_KEY`; the repo ships
   **zero secrets**. Local (Ollama) is a one-line opt-in for privacy / offline /
   no-cost use; your own fine-tuned engine plugs in later. (Tradeoff accepted: no
   longer key-free out of the box.) (§5, §5.1)

---

## 14. Open items — RESOLVED

- **Name.** ~~"Anvil" is a placeholder.~~ Shipped as **AI Calibrator**
  (`ai-calibrator` / `calibrate`).
- **License.** ~~MIT or Apache-2.0.~~ **MIT** (see `LICENSE`).
- **Repo init + CI.** ~~Once §13-Q1 lands.~~ Done — repo, CI (3 OSes ×
  3 Python versions + locale + bandit jobs), README, CONTRIBUTING all exist.
