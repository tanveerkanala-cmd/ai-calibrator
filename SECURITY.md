# Security

## Model

AI Calibrator is a **local-first** tool: projects, materials, eval scorecards,
logs, and training datasets live on your disk and are never sent anywhere except
to the LLM provider you explicitly configure for each engine role (or to a local
Ollama server, in which case nothing leaves your machine).

- **API keys are read from environment variables only** (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`) — never written to `project.yaml`, exports, reports, logs,
  or training bundles.
- **`calibrate serve` binds `127.0.0.1` by default** and has **no
  authentication**. It refuses silently-broad exposure: binding any non-local
  host prints an explicit warning. Do not expose it to a network; put nothing
  in front of it that forwards untrusted traffic.
- The API guards project names and run ids against path traversal, caps upload
  sizes, and validates Host/Origin headers against DNS rebinding.
- `calibrate init` drops a `.gitignore` into new projects so eval outputs, logs,
  and credentials can't be swept into a user's repository by accident.
- Local logs (`logs/*.jsonl`) and the project lock are created owner-only
  (`0600`); artifact writes use an owner-only temp + atomic rename.
- Ingest skips symlinks and over-size files, so a shared project cannot make the
  tool follow a link to a file outside its `materials/` directory.

## What the certification / badge actually attest

`calibrate ci`, the boot gate, and the badge reflect **the last gate run on
*this* machine** — they are a *local build artifact*, exactly like a green
checkmark in your own CI. `evals/last-gate.json` is plain JSON bound to the
current spec/subject/tests by a hash (change-detection, so a stale gate reads as
`stale`, not `pass`). It is **not** a cryptographic attestation: anyone who can
write that file — including whoever hands you a project — can set it green.

There is no trusted secret on a single-user tool to sign it with, so we do not
pretend to. The rule is simple: **do not trust a badge or certification that came
from a project you did not calibrate yourself — run `calibrate ci` and let it
re-earn the gate** (it is fast and deterministic). Trust the run, not the file.

## Trusting your own materials and spec

Ingested documents are **trusted input**: their text flows into the behavior spec
and thus the deployed system prompt. Only ingest documents you trust, and
**review the compiled spec** (`calibrate report`, or `build/system_prompt.txt`)
before deploying — a malicious or careless document could otherwise inject
standards you did not intend. The extractor prompt fences materials as data as a
defense-in-depth measure, but the human review of the spec is the real control.

With RAG retrieval on, ingested chunks also reach the served AI's system prompt
at query time. Those chunks are JSON-encoded and framed as untrusted reference
data (never instructions) so a poisoned chunk cannot break out of the knowledge
block — again defense-in-depth, not a guarantee. Only index documents you trust.

## Reporting a vulnerability

Please **do not** open a public issue for security reports. Use GitHub's
**private vulnerability reporting** ("Report a vulnerability" under the
repository's Security tab). Include a minimal reproduction. You should hear back
within a week.

## Scope notes

- Prompt-injection via ingested materials is an inherent LLM-layer risk: the
  extractor reads your documents, so only ingest documents you trust. The
  red-team command (`calibrate redteam`) exists to probe the *configured* AI's
  rule-keeping, not to sanitize inputs.
- Regex checks (`calibrate add-check … regex`) execute owner-authored patterns
  with a hard timeout (see `checks.py`) so a catastrophic pattern cannot hang an
  eval.
