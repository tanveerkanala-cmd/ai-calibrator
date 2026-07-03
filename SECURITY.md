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
