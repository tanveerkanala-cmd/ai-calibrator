# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `calibrate --version` / `-V`.
- `all` package extra (`pip install 'ai-calibrator[all]'`) — every engine, the
  web UI, and document ingestion in one install.
- Host/Origin guard on the `calibrate run` OpenAI-compatible endpoint (shared
  with `calibrate serve` via `webguard.py`) — blocks browser CSRF and DNS
  rebinding against the local server.

### Changed
- Renamed the import package `calibrator` → `ai_calibrator` (the `calibrate`
  command is unchanged) to avoid a top-level import clash with existing PyPI
  packages.
- CLI help no longer leaks internal build-plan milestones (`(M1)`, `(M3+)`) or
  architecture section references (`§9`) into user-facing text.

### Fixed
- **Deterministic checks grade the AI's words, not the user's.** A multi-turn
  test used to run every check over the whole transcript, so a `not_contains`
  check failed on a word the *user* typed and `max_chars` charged the AI for the
  question. Checks now see the assistant's turns only; the judge and the recorded
  scorecard still get the full transcript.
- **An empty answer fails every criterion.** The blank-output guard ran only
  ahead of the *judged* criteria, so a test whose criteria all carried
  deterministic checks scored 100% for an answer of `""` (`not_contains` and
  `max_chars` are both trivially satisfied by an empty string) — a certified gate
  in front of a silent AI. The guard now precedes every grading layer.
- **A partial scorecard can no longer become a drift baseline.** `calibrate ci`
  and `calibrate drift` defaulted to the newest saved run, so a `--max-tests`
  smoke run became the reference point and every regression on a test it skipped
  reported as "no regressions". Both now default to the latest **full** run and
  refuse an explicitly pinned partial one.
- **`calibrate interview --regenerate` no longer destroys ratified answers.** It
  rebuilt the interview list from the gaps and overwrote the saved one, discarding
  the only artifact in a project that cannot be recomputed. Answered items are now
  carried through untouched (including ones whose gap has since disappeared); only
  unanswered drafts regenerate.
- **A test expecting a missing criterion is a lint error.** Such tests still ran
  and still cost an engine call, but graded against nothing and dropped out of the
  pass-rate denominator entirely — a green gate over ungraded behavior. `ci` now
  refuses to certify until the expectation is fixed.
- **The badge and certificate report the certified numbers.** A green badge took
  its colour from the gate but its pass rate from the newest run, so a one-test
  smoke run after a passing gate published its own 100% in green. The badge now
  reports the run the gate certified, and no report headlines a partial run.
- **Certification measures the prompt that actually gets served.** Single-turn
  tests were sent as the bare input while `calibrate run` and the API's `/try`
  send a transcript-encoded turn; eval now uses the same encoding as the runtime.
- **Recompile no longer blanks the persona, format, or refusal policy.** Only the
  list fields were carried forward, so a synthesis that didn't restate the scalar
  fields silently deleted them from the rendered prompt — a safety rule could
  evaporate while the compile summary looked unchanged.
- **Recompile no longer discards pinned regressions.** `calibrate compile` now
  preserves deterministic `add-check` criteria, red-team criteria, `fb_*`/`rt_*`
  **and `ex_*`** regression tests, and edge cases instead of regenerating the
  suite from scratch (the "can never silently regress" guarantee).
- **`calibrate diff` reports scalar behavior changes.** It compared only lists and
  criteria, so a reversed refusal policy, a changed persona, a dropped format rule,
  or a retargeted deterministic check reviewed as "No behavior change".
- **Red team reports zero probes as a warning, not a 100% hold** — nothing was
  attacked, so nothing held.
- **`--guard` no longer claims an enforcement it can't perform.** Only
  `calibrate add-check` creates a deterministic check, so on a normally compiled
  project the guard had nothing to run while the CLI printed "guard ON" and `GET /`
  reported `"guard": true`. It now warns at boot and reports `"inactive"`.
- Friendlier failures: a malformed `.yaml`/`.json` example file reports the
  problem and its location instead of dumping a parser traceback, and uploading a
  material named `.`, `..`, or longer than the filesystem allows returns a `400`
  instead of a `500`.
- **Zip-bomb caps rewritten to stream** the real decompressed size instead of
  trusting the archive's self-declared size (which is forgeable); PDF extraction
  is bounded per page and in total.
- **Ingest isolates per-file failures** — one corrupt/oversized file is skipped
  and reported instead of aborting the whole batch unnamed.
- **Judge verdicts are strictly coerced** — a non-compliant judge returning the
  string `"false"` no longer grades as a pass and inflates the score. Duplicate
  criterion ids in a test no longer multiply that criterion's weight.
- **`calibrate run` stays concurrent** — the blocking engine call runs off the
  event loop, so one in-flight completion no longer freezes every request.
- Friendlier errors: reserved/invalid project names on `calibrate init`, and
  `NaN`/`Infinity` request bodies now return a clean `422` instead of `500`.
- Top-level `tools`/`functions` in a serve request are rejected with a clear
  `400` (were silently dropped).

## [0.0.1] — unreleased development version

Initial development: ingest → interview → compile → eval/refine → red-team →
drift → serve/export pipeline, guided + advanced (fine-tuning) modes, web UI,
Claude/OpenAI/Ollama engines.
