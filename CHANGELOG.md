# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Example provenance.** Every `Example` records its `source` (`human`,
  `human_ratified`, or `engine`), and only human-authored or human-ratified rows
  become fine-tuning targets. The "never self-distill" rule was previously advice
  the code could not enforce; it is now enforced, and the excluded count is
  reported so an empty dataset is explained rather than mysterious.
- **The prove-it gate scores held-out tests only.** It detects which graded tests
  were also training prompts, excludes them, and decides on the remainder —
  refusing to judge (exit 2) when every graded test was a training prompt, rather
  than passing a fine-tune that may only have memorized its dataset.
- **A `package` CI job.** Every other job installs with `pip install -e`, which
  imports from `src/` and proves nothing about the distribution. This builds the
  wheel and sdist, checks the wheel ships `py.typed` and the web assets and leaks
  no tests or docs, installs both into clean venvs, and runs the console script
  and the UI from site-packages.
- `calibrate --version` / `-V`.
- `all` package extra (`pip install -e '.[all]'`) — every engine, the
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
- **An empty answer is no longer scored as a red-team "hold".** A subject that
  said nothing was counted as having withstood the probe, inflating the hold rate,
  while `eval` fails every criterion on that same empty output. It is now ungraded
  — neither held nor broke — and excluded from the rate.
- **The API's `/teach/learn` saves judgments before inferring standards**, as the
  CLI does, so an engine failure no longer discards a whole judging session. The
  second call folds in the standards only, so nothing is recorded twice.
- **The Engine-Trainer's generated `train.py` now runs.** It shipped with
  `__EPOCHS__` / `__MAX_STEPS__` unsubstituted; those are valid Python
  identifiers, so the file parsed and only failed with `NameError` after the
  multi-GB base model had downloaded. Both bundle writers now share one renderer
  that asserts every placeholder was replaced.
- **Deleting a document removes it from the retrieval index.** A re-ingest that
  produced no chunks skipped the index rebuild and nothing ever deleted the
  table, so the deleted text kept being injected into every graded and served
  prompt. (Via the web UI there was no way to purge it at all.)
- **`calibrate merge` no longer resolves safety rules by argument order.** The
  refusal policy, format, persona and task type were picked from whichever
  `--from` came first, silently, while the audit file recorded `conflicts: []`.
  Field conflicts are now detected, shown, resolved deterministically by
  stakeholder name, and written to `reconciliation.yaml`. A criterion id shared
  by two stakeholders is namespaced instead of dropping one — along with its
  deterministic check.
- **An interrupted `eval --refine` keeps its refinements.** Each round's
  scorecard was saved immediately but the refined spec only after the whole loop,
  so a Ctrl-C or a 5xx left scorecards on disk that no recorded spec produced.
- **`examples-to-tests` mints ids against the ids already taken.** Deriving them
  from the example's position re-issued an id a pinned test already owned, and a
  duplicate test id made drift and snapshot silently blind to that anchor.
  Duplicate test ids are now a lint error, so `ci` refuses to certify.
- **The API's `POST /drift` skips partial baselines**, matching the CLI and `ci`.
- **`calibrate snapshot` will not pin a golden from a partial run**, which would
  replace a complete golden with a subset.
- **Live feedback: the latest verdict wins.** A `down` on an answer an earlier
  `up` had stored now retracts it instead of leaving the spec asserting the same
  text is both good and bad — and `examples --dedup` keeps the correction rather
  than the rejected answer.
- **`calibrate teach` saves your judgments before inferring standards**, so an
  engine failure no longer discards the whole judging session.
- **A merged project can be compiled** — the next step `merge` itself prints
  used to refuse, making a merged spec impossible to test or certify.
- **Judge scores are clamped to [0, 1].** A judge answering on a 0-100 scale
  pushed the weighted mean above 1, which rendered as a reassuring ">99%".
- **`--json` on `calibrate ci` emits JSON on every exit path**, not just success.
- **Windows: the project lock honours `blocking=False`** — the API's 423
  fast-fail and the CLI's "waiting" notice were unreachable there.
- **`calibrate login claude` no longer runs Apache Ant.** Detection was a bare
  `which ant`; it now verifies the binary is the Anthropic CLI.
- **`anthropic>=0.77`** (the floor for the `output_config` parameter the adapter
  passes) and **`trl>=1.0`** (for `SFTConfig(max_length=…)`); `calibrate train`
  installs those floors instead of bare module names.
- Smaller correctness fixes: the OpenAI strict-schema fallback no longer triggers
  on a timeout or 429 (only on a 400); retrieval failure is reported instead of
  silently degrading to prompt-only; RAG retrieves for the last *user* turn; an
  unreadable material is reported in `skipped` rather than dropped; a `DELETE` of
  a project that removed nothing returns 409; non-scalar judge-label ids are
  rejected instead of raising; the promptfoo export flags omitted multi-turn
  tests; `diff` notices a changed knowledge base; the Ollama `Modelfile` says so
  when it had to rewrite the system prompt; `--host 0.0.0.0` warns that the Host
  guard will reject every client; `/v1/feedback` is size-capped; a failed
  `import`/`merge` cleans up the directory it created; and the web UI keys
  requests on the routing name so a copied project folder can't be mutated by
  mistake.
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
