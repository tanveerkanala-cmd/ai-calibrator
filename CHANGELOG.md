# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **`lint` warns when the judge is the subject.** One model answering and
  grading measures its own opinion of itself: the blind spots are shared by
  construction, so the failures it cannot see are exactly the ones it cannot
  report, and a shared idiosyncrasy reads as agreement. It is easy to reach by
  accident — `calibrate engines <project> --all <model>` does it in one command,
  and the README's local quickstart said to run precisely that. A warning, not
  an error: it is a reasonable way to work when no second model is available,
  and it is skipped entirely when every criterion is graded by a deterministic
  `check`, since no judge is consulted then. The README now says it too.
- **`schema_version` in `project.yaml` and `scorecard.json`.** Nothing reads it
  yet, which is the point: when this ships, those files become a compatibility
  contract with strangers, and a format with no version marker can only be
  migrated by guessing what wrote it. It cannot be added retroactively to files
  already on disk. A file written before the field loads as version 1; a file
  from a newer calibrator is flagged by `lint`, because unknown *fields* were
  already reported but a change in what an existing field MEANS reads as
  ordinary data. Stamping it does not move `config_hash`, so no existing
  certification goes stale on upgrade.
- **`pip-audit` in CI, and Dependabot.** Nothing is pinned — every job installs
  floating versions — so an advisory in a transitive dependency is the likeliest
  way a vulnerability reaches a user, and it would otherwise present as our bug.
  Bandit reads our code; this reads what we depend on.
- **A regression net under the CLI surface, and a coverage floor to keep it.**
  Every contract a pipeline reads — `ci`'s 0/1/2, `run`'s refusal to serve a
  failed gate and the `--force` override, the exit-2 signals of `lint`, `drift`
  and `snapshot --check`, the refusal to pin a golden from a partial run, the
  two guards on the one function that removes a user's directory, and the
  coverage percentage itself — was reached by no test: each could be deleted or
  inverted with the suite green. `tests/test_cli_exit_codes.py` pins all of
  them and walks every registered command against a missing and a corrupt
  project, and CI now holds `cli.py` to a coverage floor.
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
- **`calibrate serve` reads the current directory.** It used to keep its own
  registry under `~/.ai-calibrator/projects` while every other command takes a
  path, so `calibrate init my-ai && calibrate serve` — the documented quickstart
  — ended at an empty list. A project is a directory of plain files you keep in
  your repo; the server no longer holds a second opinion about where they live.
  Running it inside a project serves that project. `--projects DIR` still points
  anywhere, and the startup banner names the root it read and what it found. If
  your projects are still in the old location, it says so and gives the command.
- **A deterministic `regex` check exports to promptfoo only when both engines
  read it alike.** promptfoo compiles the pattern with JavaScript's `RegExp`;
  outside the dialects' common subset (named groups, `\A`/`\Z`, inline flags,
  possessive quantifiers, Unicode properties) JS either throws or matches
  something else, so the exported suite graded a different rule. Patterns outside
  that subset now take the documented `llm-rubric` downgrade instead.
- **`CALIBRATOR_ANTHROPIC_MAX_TOKENS` is clamped to 21333**, the most one
  non-streaming request can carry, and says so on stderr. The truncation error
  previously suggested doubling the value, which produced a number the SDK
  rejects before sending — turning one truncated call into every call failing.
- **mypy is a gate, not advice.** The tree is clean and CI enforces it.
- **A judge-check label now corrects the logged judge row even for a criterion
  that has since gained a deterministic check.** No new row is minted for it (the
  judge is never asked again), but the call recorded before the check was
  attached still carries the verdict the human overturned.
- Renamed the import package `calibrator` → `ai_calibrator` (the `calibrate`
  command is unchanged) to avoid a top-level import clash with existing PyPI
  packages.
- CLI help no longer leaks internal build-plan milestones (`(M1)`, `(M3+)`) or
  architecture section references (`§9`) into user-facing text.

### Fixed
- **Three assertions that could not fail.** `_has_no_traceback` — 37 uses across
  two files — checked `result.output` for "Traceback", which typer's CliRunner
  never writes there (it stores the exception on `result.exception`), so a
  command that printed a friendly message and then crashed passed every one of
  them; it now asks the result. A lock-parallelism test used `or` where the
  property needs `and`, and so passed under total serialization — the exact
  thing it existed to rule out. A stateful invariant compared `config_hash` to
  itself in the same process, which only rejects internal randomness; it is now
  the save/load round trip the certification gate actually depends on.
- **The regression gate no longer certifies a comparison of different
  questions.** A test id names a slot, not a question, and `compile` re-mints
  `t1..tN` positionally on every run — so after the ordinary loop (compile →
  eval → answer more questions → compile → ci) two scorecards could share every
  id while grading two different exams. `drift` reported `Δ ±0%, no
  regressions` over a suite whose questions had all been replaced, hiding a real
  regression whose probe was gone; in the other direction it announced
  `✓ 1 improved (fail→pass)` for a failing question that had simply been
  deleted. `finetune --gate` accepted on the same false difference, `snapshot`
  compared a golden against a question it was never pinned to, `finetune`'s
  memorization check read the current suite's input for an old run, and
  `train_engine` paired the current question with an old answer and stamped a
  human's verdict on it as judge training data. The content check that already
  guarded `report` now lives in `identity` and governs all six. Results that
  are no longer comparable are reported as such — `drift` skips with a reason
  instead of passing, `ci` names how many ids were re-minted, `snapshot`
  distinguishes a *replaced* pin from a *removed* test, and `--gate` refuses.
  Scorecards and goldens written before the content was recorded still compare
  by id exactly as they did.
- **`snapshot --check` no longer reports drift for an interrupted run.** It
  resolved the newest run without excluding partial ones, so every test a
  `--max-tests` or interrupted run never graded read as `removed` — exit 2 under
  the self-contradicting summary "0 output(s) changed vs golden". Pinning
  already refused partial runs; checking now does too, in the CLI and the API.
- **A corrupt `golden.json` is no longer reported as "No golden yet".**
  `load_golden` answers `None` for both, and the CLI printed re-pin
  instructions — advice that would overwrite the only copy of the pinned
  outputs with the current run's. `ci` already told the two apart; the CLI now
  does, and says the check is not running.
- **`interview --regenerate` no longer destroys ratified answers.** It saved
  after each gap, and the payload it saved was a prefix of the interview —
  answered items are folded back in lazily — so a failure partway (a timeout, a
  429) left `project.yaml` holding only the gaps reached so far, and every answer
  belonging to a later gap was gone. The CLI then said "Progress was saved", and
  the re-run it advertised refilled those slots with model drafts and exited 0.
- **Material ingestion no longer reads hidden directories.** `rglob` yields a
  hidden folder's children as paths of their own, so skipping `.git` by its leaf
  name still read `.git/config` — a remote URL with a token in it — into the
  spec, the index, and every served prompt.
- **A test id is no longer treated as a test's identity.** `compile` mints
  `t1..tN` positionally and regenerates the whole range, so the ordinary loop
  (compile, eval, answer more questions, compile again) replaces every probe with
  different text under the same id — and the report credited the old run's
  verdicts to tests that had never been executed. A run now records a hash of
  what it asked; scorecards written before this field are still matched by id, so
  existing projects report exactly as they did.
- **`POST /v1/feedback` can no longer hang the served AI.** It took the project
  lock with no deadline while running in a threadpool, and that lock is held
  across whole engine runs by `calibrate eval` / `ci` — so parked feedback
  requests exhausted the worker slots the process shares and the server stopped
  answering anything. Both feedback routes now wait the same bounded window as
  every other mutating route and answer `423`.
- **The fine-tune prove-it gate cannot accept on evidence it does not have.**
  Every comparability check sat inside the "some tests were training prompts"
  branch, so a project with no overlap took an unguarded path; and the check that
  did run compared how MANY held-out tests each run graded, not which ones. Both
  rates are now computed over the tests the two runs actually share, and no
  shared test means it refuses to judge.
- **The promptfoo export cannot be made to execute by the spec.** The system
  prompt was wrapped in `{% raw %}`, whose terminator a spec can contain in any
  of several spellings; promptfoo registers `process.env` as a template global,
  so the region after it could read the operator's keys into a prompt sent to a
  third-party model. The delimiters are escaped instead.
- **Request bodies are bounded on both servers.** The cap lived in `calibrate
  serve` only, so `calibrate run` — same guard, same `--host` exposure, no
  authentication — accepted unbounded bodies. It belongs to `install_guard` now,
  which both servers call, and an announced oversize body is refused at the ASGI
  boundary before the application is entered.
- **An ingest that could read nothing leaves the project alone.** It reported a
  green ✓ and exit 0 while replacing the materials, facts, gaps and index with
  emptiness; a populated source that yields no text is a failure, and the project
  is untouched.
- **Text is decoded in the encoding it was written in.** UTF-32's byte-order mark
  begins with UTF-16's, NUL is valid UTF-8 (so BOM-less UTF-16 "decoded" fine),
  the wrong endianness of ASCII text is valid CJK, and cp1252 maps almost every
  byte — so a single damaged character turned a whole document into mojibake.
  Each of those ingested garbage as your corpus.
- **A corrupt `golden.json` fails the gate instead of skipping it.** The loader
  answers "absent" and "unreadable" alike so a hand-edit cannot traceback, and
  the gate could not tell them apart — so an unresolved merge conflict silently
  disabled the pinned snapshot check and the stage reported `skip`.
- **Deleting a project no longer destroys the lock providing its own mutual
  exclusion.** On POSIX the tree is renamed aside while the lock is held, so a
  waiter cannot acquire a fresh lock at the same path mid-delete; on Windows,
  where that rename is refused, the open handle is exactly what keeps the
  exclusion intact.
- **A rewrite keeps the permissions its owner set.** Every artifact went out
  through a temp file carrying `mkstemp`'s private 0600, so a `chmod` made to
  serve a report or let a CI user read a scorecard was reverted by the next
  command that rewrote it.
- **The calibration report cannot overstate.** `calibrate report` printed a
  confidence computed differently from the report and certificate it wrote in the
  same breath; duplicate test ids let one passing result satisfy two suite rows;
  and deleting a test the run failed raises the headline, which is allowed but is
  now named in the report.
- **`calibrate examples --import` and `--dedup` re-check the spec against the
  load they mutate**, `teach` no longer writes a build bundle without one, and
  `import` validates its derived project name before spending an engine call.
- Unparsable lines in the feedback inbox are kept rather than truncated away with
  the records that were absorbed — they were the only copy of what someone sent.
- Both feedback size caps count `reason`, which was outside the limit and became
  a permanent test input.
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
