# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Security
- **`/api/import` and `/api/merge/apply` no longer adopt a directory the tool
  did not create.** Both gated only on `project.yaml`, so a project whose name
  collided with an ordinary folder in the served root — which defaults to the
  directory you are standing in — wrote itself *into* that folder. `DELETE
  /api/projects/{name}` then removed the whole tree, the user's own files with
  it: no stash, no confirmation, `ignore_errors=True`. Reproduced end to end.
  `create_project` already refused exactly this, with a comment naming the
  hazard; that refusal is now one helper all three routes call, and in the
  import route it runs *before* the engine call, so a refused name never spends
  a token. **Anyone on 0.0.2 or earlier should upgrade.**
- **Exported promptfoo assertion operands can no longer read the environment.**
  promptfoo renders assertion *values* through Nunjucks — verified against
  0.122.0, where a `{{ env.PATH }}` in an operand evaluated to the real PATH.
  Operands are now escaped, so a spec that quotes a template tag stays text.

### Fixed
The sixth audit of this repo: 50 verified defects, each with a regression test
confirmed failing before its fix. The findings that change what a number means:

- **The prove-it gate now scores rows the candidate never trained on.** It
  replayed its own training set, so a model that had merely memorised it scored
  perfect agreement and passed. The bundle writer and the gate now partition the
  log with one shared rule (hashing the question, so a row cannot migrate sides
  as the log grows), the gate subtracts whatever the shipped dataset actually
  contains, and it *refuses* rather than certifying when the log is too small to
  hold anything back. It also scored against the raw cloud log instead of the
  ground-truth-corrected answers the bundle ships — failing a candidate for
  having learned a human's correction — counted one question logged N times as N
  samples, and let a retracted verdict outrank the label that superseded it.
- **Drift compares like with like.** Re-minted tests the comparison had already
  excluded still drove the delta, so a recompile read as a regression and the CI
  stage failed on a population its own message did not describe. The delta is
  tallied over exactly the compared ids; "nothing was comparable" is one state
  (`delta is None`) every layer reads the same way; `drift` exits 1 there instead
  of printing a green tick beside a -100%.
- **`--tolerance` does something again.** Confining the delta to the comparable
  population left it unable to change any verdict. It now gates the share of
  compared tests that flipped pass→fail — identical to the old rule at the
  default of `0`, exhaustively, on all 1,398,100 pass/fail shapes a suite of one
  to ten comparable tests can take.
- **The judge sees the knowledge the subject was given.** RAG-grounded answers
  were graded as invented, because the judge got the un-augmented system prompt.
  The retrieved section now rides in the per-test prompt, leaving the cached
  system message untouched.
- **An off-scale judge score follows its own verdict** instead of clamping to a
  weighted 100% sitting beside a 0% pass rate.
- **Human labels survive a corrupt labels file.** `save_labels` merged onto a
  failed read and rewrote the file whole, silently discarding every label saved
  before it. It now refuses, naming the file.
- **DOCX ingestion reads tables.** `.paragraphs` excludes them, so policy tables
  in a user's materials reached the tool as nothing at all.
- **`rightsize` ranks across providers.** It called a Claude model "cheapest"
  while silently dropping every non-Claude candidate that passed, because the
  price table held only Claude ids.
- **The exported suite behaves like the eval that certified it.** Checks whose
  promptfoo assertion grades differently now export as `javascript` that
  normalises the way `run_check` does; the escape round-trips every delimiter;
  duplicate ids no longer multiply a criterion's weight; the runner sends the
  turn shape the eval graded; the one shape promptfoo cannot reproduce is stated
  in the file.
- **The training tier runs on hardware it was not written on**: bf16 only where
  the card reports Ampere+ and fp16 where it does not (T4/V100/RTX 20xx could not
  start training at all), a merge that loads at the size it saves, a
  `learning_rate` in exponent form that actually reaches the trainer, and
  hand-edited hyperparameters that survive a re-export in both tiers.
- **The suite was red at HEAD** in any environment with uvicorn but without the
  anthropic SDK — including this repo's own `.venv` — because two boot-gate tests
  assumed a cloud SDK was installed and the full-extras job masked it. They now
  run with those SDKs made unimportable, so the engine-free promise is tested
  everywhere.
- Smaller: an interrupted interview no longer resumes as complete; `diff` prints
  an examples-only change; `judge-check` continues past the verdicts already
  labelled and builds ground truth without logging; `examples --import/--dedup`
  refreshes `build/`; `train-engine` refuses an empty bundle and names the
  project in its next-step command; the judge-is-subject warning survives two
  spellings of one engine; a merged spec no longer orders grounding in an index
  it has none of; `.githooks/pre-push` is installed rather than merely claiming
  to be.

Three failures found by pointing `calibrate compare` at this repo's own docs
with local models (subject llama3.2:3b, judge gemma4:12b, via Ollama) — the
first real run of the experiment the tool exists for:

- **The judge's schema now demands exactly the rows being graded.** The
  results array was unbounded, and a grammar-constrained local model accepts
  that invitation: decoding loops, appending rows (or growing one rationale)
  until the output limit kills the call — and a truncated grade is an engine
  error that takes the whole run with it. `judge_schema(n)` bounds the array
  to the batch and caps the free-text field; the Anthropic and OpenAI adapters
  strip those bounds before sending (their schema dialects reject array-size
  and string-length keywords, and constrained decoding doesn't grammar-loop
  there, so nothing is lost).
- **Ollama schema calls no longer pay for invisible thinking.** A thinking
  model spends its output budget on unconstrained thinking BEFORE the
  grammar-constrained JSON — ~12K characters of it per judge call in the run
  that surfaced this — flakily starving the actual output past `num_predict`.
  A structured call's entire product is the JSON, so `think` is off for those
  calls; plain subject calls are untouched, because the subject's answers are
  the thing being measured.
- **`compare` reports retrieval as it ran, not as it was enabled.** Passing a
  project directory enables retrieval; it does not make it happen. A project
  with no usable index retrieves nothing, and the first live report said
  "retrieval ON" for a prompt-only bot. The report now probes the index the
  same way eval's own retrieval-off warning does.

## [0.0.2] — 2026-08-19

### Added
- **`calibrate compare` — the experiment the tool exists to run.** Every other
  command measures the calibrated AI against your standards; nothing measured
  what the calibration *bought*. `compare` runs the identical suite twice on
  the same model — once as deployed (compiled prompt + RAG when indexed), once
  as a baseline — and reports the delta. The default baseline gets your
  one-line goal as its whole prompt: what a person gets by pasting their ask
  into a chat window (beating a model that was never told the job proves
  nothing; `--vs bare` keeps that floor available, off the default). The judge
  grades both sides under the same context — your compiled spec — because two
  runs graded under different context are not comparable; deterministic checks
  are reported apart from judged criteria, since that part of the delta owes
  nothing to any judge's opinion; the judge-is-the-subject warning prints here
  too, where it matters twice; and a tied or losing calibration is reported in
  exactly those words — this is an instrument, not a gate. The summary goes to
  `evals/compare.json` and never into the run history `drift` and `ci`
  baseline against, so a baseline-configuration run can never poison a later
  comparison.

## [0.0.1] — 2026-08-19

Initial release: ingest → interview → compile → eval/refine → red-team →
drift → serve/export pipeline, guided + advanced (fine-tuning) modes, web UI,
Claude/OpenAI/Ollama engines.

### Added
- **The web UI was opened in a browser for the first time, and three things it
  found.** Every API test drives `api.py` through TestClient, which proves the
  routes work and says nothing about whether the shipped UI calls the routes
  that exist. `tests/test_web_contract.py` now checks that every endpoint
  `app.js` calls is one `api.py` serves, that the report is rendered rather than
  dumped, that the renderer escapes before it marks up, and that the assets are
  served revalidating.
- **Commands that scale with your data say what they will spend, first.**
  `interview` makes one engine call per gap, `rightsize` runs the whole suite
  once per model and grades every answer, and `eval --refine` repeats the suite
  up to `--rounds` times. Each of those numbers is knowable in advance and none
  of them was shown, so pointing this at a large folder with a metered API key
  cost whatever it cost. A printed estimate, not a confirmation prompt: these
  run in scripts and over the API, where a blocking prompt would hang them.
  The estimates are covered by tests that check them against the calls actually
  made, because a wrong estimate is worse than none.
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
- **The calibration report was shown as raw markdown.** The UI wrote it into a
  `<pre>` as source, so the non-technical owner the product is built for saw
  `## Coverage`, `**67%**` and backticks instead of a report — on the one
  artifact the whole tool points at. It is rendered now, by a small renderer
  that escapes first and supports no link syntax, since every word in that
  document was written by a model from ingested files.
- **A failing test named its rationales but not its criteria.** The weak-spot
  list used `rationale or criterion_id`, dropping the id whenever the judge
  supplied any text, so a test that failed two criteria read as two rationales
  joined by "; " — and two rationales from the same judge tend to sound alike.
  It is the part of the certificate someone acts on, so it now says what broke.
- **The UI could keep running against an API it no longer matches.** The asset
  URLs never change (no build step, no content hash) and were served with only
  an etag, leaving a browser free to reuse `app.js` without asking — so
  upgrading could leave a tab running the old UI against the new API. Observed
  in a real browser, where even a reload kept the stale file. They are served
  `no-cache` now, which means revalidate, not "don't store": the etag still
  answers 304.
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
