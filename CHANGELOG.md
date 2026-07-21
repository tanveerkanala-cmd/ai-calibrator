# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `all` package extra (`pip install 'ai-calibrator[all]'`) — every engine, the
  web UI, and document ingestion in one install.
- Host/Origin guard on the `calibrate run` OpenAI-compatible endpoint (shared
  with `calibrate serve` via `webguard.py`) — blocks browser CSRF and DNS
  rebinding against the local server.

### Changed
- Renamed the import package `calibrator` → `ai_calibrator` (the `calibrate`
  command is unchanged) to avoid a top-level import clash with existing PyPI
  packages.

### Fixed
- **Recompile no longer discards pinned regressions.** `calibrate compile` now
  preserves deterministic `add-check` criteria, red-team criteria, `fb_*`/`rt_*`
  regression tests, and edge cases instead of regenerating the suite from
  scratch (the "can never silently regress" guarantee).
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
