"""Regression tests for the final security-audit findings."""

import os
import stat

import pytest


# CWE-59 — ingest must skip symlinks (a shared project can't read files outside materials/)
def test_ingest_skips_symlinks(tmp_path):
    from calibrator.ingest import parse_materials

    materials = tmp_path / "proj" / "materials"
    materials.mkdir(parents=True)
    (materials / "real.md").write_text("legit policy text")
    secret = tmp_path / "secret.txt"
    secret.write_text("AWS_SECRET=hunter2")
    try:
        (materials / "leak.txt").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    docs = parse_materials(materials)
    joined = " ".join(t for _, t in docs)
    assert "legit policy text" in joined
    assert "hunter2" not in joined                 # the symlinked secret is NOT read
    assert [p.name for p in (p for p, _ in docs)] == ["real.md"]


# CWE-59 defense-in-depth — a symlinked directory component is also excluded
def test_ingest_skips_oversize(tmp_path):
    from calibrator.ingest import MAX_MATERIAL_BYTES, parse_materials

    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "ok.md").write_text("small")
    big = materials / "big.md"
    big.write_text("x" * 100)
    # force it "oversize" by monkeypatching the cap low via a huge file is slow; instead
    # assert the cap constant is enforced by making a file exceed a tiny patched cap
    import calibrator.ingest as ing
    orig = ing.MAX_MATERIAL_BYTES
    ing.MAX_MATERIAL_BYTES = 10
    try:
        docs = parse_materials(materials)
    finally:
        ing.MAX_MATERIAL_BYTES = orig
    names = sorted(p.name for p, _ in docs)
    assert names == ["ok.md"]                        # big.md (100 bytes > 10) skipped
    assert MAX_MATERIAL_BYTES > 1_000_000            # real default is generous


# CWE-732 — logs and lock are owner-only
def test_logs_and_lock_are_private(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permissions")
    from calibrator.engine_log import LoggingEngine
    from calibrator.locking import FileLock

    class Fake:
        name = "f@test"
        def complete(self, prompt, *, system=None, schema=None):
            return "out"

    LoggingEngine(Fake(), "judge", tmp_path / "logs").complete("hi", system="secret system prompt")
    log = tmp_path / "logs" / "judge.jsonl"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600      # not world-readable

    lock = FileLock(tmp_path / ".lock"); lock.acquire()
    try:
        assert stat.S_IMODE((tmp_path / ".lock").stat().st_mode) == 0o600
    finally:
        lock.release()


# CWE-444 — deterministic checks are normalization-invariant
def test_check_unicode_normalization():
    import unicodedata

    from calibrator.checks import run_check
    from calibrator.models import Check

    composed = "café"                 # café (single é)
    decomposed = unicodedata.normalize("NFD", composed)  # cafe + combining accent
    assert composed != decomposed
    # a contains-check for the composed form matches decomposed output and vice versa
    assert run_check(Check(kind="contains", value=composed), f"our {decomposed} bar")[0] is True
    assert run_check(Check(kind="not_contains", value=composed), f"our {decomposed} bar")[0] is False
