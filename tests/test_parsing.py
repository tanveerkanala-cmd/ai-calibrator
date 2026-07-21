from ai_calibrator.parsing import chunk_text, read_document


def test_read_text_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nHello world.")
    assert "Hello world" in read_document(f)


def test_read_missing_file_is_empty(tmp_path):
    assert read_document(tmp_path / "nope.txt") == ""


def test_chunk_packs_paragraphs_within_size():
    text = "\n\n".join("para " + str(i) * 40 for i in range(12))
    chunks = chunk_text(text, size=200)
    assert chunks
    # packed chunks stay within the hard-split ceiling (2x target)
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_empty_text():
    assert chunk_text("   \n\n  ") == []


def test_capped_join_bounds_extracted_text(monkeypatch):
    """Decompression-bomb guard: extracted text stops growing at the cap."""
    from ai_calibrator import parsing

    monkeypatch.setattr(parsing, "MAX_EXTRACTED_CHARS", 25)
    joined = parsing._capped_join(iter(["a" * 10, "b" * 10, "c" * 10, "d" * 10]))
    assert len(joined) <= 25
    assert "d" not in joined  # iteration stopped — later parts never accumulated


def test_zip_streaming_cap_counts_real_bytes_not_declared(tmp_path):
    """The cap streams the ACTUAL decompressed bytes; it must not trust the size
    the archive declares (ZipInfo.file_size is attacker-forgeable — the bug the
    old `sum(info.file_size)` check had). A 2 MB member that is tiny on disk
    still trips a small cap."""
    import io
    import zipfile

    import pytest

    from ai_calibrator import parsing

    # 2 MB of one byte → compresses to a few KB on disk, but decompresses to 2 MB.
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"A" * (2 * 1024 * 1024))
    raw.seek(0)
    with zipfile.ZipFile(raw) as z:
        # cap well below the real 2 MB → must raise while streaming, regardless
        # of whatever file_size the central directory reports.
        with pytest.raises(ValueError, match="safety cap"):
            parsing._zip_decompressed_size_capped(z, 512 * 1024)
        # an honestly-small member passes
    ok = io.BytesIO()
    with zipfile.ZipFile(ok, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"small")
    ok.seek(0)
    with zipfile.ZipFile(ok) as z:
        parsing._zip_decompressed_size_capped(z, 512 * 1024)  # no raise


def test_docx_decompression_cap(monkeypatch, tmp_path):
    """End-to-end: an oversized .docx is refused before python-docx parses it."""
    import zipfile

    import pytest

    pytest.importorskip("docx")
    from ai_calibrator import parsing

    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"x" * 4096)
    monkeypatch.setattr(parsing, "MAX_DOCX_DECOMPRESSED_BYTES", 1024)
    with pytest.raises(ValueError, match="safety cap"):
        parsing.read_document(bomb)


def test_corrupt_docx_is_a_clean_error(tmp_path):
    """Not-a-zip .docx raises a friendly ValueError, not a raw BadZipFile."""
    import pytest

    pytest.importorskip("docx")
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError, match="not a valid"):
        read_document(bad)
