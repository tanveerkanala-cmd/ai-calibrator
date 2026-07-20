from calibrator.parsing import chunk_text, read_document


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
    from calibrator import parsing

    monkeypatch.setattr(parsing, "MAX_EXTRACTED_CHARS", 25)
    joined = parsing._capped_join(iter(["a" * 10, "b" * 10, "c" * 10, "d" * 10]))
    assert len(joined) <= 25
    assert "d" not in joined  # iteration stopped — later parts never accumulated


def test_docx_declared_size_cap(monkeypatch, tmp_path):
    """A .docx whose zip entries declare a huge decompressed size is refused
    BEFORE the XML parser runs (zip-bomb guard)."""
    import zipfile

    import pytest

    pytest.importorskip("docx")
    from calibrator import parsing

    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w") as z:
        z.writestr("word/document.xml", "x" * 4096)
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
