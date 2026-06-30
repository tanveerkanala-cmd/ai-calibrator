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
