import pytest

from ai_calibrator.parsing import chunk_text, read_document


def test_read_text_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nHello world.")
    assert "Hello world" in read_document(f)


def test_read_missing_file_is_reported(tmp_path):
    # Reporting beats silence: returning "" made parse_materials drop the file with
    # no entry in `skipped`, so a permission-denied file — or one deleted between
    # the scan and the read — vanished from the corpus unannounced.
    with pytest.raises(ValueError, match="nope.txt"):
        read_document(tmp_path / "nope.txt")


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


class _Par:
    def __init__(self, text: str) -> None:
        self.text = text


class _Cell:
    """A table cell: a block container of its own, like python-docx's `_Cell`."""
    def __init__(self, *texts: str) -> None:
        self._items = [_Par(t) for t in texts]

    def iter_inner_content(self):
        return list(self._items)


class _Row:
    def __init__(self, *cells: _Cell) -> None:
        self.cells = list(cells)


class _Table:
    def __init__(self, *rows: _Row) -> None:
        self.rows = list(rows)


class _StubDocument:
    """python-docx's shape: paragraphs and tables are DIFFERENT block types, and
    a table's paragraphs are not the document's."""
    def __init__(self, *items) -> None:
        self._items = list(items)

    def iter_inner_content(self):
        return list(self._items)

    @property
    def paragraphs(self):
        return [i for i in self._items if isinstance(i, _Par)]

    @property
    def tables(self):
        return [i for i in self._items if isinstance(i, _Table)]


def _stub_docx(tmp_path, monkeypatch, document):
    """A .docx on disk whose parsed document is `document` (no `docs` extra)."""
    import sys
    import types
    import zipfile

    monkeypatch.setitem(sys.modules, "docx",
                        types.SimpleNamespace(Document=lambda _p: document))
    f = tmp_path / "faq.docx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("word/document.xml", b"<w:document/>")
    return f


def test_docx_table_content_is_read(tmp_path, monkeypatch):
    """An FAQ or policy matrix is a Word TABLE, and `document.paragraphs` holds
    only the body's own paragraphs — never a table cell's. Reading just those
    returned "" for a table-only file, which ingest counts as a read material
    while none of its content reaches the facts, the gaps or the index."""
    doc = _StubDocument(
        _Par("Support handbook, revision 4."),
        _Table(_Row(_Cell("Q: refunds?"), _Cell("Refunds are accepted within 30 days."))),
        _Par("Escalate anything older."),
    )
    text = read_document(_stub_docx(tmp_path, monkeypatch, doc))
    assert "Support handbook, revision 4." in text
    assert "Refunds are accepted within 30 days." in text   # the table row
    assert "Escalate anything older." in text
    # Document order: a policy row keeps the heading it sits under.
    assert text.index("revision 4") < text.index("30 days") < text.index("Escalate")


def test_docx_table_content_is_read_without_iter_inner_content(tmp_path, monkeypatch):
    """Older python-docx has no `iter_inner_content`; tables must still be read."""
    class _OldStub:
        def __init__(self, paragraphs, tables):
            self.paragraphs, self.tables = paragraphs, tables

    doc = _OldStub([_Par("Support handbook, revision 4.")],
                   [_Table(_Row(_Cell("Refunds are accepted within 30 days.")))])
    text = read_document(_stub_docx(tmp_path, monkeypatch, doc))
    assert "Support handbook, revision 4." in text
    assert "Refunds are accepted within 30 days." in text


def test_docx_nested_table_content_is_read(tmp_path, monkeypatch):
    """A cell can hold its own table — the walk has to recurse into cells."""
    inner = _Table(_Row(_Cell("Final-sale items cannot be returned.")))
    outer_cell = _Cell()
    outer_cell._items = [_Par("Exceptions:"), inner]
    doc = _StubDocument(_Table(_Row(outer_cell)))
    text = read_document(_stub_docx(tmp_path, monkeypatch, doc))
    assert "Final-sale items cannot be returned." in text


def test_docx_table_content_is_read_by_the_real_library(tmp_path):
    """End-to-end against python-docx itself, where the extra is installed."""
    import pytest

    docx = pytest.importorskip("docx")
    f = tmp_path / "faq.docx"
    d = docx.Document()
    d.add_paragraph("Support handbook, revision 4.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Q: refunds?"
    table.rows[0].cells[1].text = "Refunds are accepted within 30 days."
    d.save(str(f))
    text = read_document(f)
    assert "Support handbook, revision 4." in text
    assert "Refunds are accepted within 30 days." in text
