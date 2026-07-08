"""RAG indexing — guarded (the `rag` extra is heavy; skip when absent).

Closes a real coverage gap: build_index had NO tests because the rag extra was
never installed in this session's CI. These run wherever lancedb +
sentence-transformers are present."""

import pytest

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("sentence_transformers")

from calibrator import rag  # noqa: E402


def test_index_available_true_when_extra_present():
    assert rag.index_available() is True


def test_build_index_embeds_and_is_queryable(tmp_path):
    records = [
        {"id": "c1", "text": "Returns accepted within 30 days with a receipt.", "source": "p.md"},
        {"id": "c2", "text": "Refunds go to the original payment method.", "source": "p.md"},
        {"id": "c3", "text": "Final-sale items cannot be returned.", "source": "p.md"},
    ]
    n = rag.build_index(tmp_path, records)
    assert n == 3
    assert (tmp_path / "knowledge.lancedb").exists()

    # the index is genuinely queryable — semantic retrieval works
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(rag.EMBED_MODEL)
    db = lancedb.connect(str(tmp_path / "knowledge.lancedb"))
    tbl = db.open_table(rag.TABLE)
    qv = model.encode(["how do I get my money back?"]).tolist()[0]
    hits = tbl.search(qv).limit(1).to_list()
    assert hits[0]["id"] == "c2"          # the refund chunk is nearest


def test_build_index_empty_records(tmp_path):
    assert rag.build_index(tmp_path, []) == 0     # no crash, no table


def test_ingest_reports_indexed_count(tmp_path):
    """The full ingest wiring: with the extra present, build_index runs (not skipped)."""
    from calibrator.ingest import ingest_project
    from calibrator.models import Project

    class FakeExtractor:
        name = "fake@test"
        def complete(self, prompt, *, system=None, schema=None):
            return {"facts": ["Returns within 30 days."], "gaps": []}

    (tmp_path / "materials").mkdir()
    (tmp_path / "materials" / "p.md").write_text("Returns are accepted within 30 days with a receipt.")
    result = ingest_project(Project(name="p", goal="g"), tmp_path / "materials",
                            FakeExtractor(), project_dir=tmp_path)
    assert result.indexed == result.chunks and result.indexed >= 1   # built, not skipped (None)
