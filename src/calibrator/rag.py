"""Optional retrieval index over the materials (RAG).

Lazily uses LanceDB + sentence-transformers (the `rag` extra). If they aren't
installed, indexing gracefully no-ops and the rest of ingest still works — the
gap extraction does not depend on the vector index. The index is consumed later
(M3 compile / runtime retrieval).
"""

from __future__ import annotations

from pathlib import Path

EMBED_MODEL = "all-MiniLM-L6-v2"
TABLE = "chunks"


def index_available() -> bool:
    """True if the `rag` extra is installed."""
    try:
        import lancedb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def build_index(project_dir: str | Path, records: list[dict]) -> int | None:
    """Embed `records` (each: {id, text, source}) into a LanceDB table under the
    project dir. Returns the number of chunks indexed, or ``None`` if the `rag`
    extra isn't installed (caller treats None as "skipped")."""
    try:
        import lancedb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    if not records:
        return 0

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([r["text"] for r in records]).tolist()
    rows = [{"vector": vec, **rec} for vec, rec in zip(vectors, records)]

    db = lancedb.connect(str(Path(project_dir) / "knowledge.lancedb"))
    db.create_table(TABLE, data=rows, mode="overwrite")
    return len(rows)
