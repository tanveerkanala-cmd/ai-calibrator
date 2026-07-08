"""Optional retrieval index over the materials (RAG).

Lazily uses LanceDB + sentence-transformers (the `rag` extra). If they aren't
installed, indexing gracefully no-ops and the rest of ingest still works — the
gap extraction does not depend on the vector index. The index is consumed later
(M3 compile / runtime retrieval).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

EMBED_MODEL = "all-MiniLM-L6-v2"
TABLE = "chunks"
TOP_K = 5  # chunks retrieved per query (matches rag_config)


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
    rows = [{"vector": vec, **rec} for vec, rec in zip(vectors, records, strict=True)]

    db = lancedb.connect(str(Path(project_dir) / "knowledge.lancedb"))
    db.create_table(TABLE, data=rows, mode="overwrite")
    return len(rows)


# --- retrieval: consumed by eval and the runtime so the AI you TEST is the ---
# --- RAG-augmented AI you SERVE (both call augment_system identically) --------

@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def retrieve(project_dir: str | Path, query: str, top_k: int = TOP_K) -> list[str]:
    """Top-k chunk texts most similar to ``query`` from the project's index.

    Best-effort and fully graceful: returns ``[]`` when the query is empty, the
    ``rag`` extra isn't installed, no index exists, or anything goes wrong — so
    retrieval NEVER breaks an eval or the serving endpoint. It checks the index
    path before importing the (heavy) embedder, so a project without an index
    pays nothing."""
    if not isinstance(query, str) or not query.strip():
        return []
    db_path = Path(project_dir) / "knowledge.lancedb"
    if not db_path.exists():
        return []
    try:
        import lancedb
        db = lancedb.connect(str(db_path))
        if TABLE not in db.table_names():
            return []
        vec = _embedder().encode([query]).tolist()[0]
        hits = db.open_table(TABLE).search(vec).limit(max(1, top_k)).to_list()
        return [h["text"] for h in hits if isinstance(h.get("text"), str)]
    except Exception:
        return []


def index_fingerprint(project_dir: str | Path) -> str:
    """A stable content hash of the knowledge index (every chunk's id+text+source),
    or '' when no index exists or the extra is absent.

    Part of the certification fingerprint: re-ingesting EDITED materials rebuilds
    the index, changing this hash, so the gate goes stale — because eval/run
    retrieve from the index, a changed index is a changed deployed AI. Editing a
    material WITHOUT re-ingesting leaves the index (and this hash) unchanged,
    which is correct: the served AI hasn't changed until you rebuild."""
    db_path = Path(project_dir) / "knowledge.lancedb"
    if not db_path.exists():
        return ""
    try:
        import hashlib
        import json

        import lancedb
        db = lancedb.connect(str(db_path))
        if TABLE not in db.table_names():
            return ""
        # Read the whole table (to_list is a SEARCH-result method, not a table
        # method) selecting only the content columns — no vectors.
        arrow = db.open_table(TABLE).to_arrow().select(["id", "text", "source"])
        items = sorted(
            (str(r.get("id", "")), str(r.get("text", "")), str(r.get("source", "")))
            for r in arrow.to_pylist()
        )
        return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""  # best-effort — a fingerprint failure must not break certification


def knowledge_block(chunks: list[str]) -> str:
    """Format retrieved chunks as a system-prompt section, or '' if none."""
    if not chunks:
        return ""
    body = "\n".join(f"- {c}" for c in chunks)
    return ("\n\nRELEVANT KNOWLEDGE (retrieved for this question — ground your answer "
            f"in it; do not invent facts it does not support):\n{body}")


def augment_system(system: str | None, project_dir: str | Path | None, query: str,
                   top_k: int = TOP_K) -> str | None:
    """Return ``system`` with a retrieved-knowledge section appended for ``query``.

    THE shared injection point: eval and the runtime both call this so the graded
    AI and the served AI receive byte-identical augmentation. No index / no
    project_dir → returns ``system`` unchanged."""
    if project_dir is None:
        return system
    block = knowledge_block(retrieve(project_dir, query, top_k))
    if not block:
        return system
    return (system or "") + block
