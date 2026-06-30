"""Parse uploaded materials into plain text, and chunk them for retrieval.

Text/markdown work with the stdlib. PDF and DOCX need the `docs` extra
(`pip install -e '.[docs]'`) and are imported lazily so the rest of the tool
runs without them.
"""

from __future__ import annotations

from pathlib import Path

# Suffixes we read directly as UTF-8 text.
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".log", ".html", ".htm",
}


def read_document(path: str | Path) -> str:
    """Return the plain-text content of a file (best effort)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(p)
    if suffix == ".docx":
        return _read_docx(p)
    # Default: treat as text. Unknown binary types yield mostly-empty/garbled
    # text and get filtered out by the caller's `.strip()` check.
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "PDF parsing needs the `docs` extra:  pip install -e '.[docs]'"
        ) from exc
    reader = PdfReader(str(p))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(p: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "DOCX parsing needs the `docs` extra:  pip install -e '.[docs]'"
        ) from exc
    document = docx.Document(str(p))
    return "\n".join(par.text for par in document.paragraphs)


def chunk_text(text: str, *, size: int = 1000) -> list[str]:
    """Split text into ~`size`-char chunks by packing whole paragraphs.

    Paragraphs (blank-line separated) stay intact where possible; an oversized
    single paragraph is hard-split as a fallback.

    ``size`` must be a positive integer; ``size <= 0`` is rejected (a zero step
    would otherwise raise an opaque ``range()`` error in the hard-split path).
    """
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"chunk size must be an integer >= 1 (got {size!r})")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)

    # Hard-split any chunk that is still much larger than the target.
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= size * 2:
            out.append(chunk)
        else:
            for i in range(0, len(chunk), size):
                out.append(chunk[i : i + size])
    return out
