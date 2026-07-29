"""Parse uploaded materials into plain text, and chunk them for retrieval.

Text/markdown work with the stdlib. PDF and DOCX need the `docs` extra
(`pip install -e '.[docs]'`) and are imported lazily so the rest of the tool
runs without them.
"""

from __future__ import annotations

import codecs
from pathlib import Path

# Extracted-text ceiling for compressed formats. PDF/DOCX are zip/deflate
# containers: a small file can decompress to gigabytes ("decompression bomb")
# and OOM the ingest. The caps below bound *actual* decompression by streaming,
# never by trusting a size the file declares about itself (an attacker forges
# that — zipfile still decompresses the real stream).
MAX_EXTRACTED_CHARS = 10_000_000
MAX_DOCX_DECOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_PDF_PAGES = 10_000
MAX_PDF_PAGE_CHARS = 2_000_000  # a single page over this is a bomb signal → truncate
_DECOMP_CHUNK = 1 << 16  # 64 KiB


def _capped_join(parts, sep: str = "\n") -> str:
    """Join text parts, hard-bounded to MAX_EXTRACTED_CHARS.

    Truncates the *current* part to the remaining budget before appending, so a
    single gigantic part can't be materialized into the accumulator whole (the
    old version appended each part first, then capped — unbounded peak)."""
    out: list[str] = []
    budget = MAX_EXTRACTED_CHARS
    first = True
    for part in parts:
        if budget <= 0:
            break
        if not first:
            budget -= len(sep)
        first = False
        piece = part if len(part) <= budget else part[:budget]
        out.append(piece)
        budget -= len(piece)
    return sep.join(out)


def read_document(path: str | Path) -> str:
    """Return the plain-text content of a file (best effort)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(p)
    if suffix == ".docx":
        return _read_docx(p)
    # Default: treat as text, deciding on the CONTENT rather than the suffix — a
    # plain-text material can be a .jsonl export, a .tex draft or an extensionless
    # README, and refusing those would lose real content.
    try:
        raw = p.read_bytes()
    except OSError as exc:
        # Do NOT swallow this. Returning "" makes the caller's `.strip()` filter
        # drop the file with no entry in `skipped`, so a permission-denied file —
        # or one deleted between the scan and the read — vanishes from the corpus
        # silently. parse_materials' per-file handler turns this into a report.
        raise ValueError(f"{p.name}: {exc.strerror or exc}") from exc
    return _decode_text(p.name, raw)


def _decode_text(name: str, raw: bytes) -> str:
    """Decode a material's bytes as text, or raise ValueError if they aren't text.

    Decoding everything with ``errors="replace"`` is what let a .xlsx or a .png
    count as a successfully ingested material: the mojibake is dense
    non-whitespace, so the caller's `.strip()` check keeps it, the file's real
    content never reaches the extractor, and one image can fill the extraction
    window on its own. Refusing here puts the file in ``skipped``, where the
    owner can see it and convert it."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        # Notepad's "Unicode" .txt — real text, just not UTF-8. Read as UTF-8 it
        # decodes to NUL-interleaved characters no reader or embedder can use.
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name}: looks like UTF-16 text but could not be decoded ({exc.reason})") from exc
    if b"\x00" in raw:
        raise ValueError(f"{name}: looks like a binary file (image, spreadsheet, archive), not text")
    for encoding in ("utf-8-sig", "cp1252"):  # utf-8-sig drops a BOM; cp1252 covers legacy exports
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{name}: could not be decoded as text (unrecognized encoding)")


def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "PDF parsing needs the `docs` extra:  pip install -e '.[docs]'"
            "  (in your ai-calibrator clone)"
        ) from exc
    reader = PdfReader(str(p))
    # Page-by-page with an immediate per-page + running truncation, so a bomb
    # page can't accumulate into a multi-GB join. (Residual: pypdf materializes
    # one page's text internally before we truncate it — a single crafted page
    # can still spike memory. Bounded per file; see SECURITY.md. Only ingest
    # PDFs you trust.)
    out: list[str] = []
    budget = MAX_EXTRACTED_CHARS
    for i, page in enumerate(reader.pages):
        if i >= MAX_PDF_PAGES or budget <= 0:
            break
        text = page.extract_text() or ""
        if len(text) > MAX_PDF_PAGE_CHARS:
            text = text[:MAX_PDF_PAGE_CHARS]
        if len(text) > budget:
            text = text[:budget]
        out.append(text)
        budget -= len(text)
    return "\n".join(out)


def _zip_decompressed_size_capped(z, cap: int) -> None:
    """Stream-decompress every zip member, aborting past `cap` bytes.

    Reads the real deflate stream in bounded chunks and discards them, so peak
    memory is one chunk — NOT the decompressed size. Trusts nothing the archive
    self-declares: `ZipInfo.file_size` is attacker-controlled and does not bound
    what `zipfile` actually decompresses."""
    total = 0
    for info in z.infolist():
        with z.open(info) as fh:
            while True:
                chunk = fh.read(_DECOMP_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    raise ValueError(
                        f"decompressed size exceeds the "
                        f"{cap // (1024 * 1024)} MB safety cap (possible zip bomb)"
                    )


def _read_docx(p: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "DOCX parsing needs the `docs` extra:  pip install -e '.[docs]'"
            "  (in your ai-calibrator clone)"
        ) from exc
    import zipfile
    # Bound the REAL decompression first; only then hand a verified-safe file to
    # python-docx (which would otherwise decompress an unbounded stream).
    try:
        with zipfile.ZipFile(p) as z:
            _zip_decompressed_size_capped(z, MAX_DOCX_DECOMPRESSED_BYTES)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{p.name}: not a valid .docx (zip) file") from exc
    document = docx.Document(str(p))
    return _capped_join(par.text for par in document.paragraphs)


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
