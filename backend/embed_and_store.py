"""Embed chunks from `ingest` and persist them to the local Chroma collection.

This module owns the write path into the vector store. `ingest` decides where
chunk boundaries fall; everything about turning those chunks into vectors and
rows lives here, so there is exactly one place that defines what a stored
record looks like.

Usage:
    python -m backend.embed_and_store              # index every PDF in data/
    python -m backend.embed_and_store --reset      # wipe the collection first
    python -m backend.embed_and_store notes.pdf    # index specific files
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from backend.ingest import SUPPORTED_SUFFIXES, Chunk, chunk_document, read_document
from backend.retrieval import DATA_DIR, embed_documents, get_collection

# Rows per collection.add call. Keeps peak memory bounded on large PDFs while
# still giving sentence-transformers a batch big enough to be worth the call.
BATCH_SIZE = 64


def chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Stable, unique id for one chunk.

    The index makes the id unique even when the same text appears twice in a
    document (a repeated heading or boilerplate line); the content hash makes
    the id change when the text changes, so re-ingesting an edited note does
    not leave a stale row behind under a recycled id.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}:{chunk_index:04d}:{digest}"


def build_metadata(source: str, chunk_index: int, chunk: Chunk) -> dict[str, object]:
    """Metadata stored alongside each vector.

    Chroma accepts str/int/float/bool only - no None - so absent header paths
    are stored as an empty string rather than omitted, keeping the schema
    uniform across markdown and plain-text notes.
    """
    return {
        "source": source,
        "chunk_index": chunk_index,
        "page": chunk.page,
        "header_path": chunk.header_path,
    }


def store_chunks(source: str, chunks: list[Chunk], replace: bool = True) -> int:
    """Embed `chunks` and write them to the collection. Returns rows written.

    `replace` drops any existing rows for this source first, which is what
    makes re-indexing an edited note idempotent instead of additive.
    """
    collection = get_collection()

    if replace:
        collection.delete(where={"source": source})

    if not chunks:
        return 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [chunk.text for chunk in batch]
        collection.add(
            ids=[
                chunk_id(source, start + offset, chunk.text)
                for offset, chunk in enumerate(batch)
            ],
            documents=texts,
            # Same model as query time, but task_type RETRIEVAL_DOCUMENT
            # rather than RETRIEVAL_QUERY - the asymmetry is intentional and
            # improves recall. Both are normalized identically.
            embeddings=embed_documents(texts),
            metadatas=[
                build_metadata(source, start + offset, chunk)
                for offset, chunk in enumerate(batch)
            ],
        )

    return len(chunks)


def embed_pdf(pdf_path: Path) -> int:
    """Chunk, embed, and store one note (.pdf, .md, .txt). Returns chunks stored."""
    source = pdf_path.name
    chunks = chunk_document(read_document(pdf_path))

    if not chunks:
        # Still clear old rows: a note emptied of text should leave nothing behind.
        store_chunks(source, [], replace=True)
        print(f"  {source}: no extractable text (scanned PDF? needs OCR)")
        return 0

    stored = store_chunks(source, chunks)
    structured = sum(1 for c in chunks if c.header_path)
    shape = f", {structured} under headers" if structured else ""
    print(
        f"  {source}: {stored} chunks from "
        f"{len({c.page for c in chunks})} pages{shape}"
    )
    return stored


def embed_all(paths: list[Path] | None = None, reset: bool = False) -> int:
    """Index the given PDFs, or every PDF in data/ when none are given."""
    if reset:
        collection = get_collection()
        existing = collection.get(include=[])["ids"]
        if existing:
            collection.delete(ids=existing)
        print(f"Reset collection ({len(existing)} chunks removed)")

    pdfs = paths or sorted(
        f for f in DATA_DIR.iterdir() if f.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not pdfs:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        print(f"No documents found in {DATA_DIR} ({supported}) - add some and re-run.")
        return 0

    print(f"Indexing {len(pdfs)} document(s) from {DATA_DIR}")
    return sum(embed_pdf(pdf) for pdf in pdfs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", type=Path, help="specific files to index")
    parser.add_argument("--reset", action="store_true", help="clear the collection first")
    args = parser.parse_args()

    paths = [p if p.exists() else DATA_DIR / p.name for p in args.pdfs]
    total = embed_all(paths or None, reset=args.reset)
    print(f"Done - {total} chunks indexed.")


if __name__ == "__main__":
    main()
