"""Embedding model, Chroma collection, and similarity search over ingested notes.

This module owns every piece of shared vector-store state so that `ingest.py`
(writes) and `agent.py` (reads) agree on the embedding model and collection.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BACKEND_DIR / "chroma"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "notes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))


@dataclass
class Chunk:
    """One retrieved passage plus where it came from."""

    text: str
    source: str
    page: int
    distance: float
    header_path: str = ""
    chunk_index: int = 0

    @property
    def citation(self) -> str:
        # page 0 means the format has no pages (.md, .txt).
        base = f"{self.source} p.{self.page}" if self.page else self.source
        return f"{base} - {self.header_path}" if self.header_path else base


@functools.lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the sentence-transformers model once per process (a few hundred MB)."""
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts as unit vectors, so L2 distance ranks the same as cosine."""
    vectors = get_embedder().encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


@functools.lru_cache(maxsize=1)
def get_collection() -> chromadb.api.models.Collection.Collection:
    """Open (or create) the on-disk Chroma collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Embeddings are precomputed and normalized in `embed`, so the collection's
    # default distance function is left alone.
    return client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve(question: str, k: int = TOP_K) -> list[Chunk]:
    """Return the k chunks closest to `question`, nearest first."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=embed([question]),
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        Chunk(
            text=doc,
            source=str(meta.get("source", "unknown")),
            page=int(meta.get("page", 0)),
            distance=float(dist),
            header_path=str(meta.get("header_path", "")),
            chunk_index=int(meta.get("chunk_index", 0)),
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def format_context(chunks: Iterable[Chunk]) -> str:
    """Render chunks as a numbered, citable block for the model's prompt."""
    return "\n\n".join(
        f"[{i}] ({chunk.citation})\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )


def stats() -> dict[str, object]:
    """Collection summary, used by the /health endpoint."""
    collection = get_collection()
    sources = set()
    if collection.count():
        for meta in collection.get(include=["metadatas"])["metadatas"]:
            sources.add(str(meta.get("source", "unknown")))
    return {
        "collection": COLLECTION_NAME,
        "chunks": collection.count(),
        "documents": sorted(sources),
        "embed_model": EMBED_MODEL,
    }
