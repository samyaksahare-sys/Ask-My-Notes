"""Embedding model, Chroma collection, and similarity search over ingested notes.

This module owns every piece of shared vector-store state so that `ingest.py`
(writes) and `agent.py` (reads) agree on the embedding model and collection.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BACKEND_DIR / "chroma"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "notes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# Smaller than the model's native width; keeps the index compact and is ample
# for a personal corpus. Non-default widths must be normalized (below).
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
# Per-request cap. 64 is accepted; larger batches start hitting rate limits.
EMBED_BATCH = 64
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
def get_client() -> genai.Client:
    """Gemini client, shared by embedding and answering.

    Cached: a Client closes its HTTP transport when garbage collected, so a
    throwaway client can have its connection closed mid-request.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _normalize(vector: Sequence[float]) -> list[float]:
    """Scale to unit length so L2 distance ranks the same as cosine."""
    length = math.sqrt(sum(v * v for v in vector))
    return [v / length for v in vector] if length else list(vector)


def _embed(texts: Sequence[str], task_type: str) -> list[list[float]]:
    """Embed via the Gemini API, in batches, returned as unit vectors."""
    out: list[list[float]] = []
    texts = list(texts)
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        response = get_client().models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            ),
        )
        out.extend(_normalize(e.values) for e in response.embeddings)
    return out


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed passages for storage."""
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a question.

    Deliberately a different task_type from documents: the model places a
    question and the passage that answers it nearer each other than a symmetric
    embedding would, which measurably improves recall.
    """
    return _embed([text], "RETRIEVAL_QUERY")[0]


# Back-compat alias: existing callers that embed passages.
embed = embed_documents


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
        query_embeddings=[embed_query(question)],
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
