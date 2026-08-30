"""FastAPI service for Ask My Notes.

Run with:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import math
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import agent, ingest as ingest_module
from backend.agent import ApiFailure
from backend.retrieval import TOP_K, stats

load_dotenv()

# Origins a browser frontend may call this API from. Defaults cover the usual
# local dev servers; override with a comma-separated CORS_ORIGINS in .env.
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]

# Upstream status -> what this service reports.
#
# 503 tells a client "busy, retry shortly"; 500 tells it "broken, do not
# bother" - so a transient upstream outage must not surface as 500. A 4xx from
# Gemini means this service is misconfigured (bad key, bad request), which a
# client retry cannot fix, so it becomes 502 rather than passing the 4xx
# through and implying the caller sent something wrong.
_TRANSIENT_TO_503 = {408, 500, 502, 503, 504}

app = FastAPI(
    title="Ask My Notes",
    description="RAG over your own PDF notes, answered by Gemini.",
    version="0.1.0",
)

_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
    if o.strip()
]
# Browsers reject "*" together with credentialed requests, so a wildcard turns
# credentials off rather than producing a config that silently fails in the
# browser but looks fine here.
_wildcard = "*" in _origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=not _wildcard,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


class Source(BaseModel):
    source: str
    page: int
    distance: float
    excerpt: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]


def _raise_for_failure(failure: ApiFailure) -> None:
    """Translate an upstream Gemini failure into the right HTTP response."""
    if failure.status == 429:
        status = 429
    elif failure.status in _TRANSIENT_TO_503:
        status = 503
    else:
        status = 502

    headers = None
    # Only advertise Retry-After when waiting is actually useful - an exhausted
    # daily quota will not clear on any timescale a client should wait for.
    if failure.transient and failure.retry_after:
        headers = {"Retry-After": str(max(1, math.ceil(failure.retry_after)))}
    elif failure.transient and status == 503:
        headers = {"Retry-After": "5"}

    raise HTTPException(status_code=status, detail=failure.message, headers=headers)


def _to_response(result: agent.Answer) -> "AskResponse":
    """Shared serialization for /ask and /chat."""
    return AskResponse(
        answer=result.text,
        refused=result.refused,
        sources=[
            Source(
                source=chunk.source,
                page=chunk.page,
                distance=chunk.distance,
                excerpt=chunk.text[:300],
            )
            for chunk in result.sources
        ],
    )


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness plus a summary of what is currently indexed."""
    return {"status": "ok", **stats()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = agent.answer(request.question, k=request.top_k)
    if result.failure:
        _raise_for_failure(result.failure)
    return _to_response(result)


@app.post("/chat", response_model=AskResponse)
def chat(request: ChatRequest) -> AskResponse:
    """Agentic endpoint: the model chooses whether to search notes, calculate, or both.

    Costs 2-3 upstream requests per query (one per tool round-trip), against
    /ask's one. `sources` holds every chunk the model retrieved across all of
    its searches, deduplicated.
    """
    result = agent.answer_with_tools(request.query, k=request.top_k)
    if result.failure:
        _raise_for_failure(result.failure)
    return _to_response(result)


@app.post("/ingest")
def reindex(reset: bool = False) -> dict[str, object]:
    """Re-ingest every PDF in backend/data/. Synchronous - it can take a while."""
    try:
        chunks = ingest_module.ingest(reset=reset)
    except Exception as exc:  # surface parse/embed failures to the caller
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    return {"status": "ok", "chunks_indexed": chunks, **stats()}
