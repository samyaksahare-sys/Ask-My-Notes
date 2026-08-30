"""Retrieval eval: does the right document come back in the top-k?

This measures the retrieval half of the pipeline only - no LLM call, so it
costs nothing, needs no API key, and is unaffected by Gemini quota. If answers
are wrong, run this first: a wrong answer built on the right chunks is a
prompting problem, but a wrong answer built on the wrong chunks is a chunking
or embedding problem, and only this tells you which.

Usage:
    python eval.py                  # evaluate against whatever is indexed
    python eval.py --fixtures       # index a known demo corpus first, then run
    python eval.py --k 3            # tighter cutoff than the configured TOP_K
    python eval.py --verbose        # show every retrieved hit, not just misses

Swap CASES for questions about your own notes, with `expected` naming the file
you would expect to answer each one.
"""

from __future__ import annotations

import argparse
import sys

from backend.retrieval import TOP_K, get_collection, retrieve

# (question, expected source filename)
CASES: list[tuple[str, str]] = [
    ("how do b-trees compare to LSM trees for writes?", "storage-engines.pdf"),
    ("when is a covering index worth the cost?", "storage-engines.pdf"),
    ("what durability tradeoff does synchronous replication make?", "storage-engines.pdf"),
    ("which algorithm does Chroma use for approximate search?", "vector-search.pdf"),
    ("why does normalizing embeddings matter for distance?", "vector-search.pdf"),
    ("how big should a chunk be relative to the embedding model?", "chunking.pdf"),
    ("does overlap between chunks actually help retrieval?", "chunking.pdf"),
    ("what do we pay per month for each managed index?", "costs.pdf"),
    ("how much did embedding cost us last quarter?", "costs.pdf"),
    ("what stops the model from inventing facts not in my notes?", "grounding.pdf"),
]

# A small corpus the 10 cases above are written against, so the eval is
# runnable before you have your own notes indexed. Text is indexed directly
# through the normal chunking path - no PDFs needed, so this works anywhere.
FIXTURE_DOCS: dict[str, str] = {
    "storage-engines.pdf": """# Storage engines

## Indexing

B-trees dominate OLTP workloads because range scans stay cheap and the tree
stays shallow even at large row counts. LSM trees invert the tradeoff: writes
are sequential and fast, while reads pay a merge cost across levels.

### Covering indexes

A covering index answers the query entirely from the index, so the engine never
touches the heap. Worth it for hot read paths, costly on write-heavy tables.

## Replication

Leader-follower replication is the default arrangement. Synchronous replication
trades write latency for stronger durability guarantees.
""",
    "vector-search.pdf": """# Vector search

Approximate nearest neighbour beats exact search once the corpus is large.
HNSW builds a navigable small-world graph and is what Chroma uses underneath.

## Distance

When vectors are normalized to unit length, Euclidean distance ranks results
identically to cosine similarity, so the choice of metric stops mattering.
""",
    "chunking.pdf": """# Chunking

Chunk size should never exceed the embedding model's maximum sequence length.
Anything past that limit is silently truncated at embedding time: the text is
still stored, but contributes nothing to whether the chunk is ever retrieved.

## Overlap

A modest overlap of ten to fifteen percent keeps a definition attached to the
sentence that uses it, at the cost of some duplication in the index.
""",
    "costs.pdf": """# Infrastructure costs

## Vector database

The managed Chroma tier we evaluated is billed at 47 dollars per month per
index. We run three indexes in production and one in staging.

## Embedding compute

Batch embedding runs cost roughly 12 dollars per million chunks. Last quarter
we embedded 8.5 million chunks across all corpora.
""",
    "grounding.pdf": """# Grounding

The system prompt instructs the model to answer only from retrieved excerpts
and to say plainly when the notes do not cover something, rather than filling
the gap from general knowledge. Every claim carries a citation back to a source
document and page, so an answer can be checked against what was actually
written.
""",
}


def load_fixtures() -> None:
    """Index FIXTURE_DOCS, replacing anything already in the collection."""
    from backend.embed_and_store import store_chunks
    from backend.ingest import chunk_document

    collection = get_collection()
    existing = collection.get(include=[])["ids"]
    if existing:
        collection.delete(ids=existing)

    total = 0
    for name, text in FIXTURE_DOCS.items():
        chunks = chunk_document([(1, text)])
        total += store_chunks(name, chunks)
    print(
        f"Indexed {total} fixture chunks across {len(FIXTURE_DOCS)} documents.\n"
        "NOTE: this REPLACED the collection - re-run `python -m backend.embed_and_store`\n"
        "to get your own notes back.\n"
    )


def run(k: int, verbose: bool) -> int:
    """Run every case, print the report, and return the count that passed."""
    passed = 0
    ranks: list[int] = []
    width = max(len(q) for q, _ in CASES)

    for question, expected in CASES:
        hits = retrieve(question, k=k)
        sources = [hit.source for hit in hits]
        rank = sources.index(expected) + 1 if expected in sources else None

        if rank:
            passed += 1
            ranks.append(rank)
            print(f"PASS  {question:<{width}}  {expected} @ rank {rank}")
        else:
            print(f"FAIL  {question:<{width}}  expected {expected}")
            print(f"      got: {', '.join(sources) or '(nothing retrieved)'}")

        if verbose and hits:
            for i, hit in enumerate(hits, start=1):
                mark = "*" if hit.source == expected else " "
                print(f"      {mark}{i}. {hit.citation}  (dist {hit.distance:.3f})")

    total = len(CASES)
    accuracy = passed / total * 100 if total else 0.0
    print()
    print(f"{'-' * 60}")
    print(f"Passed {passed}/{total}   accuracy {accuracy:.1f}%   (top-{k})")
    if ranks:
        # Where in the top-k the right document landed. A pass at rank 4 is a
        # near miss that a smaller k would turn into a failure.
        print(f"Mean rank of correct hit: {sum(ranks) / len(ranks):.2f}")
        print(f"Rank 1 (best possible):   {ranks.count(1)}/{passed} of passes")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true", help="index the demo corpus first")
    parser.add_argument("--k", type=int, default=TOP_K, help=f"top-k cutoff (default {TOP_K})")
    parser.add_argument("--verbose", action="store_true", help="show all retrieved hits")
    args = parser.parse_args()

    if args.fixtures:
        load_fixtures()

    if get_collection().count() == 0:
        print(
            "Nothing is indexed. Either run `python -m backend.embed_and_store` "
            "with your PDFs in backend/data/, or `python eval.py --fixtures` to "
            "evaluate against the built-in demo corpus.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    passed = run(args.k, args.verbose)
    # Non-zero exit when anything failed, so this can gate CI.
    raise SystemExit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
