# Vector search

## Embeddings

An embedding maps text to a fixed-length vector so that passages about similar
things land near each other. This project runs all-MiniLM-L6-v2 locally on CPU,
which produces 384-dimensional vectors and needs no API call.

## Distance

When vectors are normalised to unit length, Euclidean distance ranks results
identically to cosine similarity, so the choice of metric stops mattering. What
does matter is using the same model at index time and query time: embedding
documents with one model and questions with another produces a store that
returns confident nonsense.

## Approximate search

Exact nearest-neighbour search becomes too slow as a corpus grows. HNSW builds
a navigable small-world graph and trades a small amount of recall for a large
speedup. Chroma uses it underneath.
