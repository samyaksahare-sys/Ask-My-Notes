# Deploying the demo

Single image: builds the UI, ships a prebuilt demo index, serves both from one
FastAPI process. Same-origin, so no CORS and only one service to host.

## Why this shape

| Constraint | Resolution |
| --- | --- |
| Free tiers cap at 512 MB RAM | Embeddings come from the Gemini API, not a local model: **138 MB** steady state |
| No persistent disk on free tiers | The demo index is committed and copied in - read-only at runtime |
| Two services would need CORS and two hosts | FastAPI serves the built UI - one service, same-origin |
| Build needs no secrets | The index is prebuilt, so the image builds with no API key and no network |

Embedding locally with `sentence-transformers` previously cost 448 MB of RAM
and ~1.3 GB of dependencies, which fits no free tier and would not build on a
laptop with 11 GB free.

## Hosting

Any host that runs a container with >= 512 MB RAM. Both of these are free and
need no card:

- **Render** - free web service, 512 MB. Cold start 30-50s after 15 min idle.
- **Koyeb** - free instance, 512 MB. Scale-to-zero cannot be disabled.

Set `GEMINI_API_KEY` as an environment variable in the host's dashboard, and
`PORT` if the platform injects its own (both default sensibly).

Hugging Face Spaces is **not** an option on the free plan: Docker Spaces now
require PRO. Only Static Spaces are free.

## Local

```bash
docker build -t ask-my-notes .
docker run -p 7860:7860 -e GEMINI_API_KEY=... ask-my-notes
```

Needs ~2 GB of free disk for the image and build cache.

## Demo quota

`RATE_LIMIT` (default 3) caps questions per visitor per `RATE_WINDOW` (default
24h). This rations the free-tier quota; it does not create any. At ~2
visitors/day, 3 questions each costs 12-18 model requests, inside the free
tier. Note that **embedding also consumes quota**, so re-indexing a large
corpus is not free.

## Swapping the demo corpus

`demo-notes/` holds the source, `demo-index/` the prebuilt vectors. After
editing the notes, regenerate the index and commit both:

```bash
DATA_DIR=$PWD/demo-notes CHROMA_DIR=$PWD/demo-index \
  python -m backend.embed_and_store --reset
```

Do not add third-party copyrighted material - a public demo republishes
whatever is in there.
