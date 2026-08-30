# Deploying the demo

Single image: builds the UI, bakes a read-only demo index, serves both from one
FastAPI process. Same-origin, so no CORS and only one service to host.

## Why this shape

Three free-tier constraints drove it:

| Constraint | Resolution |
| --- | --- |
| ~448 MB RAM with the embedding model loaded | Host with >= 1 GB; 512 MB tiers OOM |
| No persistent disk on free tiers | Bake the index into the image; a demo corpus is fixed, so it can be read-only |
| Two services would need CORS and two hosts | Serve the UI from FastAPI - one service, same-origin |

## Hugging Face Spaces (recommended, free)

Free CPU Spaces get 16 GB RAM and 2 vCPU, which fits comfortably, and Docker
Spaces give full control. No card required.

1. Create a Space: type **Docker**, template **Blank**, visibility **Public**.
2. In *Settings -> Variables and secrets*, add secret `GEMINI_API_KEY`.
3. Push this repo to the Space remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main
   ```

Spaces builds the root `Dockerfile` and expects port 7860, which is the default
here. The build bakes the index and fails loudly if the corpus produces no
chunks.

## Anywhere else

Any host that runs a container with >= 1 GB RAM works - Fly.io, Railway, Render
(paid tiers), Cloud Run. Set `GEMINI_API_KEY`, and `PORT` if the platform
injects its own.

```bash
docker build -t ask-my-notes .
docker run -p 7860:7860 -e GEMINI_API_KEY=... ask-my-notes
```

## Demo quota

`RATE_LIMIT` (default 3) caps questions per visitor per `RATE_WINDOW` (default
24h). This rations the free-tier quota; it does not create any. With 20 model
requests/day and 2-3 per agentic question, the demo serves roughly 7 questions
daily before returning a clear "quota exhausted" message. At the expected
traffic of ~2 visitors/day, 3 questions each costs 12-18 requests, which fits
inside the free tier. Raise the limit only alongside billing.

## Swapping the demo corpus

`demo-notes/` is baked in at build time. Replace it with your own notes and
rebuild. Do not add third-party copyrighted material - a public demo
republishes whatever is in there.
