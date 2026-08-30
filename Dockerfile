# Single-image deployment: builds the UI, bakes a read-only demo index, and
# serves both from one FastAPI process. Same-origin, so no CORS and only one
# service to host -- which is what makes a free tier workable.
#
#   docker build -t ask-my-notes .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... ask-my-notes

# ---------- stage 1: build the UI ----------
FROM node:22-alpine AS ui

WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Empty base = same-origin relative requests ("/chat"), so the bundle does not
# hardcode any hostname and works under whatever URL it is deployed to.
ENV VITE_API_BASE=""
COPY frontend/ ./
RUN npm run build

# ---------- stage 2: runtime ----------
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model so the first request does not stall on a download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY backend/ ./backend/
COPY --from=ui /ui/dist ./backend/static

# Build the vector index INTO the image. A demo has a fixed corpus, so a
# read-only prebuilt index removes the need for a persistent disk entirely --
# the single hardest requirement to satisfy on a free tier.
COPY demo-notes/ ./backend/data/
RUN python -m backend.embed_and_store && python -c "\
from backend.retrieval import get_collection; \
n = get_collection().count(); \
print(f'baked index: {n} chunks'); \
assert n > 0, 'demo corpus produced no chunks - build aborted'"

# Hugging Face Spaces runs containers as UID 1000, not root. Chroma opens its
# SQLite store read-write even for queries (WAL/SHM files), so the baked index
# and its directory must be owned by that user or startup fails with a
# read-only database error.
RUN useradd -m -u 1000 appuser \
    && chown -R 1000:1000 /app /opt/huggingface
USER 1000

# Hugging Face Spaces expects 7860; override with PORT elsewhere.
ENV PORT=7860 \
    RATE_LIMIT=3 \
    RATE_WINDOW=86400
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
