# Build context is the PROJECT ROOT (see docker-compose.yml), because the app
# is imported as the package `backend.main:app`.
FROM python:3.13-slim

# curl is only here for the compose healthcheck. No build toolchain is needed:
# every dependency ships manylinux wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the sentence-transformers cache on a known path so the model baked
    # in below is the one found at runtime.
    HF_HOME=/opt/huggingface

WORKDIR /app

# Requirements before source: this layer stays cached while you edit code, so
# torch is not reinstalled on every build.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# To shrink the image substantially on x86, install CPU-only torch first --
# the default PyPI wheel pulls ~2 GB of CUDA libraries this app never uses:
#   RUN pip install --no-cache-dir torch \
#       --index-url https://download.pytorch.org/whl/cpu

# Bake the embedding model into the image (~90 MB). Without this the first
# question after every fresh container stalls on a download, and the container
# needs egress to huggingface.co to work at all.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY backend/ ./backend/

# Both are normally volumes, but create them so the app still starts if it is
# run without mounts -- embed_and_store lists DATA_DIR on startup paths.
RUN mkdir -p /app/backend/data /app/backend/chroma

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1: bound to loopback the port is unreachable from
# outside the container, which looks exactly like a crashed server.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
