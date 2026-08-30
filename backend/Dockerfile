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
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements before source: this layer stays cached while you edit code, so
# torch is not reinstalled on every build.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# Both are normally volumes, but create them so the app still starts if it is
# run without mounts -- embed_and_store lists DATA_DIR on startup paths.
RUN mkdir -p /app/backend/data /app/backend/chroma

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1: bound to loopback the port is unreachable from
# outside the container, which looks exactly like a crashed server.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
