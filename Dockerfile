FROM nvidia/cuda:13.2.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PARAKEET_HOST=0.0.0.0 \
    PARAKEET_PORT=8000 \
    PARAKEET_MODEL_NAME=nvidia/parakeet-tdt-0.6b-v3 \
    PARAKEET_MODEL_ALIASES=whisper-1,gpt-4o-transcribe,gpt-4o-mini-transcribe,parakeet-tdt-0.6b-v3,nvidia/parakeet-tdt-0.6b-v3 \
    PARAKEET_UPLOAD_DIR=/tmp/parakeet-api/uploads \
    PARAKEET_CHUNK_DURATION_SECONDS=900 \
    PARAKEET_MAX_CONCURRENT_REQUESTS=1 \
    PARAKEET_ALLOW_CPU_FALLBACK=true \
    PARAKEET_USE_LOCAL_ATTENTION=true \
    PARAKEET_ATTENTION_MODEL=rel_pos_local_attn \
    PARAKEET_ATT_CONTEXT_SIZE=128,128 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libsndfile1

RUN useradd --create-home --shell /bin/bash --uid 1000 appuser && \
    mkdir -p /models /tmp/parakeet-api/uploads && \
    chown -R appuser:appuser /models /tmp/parakeet-api

WORKDIR /app
RUN chown appuser:appuser /app

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv

USER appuser

COPY --chown=appuser pyproject.toml uv.lock .python-version README.md ./

RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --no-dev --frozen --no-install-project

COPY --chown=appuser src ./src

RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --no-dev --frozen

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["sh", "-c", "uvicorn parakeet_api.main:app --host $PARAKEET_HOST --port $PARAKEET_PORT"]
