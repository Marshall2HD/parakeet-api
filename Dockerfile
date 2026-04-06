FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
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

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock .python-version README.md ./

RUN uv sync --no-dev --frozen --no-install-project

COPY src ./src

RUN uv sync --no-dev --frozen

RUN useradd --create-home --uid 1000 appuser && \
    mkdir -p /models /tmp/parakeet-api/uploads && \
    chown -R appuser:appuser /app /models /tmp/parakeet-api

USER appuser
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "parakeet_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
