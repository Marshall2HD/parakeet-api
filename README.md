# parakeet-api

OpenAI-compatible transcription API backed by [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

The service exposes `POST /v1/audio/transcriptions` so existing Whisper/OpenAI speech-to-text clients can be pointed at a local CUDA-backed container instead of OpenAI's hosted endpoint.

The dependency workflow uses [`uv`](https://docs.astral.sh/uv/), and the container installs official [`nemo_toolkit[asr]`](https://pypi.org/project/nemo-toolkit/) on top of CUDA with PyTorch CUDA 12.8 wheels.

## What It Supports

- `POST /v1/audio/transcriptions`
- `GET /v1/models`
- `GET /healthz`
- OpenAI-style request fields:
  - `file`
  - `model`
  - `language`
  - `response_format`
  - `stream`
  - `timestamp_granularities[]`
  - `include[]=logprobs`
  - `temperature` is accepted for compatibility
  - `prompt` is accepted by clients but ignored by Parakeet
- Response formats:
  - `json`
  - `text`
  - `verbose_json`
  - `srt`
  - `vtt`
- Minimal SSE mode for `stream=true`
- Long-form audio chunking with `ffmpeg` before inference
- Native NeMo local attention enabled by default with `[128,128]`
- Optional bearer auth via `PARAKEET_API_KEY`

## VAD And OpenAI Compatibility

OpenAI's transcription API does not expose VAD as a public request parameter. If we add VAD later, it should remain an internal server-side chunking optimization, not part of the external Whisper/OpenAI-compatible contract.

## Attention Defaults

The runtime enables native NeMo local attention by default:

```python
asr_model.change_attention_model(
    self_attention_model="rel_pos_local_attn",
    att_context_size=[128, 128],
)
```

That matches the long-form inference pattern documented on the official Parakeet v3 model card and the NeMo ASR docs, and matches your stated preference for the best speed/accuracy tradeoff.

## Known Gaps

- `diarized_json` is not implemented.
- `gpt-4o-transcribe-diarize` and speaker-reference fields are rejected.
- `include[]=logprobs` returns an empty array because Parakeet/NeMo does not expose OpenAI-style token logprobs.
- `language` is treated as a compatibility hint and validation check. Parakeet still auto-detects internally.
- `/v1/audio/translations` is intentionally unsupported because this model transcribes in the source language.

## Docker

Build:

```bash
docker build -t parakeet-api .
```

Run with NVIDIA Container Toolkit:

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -e PARAKEET_API_KEY=local-dev-key \
  -v parakeet-model-cache:/models \
  parakeet-api
```

Or with Compose:

```bash
docker compose up --build
```

## Local Development With uv

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run uvicorn parakeet_api.main:app --reload
```

## Environment

- `PARAKEET_MODEL_NAME`
  - Default: `nvidia/parakeet-tdt-0.6b-v3`
- `PARAKEET_MODEL_ALIASES`
  - Default: `whisper-1,gpt-4o-transcribe,gpt-4o-mini-transcribe,parakeet-tdt-0.6b-v3,nvidia/parakeet-tdt-0.6b-v3`
- `PARAKEET_API_KEY`
  - Optional bearer token required for all endpoints except `/healthz`
- `PARAKEET_PORT`
  - Default: `8000`
- `PARAKEET_CHUNK_DURATION_SECONDS`
  - Default: `900`
  - Local attention stays enabled regardless. This chunking limit is still a server-side guardrail around NeMo's `transcribe()` path, which is more reliable on shorter fragments than on single very long files.
- `PARAKEET_MAX_CONCURRENT_REQUESTS`
  - Default: `1`
- `PARAKEET_ALLOW_CPU_FALLBACK`
  - Default: `true`
- `PARAKEET_USE_LOCAL_ATTENTION`
  - Default: `true`
- `PARAKEET_ATTENTION_MODEL`
  - Default: `rel_pos_local_attn`
- `PARAKEET_ATT_CONTEXT_SIZE`
  - Default: `128,128`
- `PARAKEET_UPLOAD_DIR`
  - Default: `/tmp/parakeet-api/uploads`

If you want to rely more heavily on NeMo local attention instead of server-side chunking for long files, set `PARAKEET_CHUNK_DURATION_SECONDS=0`.

## Example: curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local-dev-key" \
  -F file=@sample.wav \
  -F model=whisper-1 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word"
```

## Example: OpenAI SDK

Point the client at this API instead of OpenAI:

```python
from openai import OpenAI

client = OpenAI(
    api_key="local-dev-key",
    base_url="http://localhost:8000/v1",
)

with open("sample.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
    )

print(transcript)
```

## Notes On The Reference Repo

This project is intentionally smaller than [`tulas75/parakeet-api`](https://github.com/tulas75/parakeet-api). The older repo has accumulated a large amount of runtime tuning and heuristics in one Flask file. This version keeps the compatibility surface focused on the current OpenAI transcription endpoint shape, isolates the model runtime, and uses a CUDA-first FastAPI container that is easier to reason about and extend.

## Sources Used For The Compatibility Shape

- OpenAI transcription reference:
  - https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create/
- NVIDIA Parakeet model card:
  - https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- NVIDIA NeMo ASR docs:
  - https://docs.nvidia.com/nemo-framework/user-guide/25.04/nemotoolkit/asr/intro.html
- NVIDIA NeMo software component versions:
  - https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html
- uv docs:
  - https://docs.astral.sh/uv/
  - https://docs.astral.sh/uv/guides/integration/pytorch/
- Package releases:
  - https://pypi.org/project/fastapi/
  - https://pypi.org/project/uvicorn/
  - https://pypi.org/project/python-multipart/
  - https://pypi.org/project/nemo-toolkit/
