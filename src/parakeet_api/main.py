from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile

from parakeet_api.audio import AudioProcessingError
from parakeet_api.config import (
    LANGUAGE_NAME_TO_CODE,
    SUPPORTED_LANGUAGE_CODES,
    get_settings,
)
from parakeet_api.formatters import (
    build_json_payload,
    build_srt,
    build_verbose_payload,
    build_vtt,
    iter_sse_events,
)
from parakeet_api.service import ModelRuntimeError, ParakeetRuntime

LOGGER = logging.getLogger("parakeet_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Parakeet API", version="0.1.0")
settings = get_settings()
runtime = ParakeetRuntime(settings)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


class OpenAIHTTPException(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        error_type: str = "invalid_request_error",
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": param,
                    "code": code,
                }
            },
        )


@app.exception_handler(OpenAIHTTPException)
async def openai_http_exception_handler(_request: Request, exc: OpenAIHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": None,
                "code": "validation_error",
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled request error")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "param": None,
                "code": None,
            }
        },
    )


def _require_api_key(request: Request) -> None:
    if not settings.api_key:
        return
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise OpenAIHTTPException(
            401,
            "Missing bearer token.",
            code="invalid_api_key",
        )
    provided = header.removeprefix("Bearer ").strip()
    if provided != settings.api_key:
        raise OpenAIHTTPException(
            401,
            "Invalid API key.",
            code="invalid_api_key",
        )


def _collect_list(form: Any, *names: str) -> list[str]:
    items: list[str] = []
    for name in names:
        for value in form.getlist(name):
            if value is None:
                continue
            text = str(value).strip()
            if text:
                items.append(text)
    return items


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_language_hint(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in SUPPORTED_LANGUAGE_CODES:
        return normalized
    if normalized in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[normalized]
    raise OpenAIHTTPException(
        400,
        "Unsupported language hint for parakeet-tdt-0.6b-v3.",
        param="language",
        code="unsupported_value",
    )


def _parse_temperature(raw_value: str | None) -> float:
    if raw_value in (None, ""):
        return 0.0
    try:
        return float(raw_value)
    except ValueError as exc:
        raise OpenAIHTTPException(
            400,
            "temperature must be a number.",
            param="temperature",
            code="invalid_value",
        ) from exc


def _persist_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(upload.file, output)


def _validate_model(model_name: str | None) -> str:
    if not model_name:
        raise OpenAIHTTPException(
            400,
            "model is required.",
            param="model",
            code="missing",
        )
    if model_name not in settings.accepted_models:
        raise OpenAIHTTPException(
            400,
            f"Unsupported model '{model_name}'. Accepted values: {', '.join(settings.accepted_models)}",
            param="model",
            code="model_not_found",
        )
    return model_name


def _build_transcription_response(
    *,
    result,
    response_format: str,
    stream: bool,
    include_logprobs: bool,
    include_words: bool,
    include_segments: bool,
    temperature: float,
) -> Response:
    if stream:
        return StreamingResponse(
            iter_sse_events(result, include_logprobs),
            media_type="text/event-stream",
        )
    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "json":
        return JSONResponse(build_json_payload(result, include_logprobs))
    if response_format == "verbose_json":
        return JSONResponse(
            build_verbose_payload(
                result,
                include_words=include_words,
                include_segments=include_segments,
                include_logprobs=include_logprobs,
                temperature=temperature,
            )
        )
    if response_format == "srt":
        return Response(content=build_srt(result), media_type="application/x-subrip")
    if response_format == "vtt":
        return Response(content=build_vtt(result), media_type="text/vtt")
    raise OpenAIHTTPException(
        400,
        f"Unsupported response_format '{response_format}'.",
        param="response_format",
        code="unsupported_value",
    )


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "model": settings.model_name,
        "model_loaded": runtime.is_model_loaded(),
    }


@app.get("/openai/v1/models")
@app.get("/v1/models")
def list_models(request: Request) -> dict[str, object]:
    _require_api_key(request)
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "nvidia",
            }
            for model_name in settings.accepted_models
        ],
    }


@app.post("/audio/transcriptions")
@app.post("/openai/v1/audio/transcriptions")
@app.post("/v1/audio/transcriptions")
async def create_transcription(request: Request):
    _require_api_key(request)
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise OpenAIHTTPException(
            400,
            "file is required.",
            param="file",
            code="missing",
        )

    model_name = _validate_model(form.get("model"))
    if model_name.endswith("diarize"):
        raise OpenAIHTTPException(
            400,
            "Diarization is not supported by this Parakeet-backed implementation.",
            param="model",
            code="unsupported_value",
        )

    response_format = (form.get("response_format") or "json").strip().lower()
    if response_format == "diarized_json":
        raise OpenAIHTTPException(
            400,
            "response_format=diarized_json is not supported by this service.",
            param="response_format",
            code="unsupported_value",
        )

    includes = _collect_list(form, "include[]", "include")
    timestamp_granularities = _collect_list(
        form,
        "timestamp_granularities[]",
        "timestamp_granularities",
    )
    stream = _parse_bool(form.get("stream"), default=False)
    temperature = _parse_temperature(form.get("temperature"))
    request_language = _normalize_language_hint(form.get("language"))

    for unsupported in (
        "chunking_strategy",
        "known_speaker_names[]",
        "known_speaker_names",
        "known_speaker_references[]",
        "known_speaker_references",
    ):
        if form.get(unsupported) is not None:
            raise OpenAIHTTPException(
                400,
                f"{unsupported} is not supported by this service.",
                param=unsupported.removesuffix("[]"),
                code="unsupported_value",
            )

    invalid_granularities = [item for item in timestamp_granularities if item not in {"word", "segment"}]
    if invalid_granularities:
        raise OpenAIHTTPException(
            400,
            "timestamp_granularities only supports 'word' and 'segment'.",
            param="timestamp_granularities",
            code="unsupported_value",
        )

    include_logprobs = "logprobs" in includes
    include_words = "word" in timestamp_granularities
    include_segments = response_format in {"srt", "vtt"} or (
        response_format == "verbose_json"
        and (not timestamp_granularities or "segment" in timestamp_granularities)
    )

    suffix = Path(upload.filename or "audio").suffix or ".bin"
    with tempfile.TemporaryDirectory(dir=settings.upload_dir) as tmpdir:
        workspace = Path(tmpdir)
        upload_path = workspace / f"upload{suffix}"
        await run_in_threadpool(_persist_upload, upload, upload_path)
        try:
            result = await run_in_threadpool(
                runtime.transcribe,
                upload_path,
                workspace,
                request_language=request_language,
                want_words=include_words,
                want_segments=include_segments,
            )
        except AudioProcessingError as exc:
            raise OpenAIHTTPException(
                400,
                f"Could not process uploaded audio: {exc}",
                param="file",
                code="invalid_file",
            ) from exc
        except ModelRuntimeError as exc:
            raise OpenAIHTTPException(500, str(exc), code="runtime_error", error_type="server_error") from exc

    return _build_transcription_response(
        result=result,
        response_format=response_format,
        stream=stream,
        include_logprobs=include_logprobs,
        include_words=include_words,
        include_segments=include_segments,
        temperature=temperature,
    )


@app.post("/audio/translations")
@app.post("/openai/v1/audio/translations")
@app.post("/v1/audio/translations")
def create_translation(request: Request):
    _require_api_key(request)
    raise OpenAIHTTPException(
        400,
        "Translation is not supported by parakeet-tdt-0.6b-v3. Use /v1/audio/transcriptions instead.",
        code="unsupported_value",
    )
