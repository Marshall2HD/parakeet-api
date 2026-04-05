from __future__ import annotations

import json

from parakeet_api.config import SUPPORTED_LANGUAGE_CODES
from parakeet_api.transcription import TranscriptionResult, TranscriptSegment


def join_transcript_parts(parts: list[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return " ".join(cleaned).strip()


def coerce_language_name(language: str | None) -> str:
    if not language:
        return "unknown"
    language = language.strip().lower()
    return SUPPORTED_LANGUAGE_CODES.get(language, language)


def ensure_segments(result: TranscriptionResult) -> list[TranscriptSegment]:
    if result.segments:
        return result.segments
    if not result.text:
        return []
    return [TranscriptSegment(id=0, start=0.0, end=result.duration, text=result.text)]


def build_json_payload(result: TranscriptionResult, include_logprobs: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": result.text,
        "usage": {
            "type": "duration",
            "seconds": round(result.duration),
        },
    }
    if include_logprobs:
        payload["logprobs"] = []
    return payload


def build_verbose_payload(
    result: TranscriptionResult,
    include_words: bool,
    include_segments: bool,
    include_logprobs: bool,
    temperature: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "task": "transcribe",
        "language": coerce_language_name(result.language),
        "duration": result.duration,
        "text": result.text,
        "usage": {
            "type": "duration",
            "seconds": round(result.duration),
        },
    }
    if include_words:
        payload["words"] = [
            {
                "word": item.word,
                "start": item.start,
                "end": item.end,
            }
            for item in result.words
        ]
    if include_segments:
        payload["segments"] = [
            {
                "id": item.id,
                "seek": int(item.start * 1000),
                "start": item.start,
                "end": item.end,
                "text": item.text,
                "tokens": [],
                "temperature": temperature,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
            for item in ensure_segments(result)
        ]
    if include_logprobs:
        payload["logprobs"] = []
    return payload


def _subtitle_timestamp(seconds: float, separator: str) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def build_srt(result: TranscriptionResult) -> str:
    lines: list[str] = []
    for index, segment in enumerate(ensure_segments(result), start=1):
        lines.append(str(index))
        lines.append(
            f"{_subtitle_timestamp(segment.start, ',')} --> {_subtitle_timestamp(segment.end, ',')}"
        )
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in ensure_segments(result):
        lines.append(
            f"{_subtitle_timestamp(segment.start, '.')} --> {_subtitle_timestamp(segment.end, '.')}"
        )
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def iter_sse_events(result: TranscriptionResult, include_logprobs: bool):
    delta_payload: dict[str, object] = {
        "type": "transcript.text.delta",
        "delta": result.text,
    }
    done_payload: dict[str, object] = {
        "type": "transcript.text.done",
        "text": result.text,
    }
    if include_logprobs:
        delta_payload["logprobs"] = []
        done_payload["logprobs"] = []
    yield f"data: {json.dumps(delta_payload, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
