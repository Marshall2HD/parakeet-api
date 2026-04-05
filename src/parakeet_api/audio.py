from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AudioProcessingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioChunk:
    path: Path
    offset_seconds: float
    duration_seconds: float


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown subprocess error"
        raise AudioProcessingError(stderr) from exc


def probe_audio_duration(path: Path, ffprobe_binary: str) -> float:
    result = _run(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioProcessingError(f"Could not parse audio duration for {path.name}") from exc


def convert_to_wav(source_path: Path, target_path: Path, ffmpeg_binary: str) -> None:
    _run(
        [
            ffmpeg_binary,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target_path),
        ]
    )


def split_audio(
    source_path: Path,
    ffmpeg_binary: str,
    duration_seconds: float,
    chunk_duration_seconds: int,
    output_dir: Path,
) -> list[AudioChunk]:
    if chunk_duration_seconds <= 0 or duration_seconds <= chunk_duration_seconds:
        return [AudioChunk(path=source_path, offset_seconds=0.0, duration_seconds=duration_seconds)]

    chunks: list[AudioChunk] = []
    total_chunks = int(math.ceil(duration_seconds / chunk_duration_seconds))
    for index in range(total_chunks):
        start = index * chunk_duration_seconds
        remaining = max(duration_seconds - start, 0.0)
        chunk_length = min(float(chunk_duration_seconds), remaining)
        chunk_path = output_dir / f"chunk-{index:04d}.wav"
        _run(
            [
                ffmpeg_binary,
                "-y",
                "-i",
                str(source_path),
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{chunk_length:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(chunk_path),
            ]
        )
        chunks.append(
            AudioChunk(
                path=chunk_path,
                offset_seconds=float(start),
                duration_seconds=float(chunk_length),
            )
        )
    return chunks
