from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parakeet_api.audio import AudioChunk
from parakeet_api.config import DEFAULT_MODEL_NAME, Settings
from parakeet_api.service import ParakeetRuntime
from parakeet_api.transcription import TranscriptionResult


def make_settings(*, chunk_duration_seconds: int = 900) -> Settings:
    return Settings(
        model_name=DEFAULT_MODEL_NAME,
        model_aliases=("whisper-1", DEFAULT_MODEL_NAME),
        host="0.0.0.0",
        port=8000,
        api_key=None,
        upload_dir="/tmp/parakeet-api/uploads",
        chunk_duration_seconds=chunk_duration_seconds,
        max_concurrent_requests=1,
        allow_cpu_fallback=True,
        use_local_attention=True,
        attention_model="rel_pos_local_attn",
        att_context_size=(128, 128),
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
    )


class ServiceTests(unittest.TestCase):
    def test_transcribe_retries_empty_long_chunk_with_smaller_subchunks(self) -> None:
        runtime = ParakeetRuntime(make_settings())
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            with (
                patch.object(runtime, "_get_model", return_value=object()),
                patch("parakeet_api.service.convert_to_wav"),
                patch("parakeet_api.service.probe_audio_duration", return_value=540.0),
                patch(
                    "parakeet_api.service.split_audio",
                    side_effect=[
                        [
                            AudioChunk(
                                path=workspace / "normalized.wav",
                                offset_seconds=0.0,
                                duration_seconds=540.0,
                            )
                        ],
                        [
                            AudioChunk(
                                path=workspace / "retry-0000000000" / "chunk-0000.wav",
                                offset_seconds=0.0,
                                duration_seconds=270.0,
                            ),
                            AudioChunk(
                                path=workspace / "retry-0000000000" / "chunk-0001.wav",
                                offset_seconds=270.0,
                                duration_seconds=270.0,
                            ),
                        ],
                    ],
                ) as split_audio_mock,
                patch.object(
                    runtime,
                    "_transcribe_chunk",
                    side_effect=[
                        TranscriptionResult(text="", duration=540.0),
                        TranscriptionResult(text="hello", duration=270.0),
                        TranscriptionResult(text="world", duration=270.0),
                    ],
                ),
            ):
                result = runtime.transcribe(
                    workspace / "clip.mp3",
                    workspace,
                    request_language="en",
                    want_words=False,
                    want_segments=False,
                )

        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.duration, 540.0)
        self.assertEqual(split_audio_mock.call_count, 2)
        self.assertEqual(split_audio_mock.call_args_list[1].args[3], 270)

    def test_transcribe_keeps_empty_short_chunk_without_retrying(self) -> None:
        runtime = ParakeetRuntime(make_settings())
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            with (
                patch.object(runtime, "_get_model", return_value=object()),
                patch("parakeet_api.service.convert_to_wav"),
                patch("parakeet_api.service.probe_audio_duration", return_value=420.0),
                patch(
                    "parakeet_api.service.split_audio",
                    return_value=[
                        AudioChunk(
                            path=workspace / "normalized.wav",
                            offset_seconds=0.0,
                            duration_seconds=420.0,
                        )
                    ],
                ) as split_audio_mock,
                patch.object(
                    runtime,
                    "_transcribe_chunk",
                    return_value=TranscriptionResult(text="", duration=420.0),
                ),
            ):
                result = runtime.transcribe(
                    workspace / "clip.mp3",
                    workspace,
                    request_language="en",
                    want_words=False,
                    want_segments=False,
                )

        self.assertEqual(result.text, "")
        self.assertEqual(split_audio_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
