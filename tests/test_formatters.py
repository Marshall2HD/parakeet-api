from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parakeet_api.formatters import build_srt, build_verbose_payload, build_vtt, join_transcript_parts
from parakeet_api.transcription import TranscriptionResult, TranscriptSegment, TranscriptWord


class FormatterTests(unittest.TestCase):
    def test_join_transcript_parts(self) -> None:
        self.assertEqual(join_transcript_parts([" hello ", "", "world "]), "hello world")

    def test_build_srt(self) -> None:
        result = TranscriptionResult(
            text="hello world",
            duration=2.5,
            segments=[TranscriptSegment(id=0, start=0.0, end=2.5, text="hello world")],
        )
        payload = build_srt(result)
        self.assertIn("00:00:00,000 --> 00:00:02,500", payload)
        self.assertIn("hello world", payload)

    def test_build_vtt(self) -> None:
        result = TranscriptionResult(
            text="hello world",
            duration=2.5,
            segments=[TranscriptSegment(id=0, start=0.0, end=2.5, text="hello world")],
        )
        payload = build_vtt(result)
        self.assertTrue(payload.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:02.500", payload)

    def test_build_verbose_payload(self) -> None:
        result = TranscriptionResult(
            text="hello world",
            duration=2.5,
            language="en",
            words=[TranscriptWord(word="hello", start=0.0, end=0.9)],
            segments=[TranscriptSegment(id=0, start=0.0, end=2.5, text="hello world")],
        )
        payload = build_verbose_payload(
            result,
            include_words=True,
            include_segments=True,
            include_logprobs=True,
            temperature=0.0,
        )
        self.assertEqual(payload["language"], "english")
        self.assertEqual(payload["words"][0]["word"], "hello")
        self.assertEqual(payload["logprobs"], [])


if __name__ == "__main__":
    unittest.main()
