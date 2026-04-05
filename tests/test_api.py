from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

import parakeet_api.main as main
from parakeet_api.transcription import TranscriptionResult


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.model_name = main.settings.accepted_models[0]

    def test_v1_transcriptions_accepts_multipart_upload(self) -> None:
        with (
            patch("parakeet_api.main._require_api_key"),
            patch(
                "parakeet_api.main.runtime.transcribe",
                return_value=TranscriptionResult(text="hello world", duration=1.2),
            ) as transcribe,
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={"model": self.model_name},
                files={"file": ("sample.wav", b"not-real-audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "hello world")
        transcribe.assert_called_once()

    def test_openai_prefixed_transcriptions_route_is_available(self) -> None:
        with (
            patch("parakeet_api.main._require_api_key"),
            patch(
                "parakeet_api.main.runtime.transcribe",
                return_value=TranscriptionResult(text="prefixed route", duration=0.5),
            ),
        ):
            response = self.client.post(
                "/openai/v1/audio/transcriptions",
                data={"model": self.model_name},
                files={"file": ("sample.wav", b"not-real-audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "prefixed route")


if __name__ == "__main__":
    unittest.main()
