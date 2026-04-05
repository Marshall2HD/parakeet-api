from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from parakeet_api.audio import convert_to_wav, probe_audio_duration, split_audio
from parakeet_api.config import Settings
from parakeet_api.formatters import join_transcript_parts
from parakeet_api.transcription import TranscriptionResult, TranscriptSegment, TranscriptWord


class ModelRuntimeError(RuntimeError):
    pass


class ParakeetRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._load_lock = threading.Lock()
        self._inference_gate = threading.Semaphore(settings.max_concurrent_requests)
        self._model: Any | None = None
        self._torch: Any | None = None

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        source_path: Path,
        workspace: Path,
        *,
        request_language: str | None,
        want_words: bool,
        want_segments: bool,
    ) -> TranscriptionResult:
        model = self._get_model()
        normalized_path = workspace / "normalized.wav"
        convert_to_wav(source_path, normalized_path, self.settings.ffmpeg_binary)
        duration_seconds = probe_audio_duration(normalized_path, self.settings.ffprobe_binary)
        chunks = split_audio(
            normalized_path,
            self.settings.ffmpeg_binary,
            duration_seconds,
            self.settings.chunk_duration_seconds,
            workspace,
        )

        texts: list[str] = []
        words: list[TranscriptWord] = []
        segments: list[TranscriptSegment] = []

        for chunk in chunks:
            chunk_result = self._transcribe_chunk(
                model,
                chunk.path,
                want_words=want_words,
                want_segments=want_segments,
                chunk_duration=chunk.duration_seconds,
            )
            if chunk_result.text:
                texts.append(chunk_result.text)
            if want_words:
                for word in chunk_result.words:
                    words.append(
                        TranscriptWord(
                            word=word.word,
                            start=round(word.start + chunk.offset_seconds, 3),
                            end=round(word.end + chunk.offset_seconds, 3),
                        )
                    )
            if want_segments:
                for segment in chunk_result.segments:
                    segments.append(
                        TranscriptSegment(
                            id=len(segments),
                            start=round(segment.start + chunk.offset_seconds, 3),
                            end=round(segment.end + chunk.offset_seconds, 3),
                            text=segment.text,
                        )
                    )

        return TranscriptionResult(
            text=join_transcript_parts(texts),
            duration=round(duration_seconds, 3),
            language=request_language,
            words=words,
            segments=segments,
        )

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                import nemo.collections.asr as nemo_asr  # type: ignore
                import torch  # type: ignore
            except Exception as exc:
                raise ModelRuntimeError(
                    "Failed to import NeMo/PyTorch. Install project dependencies inside the container."
                ) from exc

            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.settings.model_name)
            self._configure_attention(model)
            if torch.cuda.is_available():
                model = model.to("cuda")
            elif not self.settings.allow_cpu_fallback:
                raise ModelRuntimeError(
                    "CUDA is not available and PARAKEET_ALLOW_CPU_FALLBACK is disabled."
                )

            model.eval()
            self._torch = torch
            self._model = model
            return model

    def _configure_attention(self, model) -> None:
        if not self.settings.use_local_attention:
            return
        try:
            model.change_attention_model(
                self_attention_model=self.settings.attention_model,
                att_context_size=list(self.settings.att_context_size),
            )
        except Exception as exc:
            raise ModelRuntimeError(
                "Failed to enable local attention for Parakeet via NeMo. "
                "Set PARAKEET_USE_LOCAL_ATTENTION=false to disable it."
            ) from exc

    def _transcribe_chunk(
        self,
        model,
        chunk_path: Path,
        *,
        want_words: bool,
        want_segments: bool,
        chunk_duration: float,
    ) -> TranscriptionResult:
        torch = self._torch
        if torch is None:
            raise ModelRuntimeError("PyTorch runtime is not initialized.")

        need_timestamps = want_words or want_segments
        with self._inference_gate:
            with torch.inference_mode():
                hypotheses = model.transcribe(
                    [str(chunk_path)],
                    timestamps=need_timestamps,
                    batch_size=1,
                )

        if not hypotheses:
            return TranscriptionResult(text="", duration=chunk_duration)

        hypothesis = hypotheses[0]
        text = self._extract_text(hypothesis)
        timestamp_dict = self._extract_timestamp_dict(hypothesis) if need_timestamps else {}

        words = self._extract_words(timestamp_dict, model) if want_words else []
        segments = self._extract_segments(timestamp_dict, model, text, chunk_duration) if want_segments else []

        return TranscriptionResult(
            text=text,
            duration=chunk_duration,
            words=words,
            segments=segments,
        )

    @staticmethod
    def _extract_text(hypothesis: Any) -> str:
        if isinstance(hypothesis, str):
            return hypothesis.strip()
        if isinstance(hypothesis, dict):
            return str(hypothesis.get("text", "")).strip()
        if hasattr(hypothesis, "text"):
            return str(hypothesis.text).strip()
        return str(hypothesis).strip()

    @staticmethod
    def _extract_timestamp_dict(hypothesis: Any) -> dict[str, list[dict[str, Any]]]:
        if isinstance(hypothesis, dict):
            candidate = hypothesis.get("timestamp") or {}
            return candidate if isinstance(candidate, dict) else {}
        candidate = getattr(hypothesis, "timestamp", {}) or {}
        return candidate if isinstance(candidate, dict) else {}

    def _extract_words(
        self,
        timestamp_dict: dict[str, list[dict[str, Any]]],
        model,
    ) -> list[TranscriptWord]:
        output: list[TranscriptWord] = []
        for entry in timestamp_dict.get("word", []):
            start, end = self._coerce_bounds(entry, model)
            value = str(entry.get("word") or entry.get("char") or "").strip()
            if value:
                output.append(TranscriptWord(word=value, start=round(start, 3), end=round(end, 3)))
        return output

    def _extract_segments(
        self,
        timestamp_dict: dict[str, list[dict[str, Any]]],
        model,
        text: str,
        chunk_duration: float,
    ) -> list[TranscriptSegment]:
        output: list[TranscriptSegment] = []
        for entry in timestamp_dict.get("segment", []):
            start, end = self._coerce_bounds(entry, model)
            value = str(entry.get("segment") or entry.get("text") or "").strip()
            if value:
                output.append(
                    TranscriptSegment(
                        id=len(output),
                        start=round(start, 3),
                        end=round(end, 3),
                        text=value,
                    )
                )
        if output:
            return output
        if not text:
            return []
        return [TranscriptSegment(id=0, start=0.0, end=round(chunk_duration, 3), text=text)]

    def _coerce_bounds(self, stamp: dict[str, Any], model) -> tuple[float, float]:
        if "start" in stamp and "end" in stamp:
            return float(stamp["start"]), float(stamp["end"])
        if "start_offset" in stamp and "end_offset" in stamp:
            stride = self._time_stride_seconds(model)
            return float(stamp["start_offset"]) * stride, float(stamp["end_offset"]) * stride
        return 0.0, 0.0

    @staticmethod
    def _time_stride_seconds(model) -> float:
        cfg = getattr(model, "cfg", None)
        preprocessor = getattr(cfg, "preprocessor", None)
        window_stride = getattr(preprocessor, "window_stride", None)
        if window_stride is None and hasattr(preprocessor, "get"):
            window_stride = preprocessor.get("window_stride")
        if window_stride is None:
            return 0.08
        return 8 * float(window_stride)
