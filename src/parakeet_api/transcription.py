from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TranscriptWord:
    word: str
    start: float
    end: float


@dataclass(slots=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    duration: float
    language: str | None = None
    words: list[TranscriptWord] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)
