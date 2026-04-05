from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_MODEL_ALIASES = (
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "parakeet-tdt-0.6b-v3",
    DEFAULT_MODEL_NAME,
)

SUPPORTED_LANGUAGE_CODES = {
    "bg": "bulgarian",
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "hr": "croatian",
    "hu": "hungarian",
    "it": "italian",
    "lt": "lithuanian",
    "lv": "latvian",
    "mt": "maltese",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sk": "slovak",
    "sl": "slovenian",
    "sv": "swedish",
    "uk": "ukrainian",
}

LANGUAGE_NAME_TO_CODE = {name: code for code, name in SUPPORTED_LANGUAGE_CODES.items()}


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_model_aliases(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_MODEL_ALIASES
    aliases = tuple(part.strip() for part in value.split(",") if part.strip())
    return aliases or DEFAULT_MODEL_ALIASES


def _parse_int_pair(value: str | None, default: tuple[int, int]) -> tuple[int, int]:
    if not value:
        return default
    normalized = value.strip().strip("[]()")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated integers, got: {value!r}")
    return int(parts[0]), int(parts[1])


@dataclass(frozen=True)
class Settings:
    model_name: str
    model_aliases: tuple[str, ...]
    host: str
    port: int
    api_key: str | None
    upload_dir: str
    chunk_duration_seconds: int
    max_concurrent_requests: int
    allow_cpu_fallback: bool
    use_local_attention: bool
    attention_model: str
    att_context_size: tuple[int, int]
    ffmpeg_binary: str
    ffprobe_binary: str

    @property
    def accepted_models(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for item in (*self.model_aliases, self.model_name):
            seen[item] = None
        return tuple(seen)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        model_name=os.getenv("PARAKEET_MODEL_NAME", DEFAULT_MODEL_NAME),
        model_aliases=_parse_model_aliases(os.getenv("PARAKEET_MODEL_ALIASES")),
        host=os.getenv("PARAKEET_HOST", "0.0.0.0"),
        port=int(os.getenv("PARAKEET_PORT", "8000")),
        api_key=os.getenv("PARAKEET_API_KEY") or None,
        upload_dir=os.getenv("PARAKEET_UPLOAD_DIR", "/tmp/parakeet-api/uploads"),
        chunk_duration_seconds=max(int(os.getenv("PARAKEET_CHUNK_DURATION_SECONDS", "900")), 0),
        max_concurrent_requests=max(int(os.getenv("PARAKEET_MAX_CONCURRENT_REQUESTS", "1")), 1),
        allow_cpu_fallback=_parse_bool(os.getenv("PARAKEET_ALLOW_CPU_FALLBACK"), True),
        use_local_attention=_parse_bool(os.getenv("PARAKEET_USE_LOCAL_ATTENTION"), True),
        attention_model=os.getenv("PARAKEET_ATTENTION_MODEL", "rel_pos_local_attn"),
        att_context_size=_parse_int_pair(
            os.getenv("PARAKEET_ATT_CONTEXT_SIZE"),
            (128, 128),
        ),
        ffmpeg_binary=os.getenv("PARAKEET_FFMPEG_BINARY", "ffmpeg"),
        ffprobe_binary=os.getenv("PARAKEET_FFPROBE_BINARY", "ffprobe"),
    )
