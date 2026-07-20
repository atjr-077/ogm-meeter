from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from openai import OpenAI

from config import settings


_UPLOAD_TARGET_BYTES = 20 * 1024 * 1024  # stay safely under the ~25MB API cap
_MIN_BITRATE_KBPS = 16
_MAX_BITRATE_KBPS = 64


def _get_duration_seconds(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _compress_for_upload(wav_path: str) -> str:
    """Whisper-based transcription APIs cap request size (~25MB); re-encode the
    lossless WAV master to a small mono mp3 before upload. Bitrate scales down
    for longer recordings so long meetings (multi-hour) still fit the cap.
    ponytail: floor of 16kbps means recordings past ~3.5h could still exceed
    the cap; switch to chunked transcription if that becomes a real need."""
    src = Path(wav_path)
    duration = _get_duration_seconds(str(src))
    bitrate_kbps = _MAX_BITRATE_KBPS
    if duration > 0:
        bitrate_kbps = int((_UPLOAD_TARGET_BYTES * 8) / duration / 1000)
        bitrate_kbps = max(_MIN_BITRATE_KBPS, min(_MAX_BITRATE_KBPS, bitrate_kbps))

    mp3_path = src.with_name(src.stem + ".upload.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-b:a", f"{bitrate_kbps}k", str(mp3_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return str(mp3_path)


class Transcriber(ABC):
    """Provider interface: swap OpenAI for Groq/Gemini/etc without touching business logic."""

    @abstractmethod
    def transcribe(self, wav_path: str) -> list[dict]:
        """Return timestamped segments: [{text, start_ms, end_ms}, ...]."""


class OpenAITranscriber(Transcriber):
    model = "gpt-4o-transcribe"

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)

    def transcribe(self, wav_path: str) -> list[dict]:
        upload_path = _compress_for_upload(wav_path)
        kwargs = {"language": settings.transcribe_language} if settings.transcribe_language else {}
        with open(upload_path, "rb") as f:
            resp = self._client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="verbose_json",
                **kwargs,
            )

        raw_segments = getattr(resp, "segments", None) or []
        segments: list[dict] = []
        for seg in raw_segments:
            seg_text = seg.text if hasattr(seg, "text") else seg["text"]
            seg_start = seg.start if hasattr(seg, "start") else seg["start"]
            seg_end = seg.end if hasattr(seg, "end") else seg["end"]
            text = seg_text.strip()
            if not text:
                continue
            segments.append(
                {
                    "text": text,
                    "start_ms": int(seg_start * 1000),
                    "end_ms": int(seg_end * 1000),
                }
            )

        if not segments:
            fallback_text = getattr(resp, "text", "") or ""
            if fallback_text.strip():
                segments = [{"text": fallback_text.strip(), "start_ms": 0, "end_ms": 0}]

        return segments


class GroqTranscriber(OpenAITranscriber):
    """Groq exposes an OpenAI-compatible Whisper endpoint — reuse the same parsing."""

    model = "whisper-large-v3"

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")


_PROVIDERS: dict[str, type[Transcriber]] = {
    "openai": OpenAITranscriber,
    "groq": GroqTranscriber,
}


def get_transcriber() -> Transcriber:
    provider = _PROVIDERS.get(settings.transcriber_provider)
    if provider is None:
        raise ValueError(f"unknown transcriber provider: {settings.transcriber_provider}")
    return provider()
