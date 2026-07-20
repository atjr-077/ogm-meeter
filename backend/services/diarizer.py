from __future__ import annotations

import logging

from pyannote.audio import Pipeline

from config import settings

logger = logging.getLogger("meetingbot.diarizer")

_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("event=diarization_model_loading model=pyannote/speaker-diarization-3.1")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=settings.hf_token,
        )
        logger.info("event=diarization_model_loaded")
    return _pipeline


def diarize(wav_path: str) -> list[dict]:
    """Run full-meeting diarization. Returns [{speaker, start_ms, end_ms}, ...]."""
    pipeline = _get_pipeline()
    diarization = pipeline(wav_path)

    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": speaker,
                "start_ms": int(turn.start * 1000),
                "end_ms": int(turn.end * 1000),
            }
        )
    return segments
