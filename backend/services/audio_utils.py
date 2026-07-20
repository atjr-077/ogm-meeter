from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

from config import settings

logger = logging.getLogger("meetingbot.audio")


def meeting_dir(session_id: str) -> Path:
    return Path(settings.data_dir) / "meetings" / session_id


def chunks_dir(session_id: str) -> Path:
    d = meeting_dir(session_id) / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_session_dirs(session_id: str) -> None:
    chunks_dir(session_id)


def _ext_from_filename(filename: str | None) -> str:
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext.isalnum() and len(ext) <= 8:
            return ext
    return "webm"


def save_chunk(session_id: str, ts: int, filename: str | None, content: bytes) -> Path:
    ext = _ext_from_filename(filename)
    path = chunks_dir(session_id) / f"{ts:020d}_{uuid.uuid4().hex[:8]}.{ext}"
    path.write_bytes(content)
    logger.info(f"session_id={session_id} event=chunk_saved bytes={len(content)} path={path}")
    return path


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (args={args}): {result.stderr[-2000:]}")


def concatenate_and_normalize(session_id: str) -> Path:
    """Convert every chunk to 16kHz mono wav, then concatenate into one file."""
    chunks = sorted(chunks_dir(session_id).glob("*"))
    if not chunks:
        raise RuntimeError(f"no audio chunks found for session {session_id}")

    work_dir = meeting_dir(session_id) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized: list[Path] = []
    skipped = 0
    for i, chunk in enumerate(chunks):
        out = work_dir / f"norm_{i:05d}.wav"
        try:
            run_ffmpeg(["-i", str(chunk), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out)])
            normalized.append(out)
        except RuntimeError:
            # A single truncated/empty chunk (network blip, tab backgrounded mid-upload)
            # shouldn't sink an entire long recording — skip it and keep going.
            skipped += 1
            logger.warning(f"session_id={session_id} event=chunk_skipped path={chunk}")

    if not normalized:
        raise RuntimeError(f"all {len(chunks)} audio chunks failed to decode for session {session_id}")
    if skipped:
        logger.warning(f"session_id={session_id} event=chunks_skipped_total count={skipped}")

    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in normalized), encoding="utf-8"
    )

    full_wav = meeting_dir(session_id) / "full_16k_mono.wav"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(full_wav)])

    logger.info(
        f"session_id={session_id} event=audio_normalized chunks={len(chunks)} output={full_wav}"
    )
    return full_wav
