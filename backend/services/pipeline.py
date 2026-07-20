from __future__ import annotations

import asyncio
import logging
import uuid

from database import AsyncSessionLocal
from models import Meeting, MeetingSummary, TranscriptEmbedding, TranscriptSegment
from services import audio_utils, diarizer, embeddings
from services.summarizer import get_summarizer
from services.transcriber import get_transcriber

logger = logging.getLogger("meetingbot.pipeline")


def assign_speakers(transcript_segments: list[dict], diarization_segments: list[dict]) -> list[dict]:
    """Label each transcript segment with the diarization speaker it overlaps most."""
    labeled: list[dict] = []
    for seg in transcript_segments:
        best_speaker: str | None = None
        best_overlap = 0
        for d in diarization_segments:
            overlap = min(seg["end_ms"], d["end_ms"]) - max(seg["start_ms"], d["start_ms"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, d["speaker"]

        if best_speaker is None and diarization_segments:
            mid = (seg["start_ms"] + seg["end_ms"]) / 2
            best_speaker = min(
                diarization_segments,
                key=lambda d: abs((d["start_ms"] + d["end_ms"]) / 2 - mid),
            )["speaker"]

        labeled.append({**seg, "speaker_label": best_speaker or "SPEAKER_00"})
    return labeled


async def process_meeting(session_id: str) -> None:
    meeting_id = uuid.UUID(session_id)
    try:
        logger.info(f"meeting_id={session_id} event=pipeline_start")

        full_wav = await asyncio.to_thread(audio_utils.concatenate_and_normalize, session_id)

        try:
            diarization_segments = await asyncio.to_thread(diarizer.diarize, str(full_wav))
            logger.info(
                f"meeting_id={session_id} event=diarization_done segments={len(diarization_segments)}"
            )
        except Exception:
            logger.exception(f"meeting_id={session_id} event=diarization_failed_falling_back")
            diarization_segments = []

        transcriber = get_transcriber()
        transcript_segments = await asyncio.to_thread(transcriber.transcribe, str(full_wav))
        logger.info(
            f"meeting_id={session_id} event=transcription_done segments={len(transcript_segments)}"
        )

        labeled_segments = assign_speakers(transcript_segments, diarization_segments)

        async with AsyncSessionLocal() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting is None:
                raise RuntimeError(f"meeting {meeting_id} not found")

            meeting.audio_path = str(full_wav)

            for seg in labeled_segments:
                db.add(
                    TranscriptSegment(
                        meeting_id=meeting_id,
                        speaker_label=seg["speaker_label"],
                        text=seg["text"],
                        started_ms=seg["start_ms"],
                        ended_ms=seg["end_ms"],
                    )
                )
            await db.commit()

            transcript_text = "\n".join(f"{s['speaker_label']}: {s['text']}" for s in labeled_segments)

            summarizer = get_summarizer()
            summary_result, raw_response = await asyncio.to_thread(summarizer.summarize, transcript_text)

            db.add(
                MeetingSummary(
                    meeting_id=meeting_id,
                    summary=summary_result.summary,
                    key_topics=summary_result.key_topics,
                    decisions=summary_result.decisions,
                    action_items=[item.model_dump() for item in summary_result.action_items],
                    participants=summary_result.participants,
                    raw_llm_response=raw_response,
                )
            )

            chunks = embeddings.chunk_text(transcript_text) if transcript_text.strip() else []
            if chunks:
                vectors = await asyncio.to_thread(embeddings.embed_texts, chunks)
                for chunk, vector in zip(chunks, vectors):
                    db.add(
                        TranscriptEmbedding(meeting_id=meeting_id, chunk_text=chunk, embedding=vector)
                    )

            meeting.status = "done"
            await db.commit()

        logger.info(f"meeting_id={session_id} event=pipeline_done")

    except Exception:
        logger.exception(f"meeting_id={session_id} event=pipeline_failed")
        async with AsyncSessionLocal() as db:
            meeting = await db.get(Meeting, meeting_id)
            if meeting is not None:
                meeting.status = "error"
                await db.commit()


def _demo() -> None:
    transcript_segments = [
        {"text": "hello everyone", "start_ms": 0, "end_ms": 1000},
        {"text": "let's start", "start_ms": 1200, "end_ms": 2000},
        {"text": "no diarization overlap here", "start_ms": 9000, "end_ms": 9500},
    ]
    diarization_segments = [
        {"speaker": "SPEAKER_00", "start_ms": 0, "end_ms": 1500},
        {"speaker": "SPEAKER_01", "start_ms": 1500, "end_ms": 3000},
    ]
    labeled = assign_speakers(transcript_segments, diarization_segments)
    assert labeled[0]["speaker_label"] == "SPEAKER_00"
    assert labeled[1]["speaker_label"] == "SPEAKER_01"
    assert labeled[2]["speaker_label"] == "SPEAKER_01"  # nearest fallback, no overlap
    assert assign_speakers(transcript_segments, [])[0]["speaker_label"] == "SPEAKER_00"
    print("pipeline assign_speakers self-check passed")


if __name__ == "__main__":
    _demo()
