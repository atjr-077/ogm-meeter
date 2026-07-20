from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import schemas
from config import settings
from database import get_db, engine, Base
from models import Meeting, MeetingSummary, TranscriptEmbedding, TranscriptSegment
from services import audio_utils, embeddings, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
)
logger = logging.getLogger("meetingbot.main")


@app.on_event("startup")
async def startup_event():
    logger.info("Meeting Bot backend starting")


app = FastAPI(title="Meeting Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/session/start", response_model=schemas.SessionStartResponse)
async def start_session(db: AsyncSession = Depends(get_db)) -> schemas.SessionStartResponse:
    meeting = Meeting(status="active")
    db.add(meeting)
    await db.commit()
    await db.refresh(meeting)
    audio_utils.ensure_session_dirs(str(meeting.id))
    logger.info(f"meeting_id={meeting.id} event=session_started")
    return schemas.SessionStartResponse(session_id=str(meeting.id))


@app.post("/audio/chunk", response_model=schemas.ChunkUploadResponse)
async def upload_chunk(
    session_id: uuid.UUID = Query(...),
    ts: int = Query(...),
    audio: UploadFile = File(...),
) -> schemas.ChunkUploadResponse:
    content = await audio.read()
    audio_utils.save_chunk(str(session_id), ts, audio.filename, content)
    return schemas.ChunkUploadResponse(ok=True)


@app.post("/session/end", response_model=schemas.SessionEndResponse)
async def end_session(
    body: schemas.SessionEndRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> schemas.SessionEndResponse:
    meeting = await db.get(Meeting, body.session_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting not found")

    meeting.status = "processing"
    meeting.ended_at = datetime.now(timezone.utc)
    await db.commit()

    background_tasks.add_task(pipeline.process_meeting, str(body.session_id))
    logger.info(f"meeting_id={body.session_id} event=session_ended")
    return schemas.SessionEndResponse(ok=True, status="processing")


@app.get("/meetings", response_model=list[schemas.MeetingListItem])
async def list_meetings(db: AsyncSession = Depends(get_db)) -> list[Meeting]:
    result = await db.execute(select(Meeting).order_by(Meeting.started_at.desc()))
    return list(result.scalars().all())


@app.get("/meetings/{meeting_id}", response_model=schemas.MeetingDetailResponse)
async def get_meeting(
    meeting_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> schemas.MeetingDetailResponse:
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting not found")

    result = await db.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.meeting_id == meeting_id)
        .order_by(TranscriptSegment.started_ms)
    )
    segments = list(result.scalars().all())

    return schemas.MeetingDetailResponse(
        id=meeting.id,
        title=meeting.title,
        started_at=meeting.started_at,
        ended_at=meeting.ended_at,
        status=meeting.status,
        segments=[schemas.TranscriptSegmentOut.model_validate(s) for s in segments],
    )


@app.get("/meetings/{meeting_id}/summary")
async def get_summary(meeting_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    meeting = await db.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="meeting not found")

    summary = await db.get(MeetingSummary, meeting_id)
    if summary is None:
        return {"ready": False, "status": meeting.status}

    return {
        "ready": True,
        "title": meeting.title or "Untitled meeting",
        "started_at": meeting.started_at,
        "ended_at": meeting.ended_at,
        "summary": summary.summary or "",
        "key_topics": summary.key_topics or [],
        "decisions": summary.decisions or [],
        "action_items": summary.action_items or [],
        "participants": summary.participants or [],
    }


@app.post("/search", response_model=schemas.SearchResponse)
async def search(body: schemas.SearchRequest, db: AsyncSession = Depends(get_db)) -> schemas.SearchResponse:
    query_vector = embeddings.embed_texts([body.query])[0]

    stmt = (
        select(
            TranscriptEmbedding.meeting_id,
            TranscriptEmbedding.chunk_text,
            TranscriptEmbedding.embedding,
            Meeting.title,
            Meeting.started_at,
        )
        .join(Meeting, Meeting.id == TranscriptEmbedding.meeting_id)
    )
    rows = (await db.execute(stmt)).all()

    scored_rows = []
    for row in rows:
        emb = row.embedding
        if isinstance(emb, str):
            import json
            emb = json.loads(emb)
        sim = sum(x * y for x, y in zip(query_vector, emb))
        scored_rows.append((row, sim))

    scored_rows.sort(key=lambda x: x[1], reverse=True)

    results = [
        schemas.SearchResultItem(
            meeting_id=str(item[0].meeting_id),
            meeting_title=item[0].title or "Untitled meeting",
            meeting_date=item[0].started_at,
            score=round(float(item[1]), 4),
            snippet=item[0].chunk_text[:280],
        )
        for item in scored_rows[:body.limit]
    ]
    return schemas.SearchResponse(results=results)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Backend is running"
    }


from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
