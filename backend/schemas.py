from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionStartResponse(BaseModel):
    session_id: str


class ChunkUploadResponse(BaseModel):
    ok: bool = True


class SessionEndRequest(BaseModel):
    session_id: uuid.UUID


class SessionEndResponse(BaseModel):
    ok: bool
    status: str


class MeetingListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    started_at: datetime
    ended_at: datetime | None
    status: str


class TranscriptSegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    speaker_label: str
    text: str
    started_ms: int
    ended_ms: int


class MeetingDetailResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    started_at: datetime
    ended_at: datetime | None
    status: str
    segments: list[TranscriptSegmentOut]


class ActionItem(BaseModel):
    owner: str
    task: str
    due: str | None = None


class SummaryReadyResponse(BaseModel):
    ready: bool = True
    summary: str
    key_topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    participants: list[str]


class SummaryNotReadyResponse(BaseModel):
    ready: bool = False
    status: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResultItem(BaseModel):
    meeting_id: str
    meeting_title: str
    meeting_date: datetime
    score: float
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
