-- Meeting Bot database schema
-- Apply with: psql -U postgres -d meetingbot -f db/schema.sql

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meetings (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title       text,
    started_at  timestamptz NOT NULL DEFAULT now(),
    ended_at    timestamptz,
    status      text NOT NULL CHECK (status IN ('active', 'processing', 'done', 'error')),
    audio_path  text
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id            bigserial PRIMARY KEY,
    meeting_id    uuid NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_label text NOT NULL,
    text          text NOT NULL,
    started_ms    integer NOT NULL,
    ended_ms      integer NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_transcript_segments_meeting
    ON transcript_segments (meeting_id, started_ms);

CREATE TABLE IF NOT EXISTS meeting_summaries (
    meeting_id        uuid PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    summary           text,
    key_topics        jsonb,
    decisions         jsonb,
    action_items      jsonb,
    participants      jsonb,
    raw_llm_response  jsonb
);

CREATE TABLE IF NOT EXISTS transcript_embeddings (
    id          bigserial PRIMARY KEY,
    meeting_id  uuid NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    chunk_text  text NOT NULL,
    embedding   vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transcript_embeddings_meeting
    ON transcript_embeddings (meeting_id);

-- ivfflat cosine index for approximate nearest-neighbor search.
-- `lists` should be roughly sqrt(row_count); 100 is a reasonable default for dev-scale data.
CREATE INDEX IF NOT EXISTS idx_transcript_embeddings_embedding
    ON transcript_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
