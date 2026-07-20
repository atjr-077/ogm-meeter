# Meeting Bot

Offline, in-person meeting recorder and AI notetaker. The browser records room
audio through the device microphone and uploads 12-second chunks to a FastAPI
backend. When you stop recording, the backend concatenates the chunks,
normalizes them to 16kHz mono WAV, runs full-meeting speaker diarization
(pyannote), transcribes the audio (OpenAI `gpt-4o-transcribe`), aligns
transcript segments to speakers, generates a structured summary (OpenAI
`gpt-4.1-mini`), stores semantic embeddings in pgvector, and exposes a search
API.

This is **not** a Zoom/Meet/Teams bot — it records whatever the device
microphone hears in the room.

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ with the `pgvector` extension available
- `ffmpeg` on your `PATH`
- An OpenAI API key
- A Hugging Face access token that has accepted the user agreements for
  [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)

## 1. Install PostgreSQL + pgvector

**Windows** — install PostgreSQL from https://www.postgresql.org/download/windows/,
then install pgvector for your PostgreSQL version from
https://github.com/pgvector/pgvector#windows (or use `pip install pgvector`
only for the Python client — the Postgres **extension itself** still needs to
be compiled/installed separately, see the pgvector README's Windows section).

**macOS (Homebrew)**

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
```

**Ubuntu/Debian**

```bash
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-16-pgvector   # package name depends on your PG version
sudo systemctl start postgresql
```

## 2. Create the database and apply the schema

```bash
# create the database (adjust user/host as needed)
createdb -U postgres meetingbot

# apply schema (creates extensions, tables, and the ivfflat index)
psql -U postgres -d meetingbot -f db/schema.sql
```

If `CREATE EXTENSION vector` fails with "extension \"vector\" is not
available", pgvector isn't installed into your PostgreSQL server yet — see
step 1.

## 3. Install ffmpeg

**Windows**: download a build from https://www.gyan.dev/ffmpeg/builds/ and add
its `bin/` folder to your `PATH`. Verify with:

```powershell
ffmpeg -version
```

**macOS**: `brew install ffmpeg`
**Ubuntu/Debian**: `sudo apt install ffmpeg`

## 4. Backend setup

```bash
cd backend
python -m venv .venv

# activate the venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows PowerShell/cmd

pip install -r requirements.txt
```

Copy the env file and fill in your secrets:

```bash
cd ..
cp .env.example .env
```

Edit `.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/meetingbot
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
DATA_DIR=backend/data
TRANSCRIBER_PROVIDER=openai
SUMMARIZER_PROVIDER=openai
```

## 5. Run the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The first request that triggers diarization or transcription will download
the pyannote and sentence-transformers models — this can take a few minutes
the first time.

## 6. Serve the frontend

In a second terminal:

```bash
cd frontend
python -m http.server 8080
```

Open **http://localhost:8080** in your browser. The page talks to the backend
on port 8000 automatically (it derives the API host from
`window.location.hostname`).

## 7. Use it from a phone on the same Wi-Fi

1. Find your PC's LAN IP:
   - Windows: `ipconfig` → look for "IPv4 Address" (e.g. `192.168.1.42`)
   - macOS/Linux: `ifconfig` or `ip addr`
2. Make sure both servers are bound to `0.0.0.0` (the commands above already
   do this for the backend; `python -m http.server 8080` also binds to all
   interfaces by default).
3. On your phone, open `http://192.168.1.42:8080` (use your actual LAN IP).
4. Allow microphone access when prompted.

Both devices must be on the same Wi-Fi network, and your PC's firewall must
allow inbound connections on ports 8000 and 8080.

## API summary

| Method | Path                        | Purpose                              |
|--------|-----------------------------|---------------------------------------|
| POST   | `/session/start`            | Create a meeting, return `session_id` |
| POST   | `/audio/chunk`               | Upload one audio chunk                |
| POST   | `/session/end`               | Stop, trigger background processing   |
| GET    | `/meetings`                  | List meetings, newest first           |
| GET    | `/meetings/{id}`             | Diarized transcript segments          |
| GET    | `/meetings/{id}/summary`     | Structured AI summary (polls until ready) |
| POST   | `/search`                    | Semantic search over all transcripts  |

## Troubleshooting

**Microphone permission denied / no audio recorded**
Browsers only allow microphone access on `localhost` or over HTTPS — accessing
the frontend via a plain LAN IP (`http://192.168.x.x:8080`) works in most
mobile browsers for `getUserMedia`, but if your browser blocks it, check
Settings → Site permissions → Microphone for the site, and make sure no other
app is holding the microphone.

**`ffmpeg failed` errors in the backend logs**
Usually means `ffmpeg` isn't on `PATH` for the process running uvicorn.
Restart your terminal after installing it, and confirm `ffmpeg -version` works
from the same shell you launch uvicorn from.

**Hugging Face token / gated model errors from pyannote**
`Pipeline.from_pretrained` will raise a 401/403 if `HF_TOKEN` is missing or if
you haven't accepted the user agreement on the model pages for
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on
huggingface.co while logged in as the account that owns the token.

**`extension "vector" is not available`**
The pgvector **extension** isn't installed on your PostgreSQL server (this is
different from the `pgvector` **Python package**, which only talks to it).
Install the extension for your OS/PG version (step 1) and re-run
`psql -f db/schema.sql`.

**iOS Safari recording quirks**
- iOS Safari requires a user gesture (the Start button tap) to begin
  `getUserMedia` — this app already does that.
- Older iOS versions only support `audio/mp4` for `MediaRecorder`, not
  `audio/webm`; the frontend picks the first supported MIME type
  automatically via `MediaRecorder.isTypeSupported`.
- If recording silently produces empty chunks, make sure Safari has
  microphone permission under iOS Settings → Safari → Microphone, and that
  Low Power Mode isn't suspending the tab in the background — keep the app in
  the foreground while recording.

## Project layout

```text
meeting-bot/
├── backend/
│   ├── main.py              FastAPI app, routes
│   ├── config.py            Settings from .env
│   ├── database.py          Async SQLAlchemy engine/session
│   ├── models.py             ORM models
│   ├── schemas.py            Pydantic request/response models
│   ├── requirements.txt
│   ├── services/
│   │   ├── audio_utils.py    Chunk storage, ffmpeg concat/normalize
│   │   ├── diarizer.py       pyannote speaker diarization
│   │   ├── transcriber.py    OpenAI transcription (swappable provider)
│   │   ├── summarizer.py     OpenAI structured summary (swappable provider)
│   │   ├── embeddings.py     sentence-transformers embeddings
│   │   └── pipeline.py       End-to-end background processing
│   └── data/                 Uploaded chunks + processed audio (gitignored)
├── frontend/
│   └── index.html            Single-file vanilla JS UI
├── db/
│   └── schema.sql
├── .env.example
├── .gitignore
└── README.md
```
