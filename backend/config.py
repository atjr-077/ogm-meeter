from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/meetingbot"
    openai_api_key: str = ""
    groq_api_key: str = ""
    hf_token: str = ""
    data_dir: str = "backend/data"
    transcriber_provider: str = "openai"
    summarizer_provider: str = "openai"
    transcribe_language: str = ""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, v: str) -> str:
        path = Path(v)
        return str(path if path.is_absolute() else (_PROJECT_ROOT / path).resolve())


settings = Settings()
