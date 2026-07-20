from __future__ import annotations

import json
from abc import ABC, abstractmethod

from openai import OpenAI
from pydantic import BaseModel

from config import settings


class ActionItem(BaseModel):
    owner: str
    task: str
    due: str | None = None


class SummaryResult(BaseModel):
    summary: str
    key_topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    participants: list[str]


SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_topics": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "task": {"type": "string"},
                    "due": {"type": ["string", "null"]},
                },
                "required": ["owner", "task", "due"],
                "additionalProperties": False,
            },
        },
        "participants": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_topics", "decisions", "action_items", "participants"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You summarize in-person meeting transcripts. Return only valid JSON matching the given "
    "schema. Do not invent facts, names, or action items that are not supported by the "
    "transcript. Preserve exact participant names as they appear in the transcript; if a "
    "speaker's real name is never stated, use their diarization label (e.g. SPEAKER_00) as "
    "their participant name. The transcript may be in Hindi, English, Hinglish (code-mixed "
    "Hindi/English), or transcribed in Devanagari, Latin, or Urdu script — read it in whatever "
    "language or script it appears, and always write the summary, key topics, decisions, and "
    "action items in clear English.\n\n"
    "The 'summary' field must be a detailed, multi-paragraph account of the meeting, not a "
    "single sentence — scale its length to how much was actually said. Walk through what was "
    "discussed in the order it came up, cover the reasoning and context behind each point (not "
    "just the topic label), note who said what when it's attributable, and call out anything "
    "notable even if it doesn't rise to a formal decision or action item. Still ground every "
    "sentence in the transcript — being detailed does not mean padding with generic filler or "
    "invented specifics.\n\n"
    "'key_topics' should capture every distinct subject, theme, or point of focus actually "
    "raised — this includes informal, personal, or off-topic threads (e.g. 'criticism of a "
    "colleague's behavior', 'small talk about the weekend'), not just formal agenda items. "
    "Phrase each topic as a short, specific noun phrase grounded in the transcript. Only return "
    "an empty key_topics array if the transcript has no discernible substantive content at all "
    "(e.g. silence, noise, or a single unintelligible fragment).\n\n"
    "'decisions' and 'action_items' are stricter: only include a decision if the speakers "
    "actually reached a conclusion or agreement, and only include an action item if a concrete "
    "task with an identifiable owner was stated or clearly implied. Casual conversation with no "
    "such outcome should leave these arrays empty — do not stretch or fabricate to fill them."
)


class Summarizer(ABC):
    """Provider interface: swap OpenAI for Groq/Gemini/etc without touching business logic."""

    @abstractmethod
    def summarize(self, transcript_text: str) -> tuple[SummaryResult, dict]:
        """Return (validated summary, raw provider response as a JSON-able dict)."""


class OpenAISummarizer(Summarizer):
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)

    def summarize(self, transcript_text: str) -> tuple[SummaryResult, dict]:
        resp = self._client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n\n{transcript_text}"},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "meeting_summary",
                    "schema": SUMMARY_JSON_SCHEMA,
                    "strict": True,
                }
            },
        )
        raw = json.loads(resp.output_text)
        result = SummaryResult.model_validate(raw)
        return result, resp.model_dump()


class GroqSummarizer(Summarizer):
    """Groq's OpenAI-compatible endpoint only has chat completions (no Responses API),
    so JSON mode + prompt-embedded schema + Pydantic validation stands in for strict
    json_schema enforcement."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")

    def summarize(self, transcript_text: str) -> tuple[SummaryResult, dict]:
        resp = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + " Respond with a single JSON object matching this JSON schema: "
                    + json.dumps(SUMMARY_JSON_SCHEMA),
                },
                {"role": "user", "content": f"Transcript:\n\n{transcript_text}"},
            ],
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)
        result = SummaryResult.model_validate(raw)
        return result, resp.model_dump()


_PROVIDERS: dict[str, type[Summarizer]] = {
    "openai": OpenAISummarizer,
    "groq": GroqSummarizer,
}


def get_summarizer() -> Summarizer:
    provider = _PROVIDERS.get(settings.summarizer_provider)
    if provider is None:
        raise ValueError(f"unknown summarizer provider: {settings.summarizer_provider}")
    return provider()
