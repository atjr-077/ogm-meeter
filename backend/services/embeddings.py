from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("meetingbot.embeddings")

_model: SentenceTransformer | None = None


def load_model() -> None:
    """Load the embedding model once. Call this at app startup."""
    global _model
    if _model is None:
        logger.info("event=embedding_model_loading model=all-MiniLM-L6-v2")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("event=embedding_model_loaded")


def get_model() -> SentenceTransformer:
    if _model is None:
        load_model()
    return _model  # type: ignore[return-value]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def chunk_text(text: str, chunk_words: int = 500, overlap_words: int = 50) -> list[str]:
    """Split into ~chunk_words word windows with overlap_words of overlap."""
    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_words]
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def _demo() -> None:
    words = [f"w{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=500, overlap_words=50)
    assert len(chunks) == 3, chunks
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w450"
    assert chunks[-1].split()[-1] == "w1199"
    assert chunk_text("") == []
    print("embeddings chunk_text self-check passed")


if __name__ == "__main__":
    _demo()
