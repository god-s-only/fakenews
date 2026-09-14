"""Pydantic request/response schemas for the Fake News Detector API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import settings

Verdict = Literal["real", "fake", "uncertain"]  # noqa: F841 - re-exported for docs

__all__ = [
    "Verdict",
    "PredictRequest",
    "UrlRequest",
    "ExplanationItem",
    "Explanation",
    "PredictResponse",
    "HealthResponse",
    "ErrorResponse",
]


class PredictRequest(BaseModel):
    """Request payload for analysing raw article text."""

    news: str = Field(
        ...,
        description="The article text to analyse.",
        max_length=settings.MAX_INPUT_LENGTH,
    )

    @field_validator("news")
    @classmethod
    def validate_news(cls, value: str) -> str:
        if value is None:
            raise ValueError("News text must contain enough content to analyze.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("News text must contain enough content to analyze.")
        # Reject near-empty input (fewer than ~10 meaningful characters).
        if len(cleaned) < 10:
            raise ValueError("News text must contain enough content to analyze.")
        return value


class UrlRequest(BaseModel):
    """Request payload for analysing an article located at a URL."""

    url: str = Field(..., description="The article URL to fetch and analyse",
                     max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError("A URL must be provided.")
        stripped = value.strip()
        lowered = stripped.lower()
        if not (lowered.startswith("http://") or lowered.startswith("https://")):
            raise ValueError("Only http and https URLs are supported.")
        return stripped


class ExplanationItem(BaseModel):
    """A single word's influence on a prediction."""

    word: str
    impact: float
    direction: Literal["real", "fake"]


class Explanation(BaseModel):
    """Explainability payload for a prediction."""

    top_influential_words: list[ExplanationItem]


class PredictResponse(BaseModel):
    """Response returned by /predict and /predict-url."""

    label: Literal["real", "fake", "uncertain"]
    confidence: float
    probability_real: float
    probability_fake: float
    explanation: Explanation | None = None
    source_type: Literal["text", "url"] = "text"
    source: str | None = None
    page_title: str | None = None


class HealthResponse(BaseModel):
    """Response returned by GET /health.

    ``model_loaded``/``vectorizer_loaded`` are the readiness flags. The
    ``model_*``/``vectorizer_*`` diagnostics expose the exact runtime backend
    and artifact fingerprints so an operator can verify WHICH trained detector
    a running server is actually serving (sklearn vs legacy Keras), without
    any model weights or data leaving the process.
    """

    status: Literal["ok"]
    model_loaded: bool
    vectorizer_loaded: bool
    model_backend: str | None = None
    model_file: str | None = None
    vectorizer_file: str | None = None
    model_sha256: str | None = None
    vectorizer_sha256: str | None = None
    vocab_size: int | None = None


class ErrorResponse(BaseModel):
    """Generic structured error returned when analysis fails."""

    detail: str
