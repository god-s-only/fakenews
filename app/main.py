"""Fake News Detector — FastAPI application entry point.

The app loads the trained model and vectorizer during application startup
(lifespan), validates inputs through Pydantic, serves a static frontend from
``/``, and exposes a shared prediction pipeline used by both the pasted-text
and URL endpoints.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.model import ModelLoadError, ModelService
from app.prediction_log import PredictionEntry, prediction_log
from app.preprocessing import ensure_stopwords_available
from app.scraper import ScrapeError, fetch_article
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    UrlRequest,
)

logger = logging.getLogger("fakenews")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class AppState:
    """Shared mutable state stored on the FastAPI application."""

    def __init__(self) -> None:
        self.model: ModelService | None = None


state = AppState()


def _sha256_file(path: Path) -> str:
    """SHA-256 of a pickle/artifact file (small; computed once at startup)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ensure NLTK data is available before any predictions.
        ensure_stopwords_available()
        # Log configuration warnings before model load.
        for warning in settings.validate():
            logger.warning("Config: %s", warning)
        # Load the model/vectorizer once during startup and store in state.
        logger.info(
            "Loading model from %s and vectorizer from %s",
            settings.model_file,
            settings.vectorizer_file,
        )
        service = ModelService(settings.model_file, settings.vectorizer_file)
        try:
            service.load()
        except ModelLoadError as exc:
            # Do not silently continue; the app is unusable for predictions.
            logger.error("Model load failed: %s", exc)
            state.model = None
            raise RuntimeError(str(exc)) from exc
        # Record artifact fingerprints so /health can prove which detector is
        # actually live (e.g. sklearn Candidate D vs legacy Keras baseline).
        service.model_sha256 = _sha256_file(settings.model_file)
        service.vectorizer_sha256 = _sha256_file(settings.vectorizer_file)
        service.vocab_size = len(service._vectorizer.vocabulary_)
        state.model = service
        logger.info(
            "Model and vectorizer loaded successfully "
            "(backend=%s model=%s vocab=%d)",
            service._backend, settings.model_file.name, service.vocab_size,
        )
        logger.info("Uncertainty threshold: %s", settings.UNCERTAINTY_THRESHOLD)
        logger.info("Top features: %s", settings.TOP_FEATURES)
        yield
        state.model = None

    application = FastAPI(
        title="Fake News Detector",
        description=(
            "Analyse whether a piece of news text (pasted or at a URL) is "
            "likely to be real or fake, including explainability and "
            "uncertainty handling."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(Exception)
    def unhandled_exception(request: Request, exc: Exception):
        """Return a clean, stack-trace-free 500 response for any unexpected error."""
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again."},
        )

    @application.exception_handler(ScrapeError)
    def scrape_error(request: Request, exc: ScrapeError):
        """Map URL-analysis failures to a friendly 422 with a stable category.

        The technical ``detail`` and full redirect trace are only logged at
        DEBUG level and never exposed to the client.
        """
        logger.debug(
            "ScrapeError category=%s %s final_url=%s redirects=%s detail=%s",
            exc.category,
            exc,
            exc.final_url,
            list(exc.redirects),
            exc.detail,
        )
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "category": exc.category},
        )

    # Serve the static frontend.
    if FRONTEND_DIR.is_dir():
        application.mount(
            "/static",
            StaticFiles(directory=FRONTEND_DIR),
            name="static",
        )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        service = state.model
        model_loaded = bool(service and service.model_is_loaded)
        vectorizer_loaded = bool(service and service.vectorizer_is_loaded)
        if (service is not None and service.is_loaded
                and getattr(service, "_backend", None) is not None):
            model_file = service.model_file.name
            vectorizer_file = service.vectorizer_file.name
            model_backend = service._backend
            model_sha256 = getattr(service, "model_sha256", None)
            vectorizer_sha256 = getattr(service, "vectorizer_sha256", None)
            vocab_size = getattr(service, "vocab_size", None)
        else:
            model_file = vectorizer_file = model_backend = None
            model_sha256 = vectorizer_sha256 = vocab_size = None
        return HealthResponse(
            status="ok",
            model_loaded=model_loaded,
            vectorizer_loaded=vectorizer_loaded,
            model_backend=model_backend,
            model_file=model_file,
            vectorizer_file=vectorizer_file,
            model_sha256=model_sha256,
            vectorizer_sha256=vectorizer_sha256,
            vocab_size=vocab_size,
        )

    @application.get("/info")
    def info():
        """Return application version and configuration summary."""
        return {
            "version": "2.0.0",
            "settings": settings.summary(),
        }

    @application.get("/history")
    def get_history(limit: int = 20):
        """Return recent predictions made via the API."""
        entries = prediction_log.recent(limit=limit)
        return [
            {
                "label": e.label,
                "confidence": e.confidence,
                "probability_real": e.probability_real,
                "probability_fake": e.probability_fake,
                "source_type": e.source_type,
                "input_preview": e.input_preview,
            }
            for e in entries
        ]

    def _require_model() -> ModelService:
        if state.model is None or not state.model.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="The detector model is not loaded. Please try again later.",
            )
        return state.model

    def _to_response(
        service: ModelService,
        raw_text: str,
        source_type: str,
        source: str | None = None,
        page_title: str | None = None,
    ) -> PredictResponse:
        prediction = service.predict(raw_text)
        response = PredictResponse(
            label=prediction.label,
            confidence=prediction.confidence,
            probability_real=round(prediction.probability_real * 100.0, 2),
            probability_fake=round(prediction.probability_fake * 100.0, 2),
            explanation={
                "top_influential_words": [
                    {
                        "word": item.word,
                        "impact": item.impact,
                        "direction": item.direction,
                    }
                    for item in prediction.explanation
                ]
            }
            if prediction.explanation
            else None,
            source_type=source_type,
            source=source,
            page_title=page_title,
        )
        prediction_log.append(
            PredictionEntry(
                label=prediction.label,
                confidence=prediction.confidence,
                probability_real=prediction.probability_real,
                probability_fake=prediction.probability_fake,
                source_type=source_type,
                input_preview=raw_text[:100],
            )
        )
        return response

    @application.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        service = _require_model()
        text = req.news.strip()
        if len(text) > settings.MAX_INPUT_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"Input text too long ({len(text)} chars). "
                       f"Maximum is {settings.MAX_INPUT_LENGTH} characters.",
            )
        return _to_response(service, text, "text")

    @application.post("/predict-url", response_model=PredictResponse)
    def predict_url(req: UrlRequest) -> PredictResponse:
        service = _require_model()
        extract = fetch_article(req.url)
        if not extract.text.strip():
            raise HTTPException(
                status_code=422,
                detail="We retrieved the page, but couldn't identify the article content.",
            )
        return _to_response(
            service,
            extract.text,
            "url",
            source=req.url,
            page_title=extract.title or None,
        )

    return application


app = create_app()
