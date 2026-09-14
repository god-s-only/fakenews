# Fake News Detector — Architecture

## Overview

A FastAPI web application that classifies news text as **real**, **fake** or
**uncertain**. The detector is a frozen, promoted scikit-learn TF-IDF +
LogisticRegression model; a legacy Keras network is preserved as a rollback
backend. Users can paste text or submit a URL (fetched and extracted safely
server-side). Predictions carry confidence, probability scores and
influential-word explanations.

```
                 ┌────────────────────────────────────────────────┐
                 │  browser (vanilla HTML/CSS/JS, localStorage)   │
                 └───────────────┬────────────────────────────────┘
                                 │ HTTP (CORS)
                 ┌───────────────▼────────────────────────────────┐
                 │        FastAPI application (app/main.py)       │
                 │  middleware: CORS, request ID, rate limit,     │
                 │              body-size, logging                │
                 │  routes: / /predict /predict-url /health*      │
                 │         /info /history /docs                   │
                 └───────┬───────────────────┬──────┬─────────────┘
                         │                   │      │
                 ┌───────▼───────┐   ┌───────▼──┐   │
                 │ ModelService  │   │ scraper  │   │ prediction_log (ring)
                 │ (singleton,   │   │ UrlFetcher│   │
                 │  loaded once) │   │ (SSRF)   │   └────────────┐
                 └───────┬───────┘   └───────┬──┘                │
                         │                   │                   │
                 ┌───────▼───────┐   ┌───────▼────────┐   ┌──────▼──────┐
                 │ preprocessing │   │ requests       │   │ config/settings
                 │ (normalize +  │   │ (bounded pool) │   │ env-driven   │
                 │  clean)       │   └────────────────┘   └─────────────┘
                 └───────┬───────┘
                 ┌───────▼───────┐
                 │ sklearn TF-IDF│  <— frozen artifacts (fingerprinted)
                 │ + LogisticReg │
                 └───────────────┘
```

## Core modules

| Module | Responsibility |
| --- | --- |
| `app/config.py` | Typed, env-driven settings; `validate()` warnings; `summary()`. |
| `app/main.py` | `create_app()` factory, lifespan, routes, middleware wiring. |
| `app/model.py` | `ModelService`: load + predict + explain (sklearn & keras backends). |
| `app/preprocessing.py` | Source-marker normalisation + training-identical cleaning. |
| `app/scraper.py` | `UrlFetcher`: SSRF-guarded fetch, size/timeout/redirect caps, extraction. |
| `app/schemas.py` | Pydantic v2 request/response models. |
| `app/prediction_log.py` | Thread-safe in-memory ring buffer of recent predictions. |
| `app/logging_config.py` | Central logging setup (`LOG_LEVEL`). |
| `app/verify_model.py` | Standalone model-verification CLI. |

## Request flow

### `/predict` (pasted text)
1. Pydantic validates `news` (non-empty, near-empty rejected, length cap).
2. `_require_model()` returns the loaded `ModelService` or 503.
3. Text length is checked again against `MAX_INPUT_LENGTH`.
4. `ModelService.predict()` runs: `normalize_news_markers` → `clean_single_text`
   → vectorize → `predict_proba` → verdict (uncertainty band) → explanation.
5. Result is stored in the prediction ring and returned.

### `/predict-url`
1. Pydantic validates the URL (scheme, non-empty, length cap).
2. `UrlFetcher.fetch_article()`: scheme/host validation, DNS + private-network
   SSRF check, manual redirect walk (each hop re-validated), content-type gate,
   size-capped read, JSON-LD/semantic extraction, listing-page rejection.
3. Extracted text flows through the same prediction pipeline.

## Concurrency model

* `/predict` and `/predict-url` are **sync** endpoints (`def`, not `async
  def`), so Starlette executes them in its worker thread pool. Blocking
  `requests` calls and CPU-bound inference never block the event loop.
* Scikit-learn estimators are read-only during inference (no state mutation),
  vectorizer `transform` is thread-safe, and a fresh `PorterStemmer()` is
  created per call in `clean_single_text` — concurrent prediction has no
  shared mutable state.
* `prediction_log`, the bounded caches and the rate limiter are guarded by
  locks/thread-local storage.

## Security posture

* SSRF: HTTP/HTTPS only, localhost/.local blocked, DNS resolution re-checked
  against private/reserved/loopback/link-local ranges, per-redirect-hop
  re-validation, response-size cap, connect/read timeouts, redirect cap.
* Input: strict Pydantic validation, length caps, request-body cap.
* Rate limiting: in-memory per-process limiter (documented as per-worker).
* Logging: structured request IDs; article bodies, credentials and
  authorization headers are never logged.
* No secrets in the repository; `.env` is git-ignored (`.env.example` tracked).

## Observability

* `/health` (blended status), `/health/live` (liveness) and `/health/ready`
  (model readiness + fingerprints).
* Per-request log lines with request ID, method, path, status and duration.
* `reports/` contain frozen model fingerprints, evaluation metrics and phase-9
  robustness findings.

## Deployment

* Docker image: `python:3.12-slim`, non-root user, `EXPOSE 8000`,
  HEALTHCHECK splitting liveness/readiness.
* `docker-compose.yml` maps `8000:8000` and merges `.env`.
* GitHub Actions CI runs the full test suite on PRs/pushes (Python 3.12).
* See `docs/deployment.md` for production guidance.