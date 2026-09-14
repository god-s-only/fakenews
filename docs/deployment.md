# Deployment Guide

This page describes how to run the Fake News Detector in production, what the
runtime needs, and what guarantees the Phase 10 hardening provides.

## Runtime requirements

- Python 3.12 (TensorFlow compat; verified on 3.12)
- The two **promoted** artifacts at the repo root:
  - `my_model_lr.pkl` (scikit-learn LogisticRegression)
  - `my_tfidf_vectorizer.pkl` (TF-IDF vectorizer)
  Their SHA-256 digests are pinned in `reports/release_manifest.json` and
  verified at load time (`ModelService.load()` → fingerprints). If they drift,
  `/health/ready` still reports 200 but operators can detect the change from the
  reported `model_sha256`/`vectorizer_sha256`.
- NLTK English stopwords (downloaded automatically on first startup if absent).

## Options

### Plain process (uvicorn)

```bash
python -m pip install -r requirements.txt
python main.py            # or: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
make docker-up            # docker compose up --build
```

The image (see `Dockerfile`):
- Runs as the unprivileged `appuser`.
- Contains **only the promoted artifacts** (the legacy Keras `artifacts/baseline`
  files are deliberately not shipped).
- Gates its `HEALTHCHECK` on `/health/ready` so the container is only *healthy*
  once the detector can actually serve predictions.
- Bounds worker memory in `docker-compose.yml` (`deploy.resources.limits.memory`).

## Operational endpoints

| Endpoint        | Use                                                                 |
| --------------- | ------------------------------------------------------------------- |
| `/health/live`  | "is it up" — orchestrator liveness. Never depends on the model.     |
| `/health/ready` | "can it serve predictions" — readiness/health-check. 503 while loading. |
| `/health`       | Backwards-compatible blended check.                                 |
| `/info`         | Version + effective settings summary.                               |
| `/docs`         | Interactive OpenAPI documentation (off by default? keep as-is: served). |

## Observability

- Every request gets a `request_id` echoed on the response (`X-Request-ID`) and
  recorded in one structured access log line per request, e.g.:
  `method=POST path=/predict status=200 duration_ms=62.3 request_id=abc123...`.
- All `fakenews.*` log records carry the same `request_id`.
- `LOG_LEVEL` controls verbosity (default `INFO`).

## Rate limiting & caching caveats (per-process)

The rate limiter and the URL-extraction cache are **in-memory per process**.
Under a multi-worker or multi-replica deployment:

- Each worker/replica enforces its own rate budget and cache. Effective limits
  scale with worker count; for strict global limits use a shared store or a
  fronting gateway (out of scope).
- The cache is cleared when the process restarts; hot URLs re-populate quickly.

## Deployment checklist

1. Verify artifact hashes match `reports/release_manifest.json`.
2. Run `pytest` (272+ tests) before shipping the image.
3. Set a real `CORS_ORIGINS` (never `*` for production sites with credentials).
4. Put the service behind TLS and set `RATE_LIMIT_*` to match your traffic.
5. Point health checks at `/health/ready` and alarms at `/health/live`.
6. Collect `fakenews.access` and `fakenews` logs into your aggregator, keyed by
   `request_id`.