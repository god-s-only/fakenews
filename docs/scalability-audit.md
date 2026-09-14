# Scalability Audit — Fake News Detector

Phase 10 engineering audit of the production detector. This document records
the findings that drive the scalability and production-hardening work; it is a
snapshot of the architecture as delivered at the start of the phase.

## System Under Audit

* **Runtime**: FastAPI (v2 factory app in `app/`), uvicorn, Python 3.12.
* **Model**: promoted scikit-learn TF-IDF + LogisticRegression
  (`my_model_lr.pkl` + `my_tfidf_vectorizer.pkl`); frozen artifacts, verified
  by SHA-256 against the phase 8/9 release manifest.
* **Legacy backend**: Keras `my_model.h5` + `countvectorizer.pkl`, preserved
  under `artifacts/baseline/` as a rollback path.
* **Frontend**: static vanilla HTML/CSS/JS served from `frontend/`.
* **Tests**: 192 pytest tests (baseline, green).
* **Deployment**: single-purpose Dockerfile + compose; no CI workflow yet.

## What Scales Well Already

* The model is loaded **once** in the FastAPI lifespan, not per request.
* `/predict` and `/predict-url` are declared as `def` (sync) endpoints, so
  FastAPI/Starlette already runs them in its worker thread pool — blocking
  requests and CPU-bound inference do **not** stall the event loop.
* URL fetching is bounded: connect/read timeouts, redirect cap, response-size
  cap, content-type gate, SSRF/private-network rejection with per-hop
  re-validation.
* The prediction log is a thread-safe, bounded ring buffer.
* Preprocessing has no hidden global mutable state per call (fresh stemmer is
  created per call); NLTK stopwords are cached.
* Config is centralized in `app/config.py` and env-driven.

## Identified Bottlenecks (Worst First)

### B1 — New `requests.Session` per URL analysis
`UrlFetcher()` builds a fresh `requests.Session` for every `fetch_article`
call. Each session opens and tears down its own connection/TLS state, so
repeated URL analysis pays a full TCP+TLS handshake every time and never
reuses sockets. This is the largest single scalability waste in the URL path.

**Fix**: thread-local session reuse with a bounded `HTTPAdapter` connection
pool, plus bounded retry/backoff.

### B2 — Per-request vectorizer feature-name regeneration
Every explanation call invokes `get_feature_names_out()` on the vectorizer
(scikit-learn rebuilds the full feature-name array). For a 36k-term
vocabulary this is pure repetition of immutable model metadata.

**Fix**: memoize feature names once at model load.

### B3 — Repeated URL analysis recomputes everything
Re-checking the same article URL re-fetches, re-parses and re-predicts from
scratch. Cheap to make deterministic and bounded with a small TTL cache keyed
on the URL; article bodies must never be persisted.

**Fix**: bounded, TTL, in-memory cache storing only the analysis *response*
(never the article text).

### B4 — No request-level limits on the *body* itself
`MAX_INPUT_LENGTH` bounds the `news` field via Pydantic and an explicit
endpoint check, but there is no bound on the raw HTTP request body, so a
client can stream a multi-MB request before validation rejects it.

**Fix**: middleware-enforced `MAX_REQUEST_BODY_SIZE`, config-driven.

### B5 — No rate limiting
The prediction endpoints are unauthenticated and open to unbounded client
abuse (each request costs a model inference and, for URLs, a network fetch).

**Fix**: lightweight in-memory, configurable rate limiter with a clear 429
response. Document the per-process caveat for multi-worker deployments.

### B6 — Single blended `/health` with no liveness/readiness split
`/health` returns `200 ok` even when the model is missing, so an orchestrator
cannot distinguish "process alive" from "ready to serve predictions".

**Fix**: split liveness (`/health/live`) from readiness (`/health/ready`).

### B7 — No request correlation / structured observability
No request IDs are propagated, no per-request latency/status is logged, and
the generic 500 handler swallows the exception without a server-side record.

**Fix**: request-ID middleware + structured per-request log line + safe error
logging (never article bodies, credentials or authorization headers).

### B8 — Thread-safety assumptions are implicit
Concurrent prediction and URL analysis rely on unstated assumptions about
scikit-learn/requests thread-safety. There is no regression test proving safe
concurrent access.

**Fix**: an explicit concurrency test suite that hammers shared services.

### B9 — Hardcoded limits scattered in schemas
`UrlRequest.url` hardcodes `max_length=2048` in the schema (`app/schemas.py`),
outside the settings object.

**Fix**: centralize every limit in `app/config.py` and reference it.

### B10 — Docker/CI gaps
* Image runs as **root** and only copies the *legacy* artifacts
  (`my_model.h5`/`countvectorizer.pkl`) while the promoted detector uses
  different files — the container cannot serve the production model.
* HEALTHCHECK uses the blended `/health` (200 even when not ready).
* No `.github/workflows` CI: no automated test/lint validation on PRs.

### B11 — DNS-rebinding window in the SSRF guard (documented limitation)
`_check_resolved_addresses` resolves the hostname and then `requests` may
resolve it again at connection time (TOCTOU). Each redirect hop is re-checked,
which bounds the risk, but the resolution is not pinned to the fetch.
Recorded here as a known limitation; closing it fully (pinning the resolved
address and connecting to the IP with a validated Host header) is deferred so
the existing, well-tested SSRF behaviour is not weakened for a TTL-sized
improvement. Do not add hostname-based caching for allow-decisions anywhere in
this phase.

## Non-Goals / Anti-Goals

* No Redis or other external service — the app runs standalone; the
  in-memory rate limiter/cache are documented as per-process.
* No retraining, no model replacement, no vectorizer replacement.
* No modification of the frozen artifacts or their fingerprints.
* No redesign of the frontend; only minimal, accessibility-preserving
  additions (loading/rate-limit/readiness states).

## Measuring Baselines

Baseline numbers for tests, startup, prediction latency, URL-analysis latency
and memory are recorded via `scripts/benchmark.py` into
`reports/baseline/*.json` at the start of the phase, before these bottlenecks
are addressed, so the fixes can be justified by comparison.