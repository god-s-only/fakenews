# Concurrency & Blocking-Operation Model

This page documents how the service behaves under concurrency, where blocking
work happens, and the guarantees (and limits) of the thread model.

## Threading model

- The FastAPI application is served by a thread-per-request executor (Starlette
  default). CPU-bound work therefore never blocks the event loop; it occupies a
  worker thread for its duration.
- `/predict` and `/predict-url` are pure request/response handlers with no
  shared mutable per-request state.

## Known blocking operations & where they run

| Operation | Cost | Runs in | Notes |
|-----------|------|---------|-------|
| Model/vectorizer `pickle.load` | ~40 ms | lifespan shutdown (worker thread)* | One-time `ModelService.load()` shares the result via `app.state`. |
| Login/feature-name materialisation (O(vocab)=36,862) | ms | `ModelService.load()` (once) | Cached; never recomputed per request. |
| `predict_proba` + sparse TF-IDF transform | ~45–50 ms | request worker thread | Single-threaded sklearn; safe cross-request because fits are immutable after load. |
| BeautifulSoup parse + text extraction | ~100 ms–1 s (div-heavy pages) | request worker thread | Largest per-request CPU cost after prediction. |
| DNS resolution + HTTP fetch | up to 2 × timeout | request worker thread (blocking `requests` with bounded retries) | See scraper session pool below. Never on the event loop. |

*A lifespan startup within `with TestClient(...)` blocks the caller; under
uvicorn the lifespan runs on a dedicated worker.

## Thread-safety guarantees

- **`ModelService`**: frozen after `load()`; attribute reads are safe. The only
  runtime mutation is lazy materialisation of `_feature_names_cache`, which is
  pre-populated by `load()` so every request sees it already set. If a caller
  ever constructs a service and skips `load()`, the lazy rebuild is a benign
  single-writer race (idempotent value; worst case two threads compute the same
  list).
- **`UrlFetcher`**: one `requests.Session` per worker thread (thread-local),
  created lazily and reused across fetches. Sessions are NOT safe to share
  across threads; keeping them thread-local gives connection reuse without
  locks or cross-talk. Connection pooling is bounded (pool_connections=10,
  pool_maxsize=20) and retries are bounded (`Retry(total=3)`, connect-only,
  GET-only, slight backoff) — see `app/scraper.py`.

## Scalability measures implemented

- Reusable per-thread HTTP sessions with bounded adapter pools (fixes audits B1).
- Feature-name vector materialised once per load instead of per request (B2).
- Early pre-fetch rejection of listing-path URLs (B3, cheap path): path markers
  checked before any network I/O; the authoritative listing check still runs on
  the final redirect URL and includes the site-root rule.
- `_container_candidates` uses a single O(n) scan instead of a full sort for
  pages without semantic containers (reduces parser cost on div-heavy pages).
- Article text extracted once per candidate, not re-extracted for both the
  threshold check and the returned result (B3).

## Known limits (documented, not hidden)

- An in-process rate limiter can only police the local process; multi-worker
  deployments share no counters (see rate-limiting design in docs). Cache is
  likewise per-process. Sharded/redis-backed coordination is out of scope.
- DNS allow-decisions are deliberately NOT cached so a hostname can never be
  pinned to a stale IP (DNS-rebinding mitigation, audit B11).
- Applies to: this document is part of the Phase 10 scalability effort; see
  `docs/scalability-audit.md` for the original bottleneck list.