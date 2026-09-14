# ADR-001 — In-process (no shared-store) hardening primitives

- Status: **Accepted** (Phase 10)
- Date: 2026-09-09

## Context

Phase 10 narrows scalability gaps without a fronting infra team. The options
for rate limiting and URL-extraction caching ranged from "in-process only" to
"introduce Redis/Postgres/nats for shared coordination".

## Decision

Keep rate limiting and caching **in-process and per-process**, configurable and
bounded, and document the multi-worker caveat. Specifically:

- Rate limiting: `SlidingWindowRateLimiter` + `RateLimitMiddleware`, per-IP,
  bounded key table, fail-open.
- Caching: `TTLCache` (bounded items + TTL), keyed by URL digest, never body.
- No Redis, no shared counters, no distributed lock in this phase.

## Consequences

- Zero new infrastructure, no new failure domains, deployable with the existing
  Docker Compose stack.
- Multi-worker/replica deployments do NOT enforce a global quota or a shared
  cache unless a gateway or shared store is introduced later.
- The known limits are recorded in `docs/concurrency.md` and
  `docs/deployment.md` so they are a conscious, documented trade-off rather than
  an accident.

## Why not other options

- Shared store: operational complexity and latency for marginal benefit at this
  scale; the dominant cost is per-request model inference, not limiter lookup.
- Redis at this stage would add a stateful dependency to a stateless app.

# ADR-002 — Request-body limit is read-time enforced

- Status: **Accepted** (Phase 10)
- Date: 2026-09-09

## Context

FastAPI/Pydantic will happily buffer an unbounded chunked body before
validation, which is a memory-exhaustion vector.

## Decision

Enforce `MAX_REQUEST_BODY_BYTES` at the ASGI layer (`RequestBodyLimitMiddleware`)
before any route reads the body: fast path on `Content-Length`, slow streaming
path for chunked/undeclared bodies, `413` on exceed, replayed buffer on success.

## Consequences

- Oversized payloads never reach Pydantic and never occupy unbounded memory.
- Small bodies are untouched (a declared, in-limit `Content-Length` passes
  through without buffering).

# ADR-003 — Cache stores extraction results, not predictions

- Status: **Accepted** (Phase 10)

## Context

Two layers could be memoised: fetched+extracted article text, or final
predictions. Caching predictions returns stale verdicts for changed content and
silently breaks the prediction-log/history semantics.

## Decision

Cache only the deterministic HTTP fetch + extraction result (`ExtractResult`),
keyed by a SHA-256 digest of the canonical URL, with TTL. Predictions and their
logging still run on every request.

## Consequences

- Repeat analyses of the same URL skip DNS/fetch/decode/parse while verdicts
  stay fresh with respect to the (mutable) prediction pipeline.
- Extraction cache entries never contain request bodies (only retrieved page
  text + title) and are bounded and expiring.