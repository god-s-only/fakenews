# Documentation index

- **`scalability-audit.md`** — Phase 10 bottleneck findings (B1–B11) that drove
  the hardening work.
- **`architecture.md`** — system architecture and runtime topology.
- **`model-lifecycle.md`** — model/artifact lifecycle, fingerprints, readiness.
- **`concurrency.md`** — threading model, blocking-op audit, thread-safety
  guarantees, per-process limits.
- **`deployment.md`** — operating the service in production (process/Docker),
  endpoints, observability, multi-worker caveats.
- **`adr/0001-hardening-primitives.md`** — accepted design decisions for the
  in-process rate limiter/cache, body-size enforcement, and cache semantics.
- **`scripts/`** — assistance tooling used during development/verification.