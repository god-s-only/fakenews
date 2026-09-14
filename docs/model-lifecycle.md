# Model Lifecycle (as-delivered, phase 9 baseline)

This document describes the model lifecycle **as it existed at the start of
Phase 10**, before the hardening work in `app/` changed it. It is the "before"
reference for the lifecycle refactor.

## Ownership

* Configuration: `app/config.py::Settings` (env-driven; `MODEL_PATH`,
  `VECTORIZER_PATH` default to the promoted artifacts).
* Loading + inference: `app/model.py::ModelService`.
* Runtime lifetime: the FastAPI lifespan in `app/main.py::create_app`.
* Live reference: a module-global `state = AppState()` with a single
  `state.model` slot.

## Steps

1. **Import** — `app.main` imports `ModelService`. No model is loaded at
   import time; `app = create_app()` merely wires routes and the lifespan.
2. **Lifespan startup** (`app.main.py::lifespan`):
   * `ensure_stopwords_available()` — pre-downloads the NLTK stopwords set if
     missing (network only on first run; the set is then cached in-memory).
   * `settings.validate()` warnings are logged.
   * `ModelService(model_file, vectorizer_file)` is constructed and `.load()`
     is called.
   * On `ModelLoadError` the loop *re-raises* as `RuntimeError`, so the
     application fails to start (fail-fast) rather than serving 503s from the
     start.
   * On success, the service's fingerprints are computed **externally**:
     `service.model_sha256`, `service.vectorizer_sha256` and `service.vocab_size`
     are hashed/filled by the lifespan code itself using `_sha256_file` and
     poking into `service._vectorizer.vocabulary_`.
3. **Readiness** — the global `state.model` is set; `/health` reports
   `model_loaded`/`vectorizer_loaded` and the fingerprints.
4. **Requests** — `/predict` and `/predict-url` call `_require_model()`, which
   returns `state.model` or raises 503 if it is missing.
5. **Shutdown** — lifespan sets `state.model = None` before the yield
   completes. `ModelService` has no explicit resource release (Keras/TF graphs
   are garbage-collected; sklearn holds no OS resources).

## Observations that motivate the refactor

1. **Fingerprinting is duplicated with the loader** — the SHA-256 hashing
   lives in the lifespan, not in `ModelService.load()`. Creating the service
   in one class and fingerprinting it from another splits a single concern.
2. **Shared module-global `state`** — the lifespan and routes both touch the
   same module-level object. This couples any *new* application instance
   created by `create_app()` to the same state, which is why the current test
   suite avoids entering the lifespan at all.
3. **No explicit readiness model** beyond booleans — `is_loaded` is derived
   from `_model`/`_vectorizer` being non-`None`; there is no notion of "loaded
   but version-mismatched" or a persisted load error for diagnostics.
4. **Fail-fast startup** — a corrupt/missing artifact prevents the process
   from starting at all. Predictable, but it means an operator cannot start
   the API for other endpoints while an artifact is being restored.
5. **No concurrent-loading guard** — `.load()` is not re-entrancy-safe; two
   threads calling it can race on `_model`/`_vectorizer`. In practice it is
   only ever called from startup, but this is an unstated assumption.

## Artifacts and fingerprints (frozen)

| Artifact | SHA-256 (pinned in `reports/release_manifest.json`) |
| --- | --- |
| `my_model_lr.pkl` | `7f555ef4...791e4ec` |
| `my_tfidf_vectorizer.pkl` | `6d504720...1a4f05a` |
| `my_model.h5` (baseline rollback) | `98b9f699...27623344` |
| `countvectorizer.pkl` (baseline rollback) | `039b18b0...a1abb9b` |

The model loads fully in **~36 ms** and performs predictions at a **~45–50 ms
median** (see `reports/baseline/model.json`).