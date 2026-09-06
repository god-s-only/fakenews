# AGENTS.md — Developer Instructions for Fake News Detector

## Project Overview
A FastAPI-based fake news detection API with a promoted scikit-learn
TF-IDF LogisticRegression detector (explainability via influential-feature
contributions), and a single-page frontend. A legacy Keras network is preserved
as a rollback backup.

## Build & Run
```bash
# Install dependencies (requires Python 3.10-3.12)
pip install -r requirements.txt

# Start dev server
python main.py
# or
uvicorn app.main:app --reload

# Open browser at http://localhost:8000
```

## Testing
```bash
# Run all tests (requires pytest + httpx)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_predict.py -v
```

## Key Commands
- `make install` — install runtime dependencies
- `make install-dev` — install runtime + test dependencies
- `make test` — run pytest
- `make fmt` — compile-check all Python files
- `make docker-up` — build and start with Docker Compose
- `make docker-down` — stop Docker Compose

## Environment Variables
All configurable via env or `.env` file (see `.env.example`):
- `PORT` (default 8000)
- `MODEL_PATH` (default `my_model_lr.pkl` — promoted TF-IDF LogisticRegression)
- `VECTORIZER_PATH` (default `my_tfidf_vectorizer.pkl`)
- `UNCERTAINTY_THRESHOLD` (default 0.10)
- `MAX_INPUT_LENGTH` (default 20000)
- `LOG_LEVEL` (default INFO)

Legacy Keras artifacts `my_model.h5`/`countvectorizer.pkl` are preserved as
rollback backups under `artifacts/baseline/` (hashes in
`reports/release_manifest.json` and `reports/baseline_sha256.txt`).

## Architecture
- `app/config.py` — Settings from env vars
- `app/main.py` — FastAPI app, lifespan, routes
- `app/model.py` — ModelService (sklearn LR/TF-IDF or legacy Keras + vectorizer)
- `app/preprocessing.py` — NLTK text cleaning pipeline
- `app/schemas.py` — Pydantic request/response models
- `app/scraper.py` — URL fetching with SSRF protection
- `app/prediction_log.py` — Server-side prediction ring buffer
- `app/logging_config.py` — Logging setup
- `app/verify_model.py` — Standalone model verification script
- `frontend/` — Single-page HTML/CSS/JS frontend
- `tests/` — pytest test suite (100+ tests)

## Model Details
- Promoted production model: scikit-learn LogisticRegression on TF-IDF features
  (sublinear TF, min_df=2, 36,862 terms), fitted ONLY on the training split of a
  cleaned ISOT + BuzzFeed-v02 corpus. P(real) = predict_proba[:,1]; verdict uses
  the 0.10 uncertainty band. Explainability uses linear feature contribution
  (coefficient × TF-IDF weight), a model-appropriate attribution, labelled
  "influential features" (not proof of truth/falsity).
- Legacy Keras model (kept as rollback backup under `artifacts/baseline/`):
  Dense(12,relu)^3 → sigmoid on CountVectorizer BOW; label 1 = REAL;
  P(fake) = 1 − P(real). Its explainability is tf.GradientTape saliency.

## Code Conventions
- Python 3.12, type hints throughout
- Pydantic v2 for validation and serialization
- pytest for testing (no unittest)
- Frontend: vanilla JS, no frameworks
- Never merge to `main`; work only on `feature/detector-overhaul`
