# Fake News Detector

A web application that analyses whether a piece of news text is likely to be
**real** or **fake**, using a pre-trained machine learning model. It supports
both pasted text and article URLs, returns an actionable verdict with
confidence and probability scores, and explains the prediction by surfacing
the words that most influenced the model.

Built with **FastAPI** on the backend and a dependency-free **vanilla
HTML/CSS/JS** frontend.

---

## Features

- **Paste-text analysis** – analyse raw article text.
- **URL analysis** – fetch a page, extract its dominant article (JSON-LD
  `NewsArticle`/`Article` first, semantic HTML fallback), and analyse it
  through the same pipeline as pasted text.  Backed by robust validation
  (HTTP/HTTPS only, SSRF/private-network guard, connect+read timeouts, response
  size cap, redirect limit with per-hop re-validation).  Non-article pages
  (homepages, category/tag listings, author pages, etc.) are rejected with a
  clear, friendly message instead of being silently misclassified.
- **Verdict with confidence** – `real`, `fake` or `uncertain`.
- **Uncertainty handling** – configurable threshold; near 50/50 predictions are
  reported as `uncertain` rather than forced into a confident answer.
- **Explainability** – returns the top words that influenced the prediction and
  whether they pushed toward *real* or *fake*.
- **Prediction history** – stored client-side using `localStorage`.
- **Frontend served by FastAPI** – no separate development server.
- **Docker support** – run the whole application with a single command.
- **API documentation** – automatic OpenAPI docs at `/docs` and `/redoc`.

---

## Architecture overview

```
Pasted text                      URL
     │                            │
     ▼                            ▼
preprocessing               fetch + article extraction
     │                            │
     └──────────►    vectorizer    ◄──────────┘
                            │
                            ▼
                       model (Keras)
                            │
                            ▼
                      prediction
                            │
                            ▼
                      explainability
```

Both the pasted-text and URL endpoints share a single prediction pipeline
(`app/model.py`), so behaviour is consistent regardless of input source.

```
app/
├── __init__.py
├── main.py          # FastAPI app, lifespan model loading, routes
├── config.py        # environment-based configuration
├── model.py         # model/vectorizer loading + shared prediction pipeline
├── preprocessing.py # text cleaning: regex, lowercase, stopwords, stemming
├── explainability   # gradient-based word attribution (in model.py)
├── scraper.py       # safe URL fetching + article extraction
└── schemas.py       # Pydantic request/response models
frontend/
├── index.html
├── style.css
└── script.js
```

---

## Requirements

- Python **3.10 – 3.12** (TensorFlow compatibility)
- pip

> TensorFlow does not yet support Python 3.13+, so use a 3.12 environment.
> The project is verified against **TensorFlow 2.21.0** (pinned in
> `requirements.txt`).  Do not downgrade the pinned version without
> re-verifying model loading and gradient explainability.

---

## Installation

### 1. Create a virtual environment

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\activate
```

Windows (Command Prompt):

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Test dependencies (optional):

```bash
pip install -r requirements-dev.txt
```

### 3. Model / vectorizer files

The application expects two trained artifacts in the project root (or at paths
configured via `MODEL_PATH` and `VECTORIZER_PATH`):

- `my_model_lr.pkl` – the promoted scikit-learn `LogisticRegression` detector.
- `my_tfidf_vectorizer.pkl` – the matching `TfidfVectorizer` fitted on the
  same training corpus.

The legacy `my_model.h5` / `countvectorizer.pkl` (Keras neural network +
CountVectorizer) are preserved as immutable rollback backups under
`artifacts/baseline/` and still load through the same code path (see
`reports/release_manifest.json`). If a model file is missing, the application
fails to start with a clear message rather than silently serving broken
predictions.

---

## Configuration

Configuration is read from environment variables (optionally via a `.env`
file). See `.env.example` for a full template:

| Variable                 | Default         | Description                                        |
| ------------------------ | --------------- | -------------------------------------------------- |
| `HOST`                 | `0.0.0.0`       | Bind host for the server.                          |
| `PORT`                 | `8000`          | Port for the server.                               |
| `LOG_LEVEL`            | `INFO`          | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `MODEL_PATH`             | `my_model_lr.pkl`   | Path to the trained model.                         |
| `VECTORIZER_PATH`        | `my_tfidf_vectorizer.pkl` | Path to the vectorizer.                    |
| `UNCERTAINTY_THRESHOLD`  | `0.10`          | Distance from 50% below which a verdict is uncertain. |
| `MAX_INPUT_LENGTH`       | `20000`         | Maximum characters accepted for pasted text.       |
| `MAX_URL_RESPONSE_SIZE`  | `1000000`       | Maximum bytes accepted from a fetched URL.         |
| `REQUEST_TIMEOUT`        | `10`            | URL read timeout in seconds.                       |
| `CONNECT_TIMEOUT`        | `5`             | URL connection (TCP/TLS) timeout in seconds.       |
| `MAX_REDIRECTS`          | `5`             | Maximum redirects while fetching a URL.            |
| `CORS_ORIGINS`           | `*`             | Comma-separated allowed origins.                   |

Copy the template and adjust as needed:

```bash
cp .env.example .env
```

---

## Running locally

```bash
python main.py
```

Or using the Makefile:

```bash
make run
```

Then open <http://localhost:8000/> in a browser.

---

## Docker

Build and start with a single command:

```bash
docker compose up --build
```

Or with plain Docker:

```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

Then open <http://localhost:8000/>.

---

## API endpoints

| Method | Path          | Description                                  |
| ------ | ------------- | -------------------------------------------- |
| GET    | `/`           | Serves the frontend.                         |
| GET    | `/health`     | Health/readiness check.                      |
| POST   | `/predict`    | Analyse pasted article text.                 |
| POST   | `/predict-url`| Analyse an article at a URL.                 |
| GET    | `/docs`       | Interactive API documentation (Swagger UI).  |
| GET    | `/redoc`      | Alternative API documentation.               |

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### `POST /predict` — example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"news": "A large earthquake struck the region early this morning..."}'
```

### Example response

```json
{
  "label": "real",
  "confidence": 87.42,
  "probability_real": 87.42,
  "probability_fake": 12.58,
  "explanation": {
    "top_influential_words": [
      { "word": "govern", "impact": 1.24, "direction": "real" },
      { "word": "report", "impact": 0.96, "direction": "real" }
    ]
  },
  "source_type": "text"
}
```

Values are real JSON numbers. `label` is one of `real`, `fake`, `uncertain`.

### Validation

Missing, empty, whitespace-only or near-empty `news` returns HTTP `422`:

```json
{
  "detail": "News text must contain enough content to analyze."
}
```

---

## URL analysis

`POST /predict-url` accepts an article URL:

```json
{
  "url": "https://example.com/article"
}
```

**Fetching** — each URL is treated as untrusted input:

- HTTP/HTTPS schemes only; `localhost`/`.local` domains are blocked.
- DNS resolution is checked against private, loopback, link-local and reserved
  ranges before the request is made (SSRF guard). Redirects are followed
  manually, with **each hop re-validated**, so a public URL cannot smuggle the
  client onto a private network. HTTP→HTTPS upgrades (and vice versa) are
  allowed; a maximum of `MAX_REDIRECTS` hops is enforced.
- A realistic browser `User-Agent` is sent, with separate connect
  (`CONNECT_TIMEOUT`) and read (`REQUEST_TIMEOUT`) timeouts, and a response
  size cap (`MAX_URL_RESPONSE_SIZE`) enforced while streaming.
- At `DEBUG` log level the fetcher records every hop, the HTTP status and
  content-type, the final URL, any redirect chain, and the extraction outcome.

**Extraction** — the dominant article is identified in order of preference:

1. **JSON-LD structured data** — `NewsArticle`/`Article` (and related types),
   preferring `articleBody`, with `headline` used as the title.
2. **Semantic HTML fallback** — the most relevant `<article>`/`<main>`/
   `[role="main"]` container (or best-scoring `div`/`section`), taking its
   paragraph text.

Pages that are **not a single news article** — site roots, category/tag list
pages, author/about/contact pages, or content too short to be an article — are
rejected with `This page doesn't appear to contain a single news article.`
A successfully retrieved page whose article content cannot be identified is
reported instead of being silently classified.

**Errors** — failures return HTTP `422` with a user-friendly `detail` and a
stable `category` (e.g. `http_error`, `dns_failure`, `timeout`,
`not_article`, `extraction_failed`, `blocked_network`).  For example:

```json
{
  "detail": "We couldn't find an article at this URL. Check the link and try again.",
  "category": "http_error"
}
```

The technical reason and redirect trace are only logged server-side at
`DEBUG` level.

**Result** — on success the extracted text is run through the exact same
prediction pipeline as pasted text, so manual and URL results are comparable
byte-for-byte for the same text (verified by tests). The response includes
everything `/predict` returns plus an optional `page_title` with the detected
article headline.

> **Extraction ≠ fact-checking.** The URL endpoint extracts article text and
> runs the same model as pasted text. It does **not** verify a page's
> publisher, date, or claims. A low-confidence extraction (blocked scrapers,
> JavaScript-only pages, paywalls) fails with an error rather than guessing.

```bash
curl -X POST http://localhost:8000/predict-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

---

## Explainability

The `explanation.top_influential_words` array lists the words that most
influenced the prediction together with their `impact` and `direction`
(`real` or `fake`). These are **model features/influences**, computed via
gradient attribution — they are **not proof** that a word is real or fake.

---

## Uncertainty handling

The winning probability must exceed 50% by more than `UNCERTAINTY_THRESHOLD`
(in probability, default `0.10`). Otherwise the label is `uncertain`. The raw
probabilities are still returned so callers always see the underlying model
scores.

---

## Testing

```bash
pytest
```

The test suite (currently **144 tests**) covers preprocessing, `/predict`
validation and schema, uncertainty handling, `/health`, URL validation and the
full URL analysis pipeline — JSON-LD and semantic extraction, redirects,
HTTP→HTTPS upgrades, SSRF/private-network guards, error categories and
user-facing messages, non-article rejection, the `/predict-url` response
contract, manual-vs-URL pipeline parity, and real-model integration (parity
with a manual vectorize+predict path, label-from-probability consistency).
External HTTP is mocked — no real network or news site is contacted.

---

## Project structure

```
.
├── app/                  # FastAPI backend package
│   ├── config.py
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── scraper.py
│   └── schemas.py
├── frontend/             # static frontend (no build step)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/                # pytest suite
├── my_model_lr.pkl        # promoted robust LR detector (tracked)
├── my_tfidf_vectorizer.pkl # matching TF-IDF vectorizer (tracked)
├── my_model.h5            # legacy Keras model (preserved backup)
├── countvectorizer.pkl    # legacy CountVectorizer (preserved backup)
├── artifacts/baseline/    # immutable legacy artifacts (tracked)
├── fake_news.ipynb       # original training notebook
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'tensorflow'`** – make sure you are
  using Python 3.10–3.12 and ran `pip install -r requirements.txt`.
- **Model fails to load at startup** – verify the file referenced by
  `MODEL_PATH` (default `my_model_lr.pkl`) matches the vectorizer at
  `VECTORIZER_PATH` (`my_tfidf_vectorizer.pkl`), and that the vectorizer was
  fitted with the same preprocessing as in `fake_news.ipynb`. The legacy
  `my_model.h5` / `countvectorizer.pkl` remain valid via those env vars.
- **Port already in use** – set a different `PORT` in your environment or
  `.env`.
- **Frontend shows "Unable to connect"** – the backend is not running; start it
  with `python main.py`.
- **Docker build is slow** – the TensorFlow wheel is large (~223 MB macOS,
  ~546 MB Linux); this is
  expected on the first build.

---

## Known limitations

- **TensorFlow is a large dependency**.  The wheel is roughly **223 MB on
  macOS** and **~546 MB on Linux** (`python:3.12-slim` / amd64).  On slow or
  metered networks the initial `pip install -r requirements.txt` (or the
  `docker build`) may take a very long time or time out.  Retry with a longer
  timeout (`pip install --timeout 600 --retries 10 …`; the `Dockerfile` sets
  this automatically) or install behind a faster network.
- **TensorFlow is pinned to `==2.21.0`**.  The pin is required for two
  reasons: (1) it is the exact version verified against the trained model and
  explainability pipeline; (2) with a wider version range (e.g.
  `>=2.16,<2.22`) pip's resolver on the Docker `python:3.12-slim` image enters
  "exploding backtracking" and fails with `ResolutionImpossible` over the
  mutually-exclusive `protobuf` constraints of the TensorFlow 2.16–2.21 line.
  Keep the pin; chang only after re-verifying.
- **The existing model is a Keras neural network** with four Dense layers
  (12-relu ×3 → 1-sigmoid) and expects bag-of-words CountVectorizer input
  with NLTK Porter-stemmed tokens.  The vectorizer and model are tightly
  coupled; replacing one without the other produces incorrect results.  The
  vectorizer was pickled with scikit-learn **1.3.2** and loaded with a newer
  scikit-learn, which emits an `InconsistentVersionWarning`; the
  `CountVectorizer` remains fully functional (40,000-feature vocabulary
  verified).
- **The trained model is strongly biased toward `fake`**.  In practice it
  returns near-zero P(real) for almost every input — including plausible
  real-world news articles (verified against the shipped `my_model.h5`).
  Uncertainty verdicts are therefore rarely produced by this particular model,
  even though the threshold logic is correct (covered by unit tests).  This is
  a property of the trained weights, which are preserved as-is; retraining on a
  better-balanced dataset would be required to address it.
- **Legacy `.h5` models loaded through Keras 3 can drop input-gradients**.
  `tf.keras.models.load_model` on a Keras-2-era `.h5` can return
  `None` gradients w.r.t. the input (breaking gradient saliency) while still
  producing correct predictions.  The app detects this at startup and
  reconstructs the identical architecture (same layer config, exact same
  trained weights) so that gradient-based explainability works.  The rebuilt
  model produces bit-for-bit identical predictions to the original weights.
- **Explainability is gradient-based**.  Because the model is a neural
  network (not a linear classifier), feature-importance values are computed
  via `tf.GradientTape` rather than model coefficients.  These are *model
  influences*, not factual proof that a word makes an article real or fake.
- **Prediction history is client-side only** (browser `localStorage`).  It
  is not shared across devices or browsers.
- **URL analysis depends on external sites** being reachable and returning
  parseable HTML.  Some sites block scrapers, serve JavaScript-only content,
  or gate articles behind paywalls; results for those URLs return an
  explicit error instead of a guess.  The detector extracts article text — it
  is **not** a fact-checker and does not assess publisher or claim accuracy.
- **Uncertainty handling** uses a configurable threshold
  (`UNCERTAINTY_THRESHOLD`).  The default value (0.10) may not be appropriate
  for all use-cases.  Tune it for your domain.

### Not yet verified in this environment

- **Docker build/run**.  The Docker configuration itself is valid:
  dependency resolution succeeds on `python:3.12-slim`, and
  `docker compose config` validates.  However, a full `docker build` could
  not complete here because downloading the ~546 MB Linux TensorFlow wheel
  exceeded all practical timeouts on the available network (~0.19 MB/s
  measured).  Run `docker compose up --build` on a faster network to verify
  the container end-to-end.

The application itself — TensorFlow import, model load, vectorizer load, real
predictions, `/health`, `/predict`, `/predict-url`, explainability, frontend
serving, history, and the automated test suite — has been verified against the
real trained model on this machine.
