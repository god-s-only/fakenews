"""Model wrapper around the trained fake-news detector.

Loads the model and vectorizer once at application startup, then exposes a
single ``predict`` entry point used by both the pasted-text and URL endpoints.

Two backends are supported:

* **keras** — a TensorFlow/Keras network (the legacy production model,
  ``Dense(12,relu) x3 -> Dense(1, sigmoid)``) paired with a scikit-learn
  CountVectorizer. The sigmoid output represents P(real); label 1 = REAL,
  0 = FAKE.  Explainability uses gradient-based saliency.
* **sklearn** — a scikit-learn estimator with ``predict_proba`` (the promoted
  robust detector: TF-IDF + LogisticRegression) paired with its vectorizer.
  Explainability uses the linear model's TF-IDF feature contributions
  (``coefficient x term weight``), which is the model-appropriate attribution
  for a logistic model. Gradient-tape saliency does not apply here.

Both backends return the same P(real) scale and the same verdict rule.

Attributions are *model influences* (which words push the prediction toward
real/fake), not factual proof that a word is real or fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app import preprocessing
from app.config import settings


def _has_input_gradients(model: Any) -> bool:
    """Return True if gradients flow from the model output back to its input.

    Some legacy ``.h5`` models loaded through Keras 3 expose weights and
    produce correct predictions, but silently drop gradient tracking to the
    input layer (``tape.gradient`` returns ``None``).  Gradient-based saliency
    needs real input gradients, so detect that condition before relying on it.
    """
    import numpy as np
    import tensorflow as tf

    shape = (1,) + tuple(model.input_shape[1:])
    probe = tf.Variable(np.zeros(shape, dtype=np.float32))
    with tf.GradientTape() as tape:
        output = model(probe, training=False)
    gradient = tape.gradient(output, probe)
    return gradient is not None


def _rebuild_functional_model(model: Any) -> Any:
    """Recreate the identical Dense architecture with a fresh Input layer.

    Copies the exact trained weights from ``model`` into a functionally-built
    network with the same layer configuration (units + activation).  This keeps
    the trained network bit-for-bit identical while restoring gradient flow to
    the input, which is required for gradient-based explainability.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=tuple(model.input_shape[1:]))
    layer_x = inputs
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if not isinstance(layer, tf.keras.layers.Dense):
            # Only the documented Dense architecture can be faithfully rebuilt.
            return model
        config = layer.get_config()
        layer_x = tf.keras.layers.Dense(
            units=config.get("units"),
            activation=config.get("activation"),
        )(layer_x)
    rebuilt = tf.keras.Model(inputs, layer_x)
    rebuilt.set_weights(model.get_weights())
    return rebuilt


class ModelLoadError(Exception):
    """Raised when the model or vectorizer cannot be loaded."""


@dataclass
class ExplanationItem:
    """A single word's contribution to a prediction."""

    word: str
    impact: float
    direction: str


@dataclass
class Prediction:
    """A complete prediction result."""

    probability_real: float
    probability_fake: float
    label: str
    confidence: float
    explanation: list[ExplanationItem] = field(default_factory=list)


class ModelService:
    """Encapsulates the trained model, vectorizer and prediction logic."""

    def __init__(self, model_file: Path, vectorizer_file: Path) -> None:
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self._model: Any | None = None
        self._vectorizer: Any | None = None
        self._backend: str | None = None  # "keras" | "sklearn"
        self._bundle: bool = False

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> "ModelService":
        """Load the model and vectorizer from disk.

        Supports both the legacy Keras/H5 backend and the promoted scikit-learn
        backend (pickled estimator, pickled vectorizer, or a single pickle
        bundling ``{"vec": ..., "model": ...}``).

        Raises :class:`ModelLoadError` with a clear message identifying the
        missing or unloadable file.
        """
        self._load_model_and_vectorizer()

        if self._model is None or self._vectorizer is None:
            raise ModelLoadError("Model or vectorizer failed to initialise.")
        return self

    def _try_load_sklearn(self) -> Any | None:
        """Attempt to load the model file as a pickled scikit-learn estimator.

        Returns the estimator, or None if the file is not a sklearn pickle
        (e.g. a Keras ``.h5`` archive).  A bundle dict ``{"vec": ..., "model": ...}``
        is unwrapped so ``self._vectorizer`` comes from the bundle when present.
        """
        import pickle

        try:
            with self.model_file.open("rb") as handle:
                obj = pickle.load(handle)
        except Exception:  # noqa: BLE001 - .h5 files are not picklable
            return None
        if isinstance(obj, dict) and "model" in obj and "vec" in obj:
            estimator = obj["model"]
            if not hasattr(estimator, "predict_proba"):
                raise ModelLoadError(
                    f"The estimator at {self.model_file} does not expose "
                    "predict_proba; only probability-exporting estimators can "
                    "serve predictions."
                )
            self._vectorizer = obj["vec"]
            self._bundle = True
            return estimator
        if hasattr(obj, "predict_proba"):
            self._bundle = False
            return obj
        return None

    def _load_model_and_vectorizer(self) -> None:
        model_file_missing = not self.model_file.exists()
        if model_file_missing:
            raise ModelLoadError(
                f"Model file not found at {self.model_file}. "
                "Place the trained model there (or set MODEL_PATH)."
            )
        vectorizer = None
        estimator = self._try_load_sklearn()
        if estimator is not None:
            self._backend = "sklearn"
            self._model = estimator
            if self._bundle:
                # Vectorizer lives inside the bundle; the standalone file is
                # optional in that case.
                return
            self._vectorizer = self._load_vectorizer_file(self.vectorizer_file)
            return
        # Not a sklearn pickle -> legacy Keras model.
        self._backend = "keras"
        self._model = self._load_keras_model()
        self._vectorizer = self._load_vectorizer_file(self.vectorizer_file)

    def _load_keras_model(self) -> Any:
        import tensorflow as tf

        try:
            loaded = tf.keras.models.load_model(self.model_file, compile=False)
            if not _has_input_gradients(loaded):
                # Keras 3 can load legacy H5 weights/predictions fine while
                # dropping input-gradients. Rebuild the identical architecture
                # from the loaded weights so saliency actually works.
                loaded = _rebuild_functional_model(loaded)
        except Exception as exc:  # noqa: BLE001 - surface any keras load failure
            raise ModelLoadError(
                f"Failed to load the model from {self.model_file}: {exc}"
            ) from exc
        if not _has_input_gradients(loaded):
            raise ModelLoadError(
                "Gradient-based explainability is unavailable because the "
                f"model at {self.model_file} does not propagate gradients "
                "to its input."
            )
        return loaded

    def _load_vectorizer_file(self, vectorizer_file: Path) -> Any:
        if not vectorizer_file.exists():
            raise ModelLoadError(
                f"Vectorizer file not found at {vectorizer_file}. "
                "Place the trained vectorizer there (or set VECTORIZER_PATH)."
            )
        try:
            import pickle

            with vectorizer_file.open("rb") as handle:
                return pickle.load(handle)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load the vectorizer from {vectorizer_file}: {exc}"
            ) from exc

    def _load_model(self) -> Any | None:
        """Legacy single-purpose loader (Keras), kept for API compatibility."""
        if not self.model_file.exists():
            raise ModelLoadError(
                f"Model file not found at {self.model_file}. "
                "Place the trained model there (or set MODEL_PATH)."
            )
        import tensorflow as tf

        try:
            loaded = tf.keras.models.load_model(self.model_file, compile=False)
            if not _has_input_gradients(loaded):
                loaded = _rebuild_functional_model(loaded)
        except Exception as exc:  # noqa: BLE001 - surface any keras load failure
            raise ModelLoadError(
                f"Failed to load the model from {self.model_file}: {exc}"
            ) from exc
        if not _has_input_gradients(loaded):
            raise ModelLoadError(
                "Gradient-based explainability is unavailable because the "
                f"model at {self.model_file} does not propagate gradients "
                "to its input."
            )
        return loaded

    def _load_vectorizer(self) -> Any | None:
        """Legacy single-purpose vectorizer loader, kept for API compatibility."""
        return self._load_vectorizer_file(self.vectorizer_file)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._vectorizer is not None

    @property
    def model_is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vectorizer_is_loaded(self) -> bool:
        return self._vectorizer is not None

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"ModelService(model={self.model_file.name}, "
            f"vectorizer={self.vectorizer_file.name}, backend={self._backend}, "
            f"status={status})"
        )

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _probability_real(self, cleaned_text: str) -> float:
        """Return the raw probability that the text is real.

        Returns 0.5 (uncertain) if the cleaned text produces an empty vector,
        so the model is never fed degenerate input.
        """
        if not cleaned_text.strip():
            return 0.5
        vector = self._vectorizer.transform([cleaned_text]).toarray()
        if np.count_nonzero(vector) == 0:
            return 0.5
        if self._backend == "sklearn":
            proba = self._model.predict_proba(vector)
            if np.asarray(proba).ndim == 2:
                col = 1 if proba.shape[1] >= 2 else 0
                raw = float(proba[0][col])
            else:
                raw = float(proba[0])
        else:
            raw = float(self._model.predict(vector, verbose=0)[0][0])
        if isinstance(raw, (list, tuple, np.ndarray)):
            raw = float(raw[0])
        return float(np.clip(raw, 0.0, 1.0))

    def predict(self, raw_text: str) -> Prediction:
        """Run the full pipeline for raw text and return a Prediction.

        The Phase 9 source-marker normaliser runs first: unambiguous datelines,
        bylines and publication stamps are stripped so format artefacts cannot
        drive the verdict (documented leak: a ``CITY (Reuters) - `` dateline
        pushed fabricated claims to REAL). The normalizer is narrowly scoped to
        those stamps; article prose (even when it mentions "Reuters",
        "officials" or "ministry") is never consumed.
        """
        cleaned = preprocessing.clean_single_text(
            preprocessing.normalize_news_markers(raw_text) or raw_text
        )
        if not cleaned:
            # Nothing meaningful remained after preprocessing (e.g. all
            # stopwords). There is no signal, so report uncertainty.
            return Prediction(
                probability_real=0.5,
                probability_fake=0.5,
                label="uncertain",
                confidence=50.0,
                explanation=[],
            )
        prob_real = self._probability_real(cleaned)
        prob_fake = 1.0 - prob_real
        label, confidence = self._verdict(prob_real, prob_fake)
        explanation = self._explain(cleaned)
        return Prediction(
            probability_real=prob_real,
            probability_fake=prob_fake,
            label=label,
            confidence=confidence,
            explanation=explanation,
        )

    def _verdict(self, prob_real: float, prob_fake: float) -> tuple[str, float]:
        """Return (label, confidence) based on the uncertainty threshold."""
        winner = max(prob_real, prob_fake)
        if winner - 0.5 < settings.UNCERTAINTY_THRESHOLD:
            return "uncertain", round(winner * 100.0, 2)
        label = "real" if prob_real >= prob_fake else "fake"
        return label, round(winner * 100.0, 2)

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    def _explain(self, cleaned_text: str) -> list[ExplanationItem]:
        """Compute per-word influence using the backend-appropriate method.

        * keras: gradient attribution of P(real) w.r.t. each active input word.
        * sklearn: linear-model feature contribution (coefficient x tf-idf) —
          the natural attribution for a logistic model; NOT a saliency because
          there is no gradient from a neural network.
        """
        if self._backend == "sklearn":
            return self._explain_sklearn(cleaned_text)
        return self._explain_keras_gradient(cleaned_text)

    def _explain_sklearn(self, cleaned_text: str) -> list[ExplanationItem]:
        """TF-IDF feature-contribution explanation for a linear model.

        For each term present in the input, contribution = coef_i * tfidf_i
        (per-unit log-odds push). Impact is reported relative to the strongest
        contributing term (strongest == 100). Direction reflects the sign of
        the contribution.  These are *influential features*, not proof of
        truth or falsity.
        """
        try:
            coefficient = np.asarray(self._model.coef_).ravel()
        except Exception:  # noqa: BLE001 - no linear coefficients available
            return []
        feature_names = self._feature_names()
        matrix = self._vectorizer.transform([cleaned_text])
        contributions: dict[int, float] = {}
        for idx, value in zip(matrix.indices, matrix.data):
            contributions[int(idx)] = float(coefficient[idx] * value)
        if not contributions:
            return []
        max_abs = max(abs(c) for c in contributions.values()) or 1.0
        items: list[ExplanationItem] = []
        for idx, contribution in contributions.items():
            if idx >= len(feature_names):
                continue
            direction = "real" if contribution > 0 else "fake"
            impact = round(abs(contribution) / max_abs * 100.0, 2)
            items.append(ExplanationItem(
                word=feature_names[idx], impact=impact, direction=direction))
        items.sort(key=lambda item: item.impact, reverse=True)
        limit = max(1, min(settings.TOP_FEATURES, 50))
        return items[:limit]

    def _explain_keras_gradient(self, cleaned_text: str) -> list[ExplanationItem]:
        vector = self._vectorizer.transform([cleaned_text]).toarray()
        active_indices = np.flatnonzero(vector[0])
        if len(active_indices) == 0:
            return []

        try:
            import tensorflow as tf

            feature_names = self._feature_names()
            input_tensor = tf.convert_to_tensor(vector.astype("float32"))
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                output = self._model(input_tensor, training=False)
            gradients = tape.gradient(output, input_tensor).numpy()[0]
        except Exception:  # noqa: BLE001 - fall back to no explanation
            return []

        # For each active word, identify the cleaned (stemmed) token.
        words = cleaned_text.split()
        word_to_index: dict[str, int] = {}
        for idx in active_indices:
            for word in words:
                feature = feature_names[idx]
                if feature == word:
                    word_to_index.setdefault(word, idx)
                    break

        contributions: list[ExplanationItem] = []
        for idx in active_indices:
            word = feature_names[idx]
            grad = float(gradients[idx])
            if abs(grad) < 1e-9:
                continue
            direction = "real" if grad > 0 else "fake"
            contributions.append(
                ExplanationItem(
                    word=word,
                    impact=float(round(abs(grad) * 100.0, 2)),
                    direction=direction,
                )
            )

        contributions.sort(key=lambda item: item.impact, reverse=True)
        # Clamp the configured limit to a sensible range to avoid excessive
        # response sizes or degenerate zero/negative configurations.
        limit = max(1, min(settings.TOP_FEATURES, 50))
        return contributions[:limit]

    def _feature_names(self) -> list[str]:
        """Return the vectorizer's feature names (works across sklearn versions)."""
        if hasattr(self._vectorizer, "get_feature_names_out"):
            return list(self._vectorizer.get_feature_names_out())
        return list(self._vectorizer.get_feature_names())
