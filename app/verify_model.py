"""Model startup verification.

Can be run standalone to verify that the model and vectorizer files are
loadable and compatible, independent of the FastAPI server:

    python -m app.verify_model

Works for both supported backends: the promoted scikit-learn TF-IDF +
LogisticRegression detector and the legacy Keras network.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.config import settings
from app.model import ModelService, Prediction


def _load_service(model_path: Path, vectorizer_path: Path) -> ModelService:
    print(f"Loading model from {model_path} ...")
    print(f"Loading vectorizer from {vectorizer_path} ...")
    service = ModelService(model_path, vectorizer_path).load()
    print(f"  Backend        : {service._backend}")
    print(f"  Model type     : {type(service._model).__name__}")
    print(f"  Vectorizer type: {type(service._vectorizer).__name__}")
    print(f"  Vocab size     : {len(service._vectorizer.vocabulary_)}")
    return service


def _run_checks(service: ModelService) -> tuple[bool, Prediction]:
    print("\nRunning test prediction ...")
    pred = service.predict(
        "Breaking news today as government announces new economic policy"
    )
    print(f"  P(real)     : {pred.probability_real:.4f}")
    print(f"  P(fake)     : {pred.probability_fake:.4f}")
    print(f"  Label       : {pred.label}")
    print(f"  Confidence  : {pred.confidence}")
    print(f"  Explanation : {len(pred.explanation)} influential feature(s)")
    ok = (
        0.0 <= pred.probability_real <= 1.0
        and abs(pred.probability_real + pred.probability_fake - 1.0) < 1e-6
        and pred.label in ("real", "fake", "uncertain")
    )
    return ok, pred


def main() -> int:
    model_path = settings.model_file
    vectorizer_path = settings.vectorizer_file

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1
    if not vectorizer_path.exists():
        print(f"ERROR: Vectorizer not found at {vectorizer_path}")
        return 1

    try:
        service = _load_service(model_path, vectorizer_path)
        ok, _pred = _run_checks(service)
        if not ok:
            return 1
        print("\nAll checks passed.")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())