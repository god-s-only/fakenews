"""Phase 8 freeze step: pin Candidate D exactly as trained (Phase 5).

Records:
  * model/vectorizer hyperparameters (introspected from the frozen artifact)
  * preprocessing pipeline used (app.preprocessing.clean_single_text)
  * floating point / decision rule (P(real) -> REAL if > 0.5; verdict band 0.10)
  * dataset composition (ISOT splits + BuzzFeed-v02 splits, seed 42)
  * SHA-256 hashes of every dataset and of the frozen candidate artifact

This step does NOT touch the model. It only writes documentation/report JSONs so
that the promoted artifact can be reproduced and verified deterministically.

Outputs:
  reports/dataset_manifests.json
  reports/candidate_d_frozen_spec.json
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.common as common  # noqa: E402
from scripts.common import save_json  # noqa: E402

CANDIDATE = ROOT / "artifacts/candidates/expD_lr.pkl"
FILES = {
    "isot_cleaned": ROOT / "data/processed/isot_cleaned.csv",
    "isot_train": ROOT / "data/splits/isot_train.csv",
    "isot_val": ROOT / "data/splits/isot_val.csv",
    "isot_test": ROOT / "data/splits/isot_test.csv",
    "buzzfeed_cleaned": ROOT / "data/processed/buzzfeed_cleaned.csv",
    "generalization": ROOT / "data/splits/generalization.csv",
    "buzzfeed_zip": ROOT / "data/buzzfeed_v02/buzzfeed-v02-originalLabels.txt.zip",
    "buzzfeed_txt": ROOT / "data/buzzfeed_v02/buzzfeed-v02-originalLabels.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not CANDIDATE.exists():
        raise SystemExit("Candidate D artifact missing; run phase5 first.")

    manifest = {}
    for name, path in FILES.items():
        if path.exists():
            manifest[name] = {"path": str(path), "sha256": sha256(path)}
        else:
            manifest[name] = {"path": str(path), "sha256": None, "note": "missing"}
    save_json(ROOT / "reports/dataset_manifests.json", manifest)

    with open(CANDIDATE, "rb") as fh:
        blob = pickle.load(fh)
    vec, model = blob["vec"], blob["model"]
    hashes = {"candidateD_artifact": sha256(CANDIDATE)}

    spec = {
        "candidate": "D (ISOT + BuzzFeed-v02, TF-IDF + LogisticRegression)",
        "frozen_from": "artifacts/candidates/expD_lr.pkl (Phase 5, seed 42)",
        "model_kind": "sklearn LogisticRegression (no Keras/TensorFlow)",
        "preprocessing": {"module": "app.preprocessing.clean_single_text",
                          "note": "identical to production preprocessing "
                                  "(lowercase, NLTK tokenize, stopword removal, PorterStemmer)"},
        "vectorizer": {
            "class": type(vec).__name__,
            "params": {k: (str(v)) for k, v in vec.get_params().items()},
            "vocabulary_size": len(vec.vocabulary_),
            "fitted_on": "cleaned ISOT train + cleaned BuzzFeed-v02 train only "
                         "(never val/test/OOD sets)",
        },
        "estimator": {
            "class": type(model).__name__,
            "params": {k: (str(v)) for k, v in model.get_params().items()},
            "classes": [int(c) for c in model.classes_],
            "n_features": int(model.coef_.shape[1]),
            "intercept": float(model.intercept_[0]),
            "C": float(model.C),
            "selected_by": "validation accuracy on combined (ISOT val + BuzzFeed-v02 val)",
        },
        "decision_rule": {
            "probability_real": "sigmoid-equivalent P(real) = predict_proba[:,1]",
            "binary_threshold": 0.5,
            "verdict_uncertainty_band": 0.10,
            "confidence": "winner * 100",
        },
        "dataset_composition": {
            "isot_train_rows": 30775,
            "isot_val_rows": 3828,
            "isot_test_rows": 3906,
            "buzzfeed_train_rows": {"REAL": 358, "FAKE": 19},
            "buzzfeed_val_rows": {"REAL": 342, "FAKE": 20},
            "buzzfeed_test_rows": {"REAL": 346, "FAKE": 21},
            "combined_train_rows": 30775 + 358 + 19,
            "combine_note": "BuzzFeed-v02 rows carried stripped datelines/formatting "
                            "from ISOT cleaning; 46 overlapping articles removed in Phase 4",
        },
        "random_seed": 42,
        "hashes": hashes,
    }
    save_json(ROOT / "reports/candidate_d_frozen_spec.json", spec)
    print("spec hashes:", hashes["candidateD_artifact"])
    print("vocab:", spec["vectorizer"]["vocabulary_size"],
          "| C:", spec["estimator"]["C"],
          "| combined_train:", spec["dataset_composition"]["combined_train_rows"])
    print("wrote reports/dataset_manifests.json + reports/candidate_d_frozen_spec.json")


if __name__ == "__main__":
    main()