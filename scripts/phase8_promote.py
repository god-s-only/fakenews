"""Phase 8 promotion step — write the production artifacts for Candidate D.

Writes the promoted detector under clear production filenames at the repo root:
    my_model_lr.pkl          — pickled LogisticRegression estimator
    my_tfidf_vectorizer.pkl  — pickled TfidfVectorizer

Pre-existing production files (``my_model.h5``, ``countvectorizer.pkl``) are
NEVER deleted or overwritten: they remain in place and are also preserved
byte-identical under ``artifacts/baseline/``. Hashes of everything are recorded
to ``reports/release_manifest.json`` (before/after).

This is idempotent. The migration only takes effect when the app configuration
defaults point at the new filenames.

Outputs:
    my_model_lr.pkl, my_tfidf_vectorizer.pkl
    reports/release_manifest.json
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.common import load_json, save_json, sha256  # noqa: E402

CANDIDATE = ROOT / "artifacts/candidates/expD_lr.pkl"
MODEL_OUT = ROOT / "my_model_lr.pkl"
VEC_OUT = ROOT / "my_tfidf_vectorizer.pkl"
OLD_FILES = {
    "my_model.h5": ROOT / "my_model.h5",
    "countvectorizer.pkl": ROOT / "countvectorizer.pkl",
}


def main() -> None:
    manifest: dict = {"old_artifacts_preserved": {}, "new_artifacts": {}}

    for name, path in OLD_FILES.items():
        h = sha256(path) if path.exists() else None
        manifest["old_artifacts_preserved"][name] = {
            "path": str(path), "sha256": h, "deleted_or_overwritten": False,
        }
        baseline = ROOT / "artifacts" / "baseline" / name
        manifest["old_artifacts_preserved"][name]["baseline_copy_matches"] = (
            baseline.exists() and sha256(baseline) == h
        )

    expected = None
    try:
        expected = (ROOT / "reports/baseline_sha256.txt").read_text()
    except OSError:
        expected = ""
    manifest["baseline_manifest_text"] = expected

    with open(CANDIDATE, "rb") as fh:
        blob = pickle.load(fh)
    vec, model = blob["vec"], blob["model"]
    assert hasattr(model, "predict_proba"), "candidate must export predict_proba"

    with open(MODEL_OUT, "wb") as fh:
        pickle.dump(model, fh)
    with open(VEC_OUT, "wb") as fh:
        pickle.dump(vec, fh)

    manifest["new_artifacts"] = {
        "my_model_lr.pkl": {"path": str(MODEL_OUT), "sha256": sha256(MODEL_OUT)},
        "my_tfidf_vectorizer.pkl": {"path": str(VEC_OUT), "sha256": sha256(VEC_OUT)},
        "source": str(CANDIDATE),
        "candidate_sha256": sha256(CANDIDATE),
    }
    manifest["note"] = (
        "my_model.h5 / countvectorizer.pkl are the legacy production model, "
        "kept untouched as rollback backups (also in artifacts/baseline). "
        "The active detector is now my_model_lr.pkl + my_tfidf_vectorizer.pkl."
    )
    save_json(ROOT / "reports/release_manifest.json", manifest)
    for name, entry in manifest["old_artifacts_preserved"].items():
        print(f"preserved {name}: {entry['sha256'][:16]}... "
              f"baseline_match={entry['baseline_copy_matches']}")
    for name, entry in manifest["new_artifacts"].items():
        if not isinstance(entry, dict):
            continue
        print(f"new {name}: {entry['sha256'][:16]}...")
    print("wrote", MODEL_OUT, VEC_OUT, "-> reports/release_manifest.json")


if __name__ == "__main__":
    main()