"""Shared helpers for the dataset/model improvement pipeline (Phases 1-10).

Run from the repository root with::

    .venv/bin/python scripts/<script>.py

This module deliberately uses only the Python standard library, NumPy,
scikit-learn and TensforFlow already required by the application, so no extra
dependencies are introduced.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = REPO / "artifacts"
REPORTS_DIR = REPO / "reports"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"

LABELS = ("FAKE", "REAL")
POS = 1  # REAL is label 1


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts (stdlib only)."""
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def write_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["text", "label", "source", "dataset"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Reuters dateline artifact
# --------------------------------------------------------------------------- #

# Matches the dateline that prefixes the overwhelming majority of ISOT REAL
# articles, e.g. "WASHINGTON (Reuters) - ", "SEATTLE/WASHINGTON (Reuters) - "
# or "LONDON (Reuters) - The ...". Captured loosely so CITY may contain
# letters, spaces, slashes, dots and apostrophes.
_DATELINE_RE = re.compile(r"^\s*[A-Z][A-Z0-9 /,.'\-]+\(Reuters\)\s*-?\s*")


def is_reuters_dateline(text: str) -> bool:
    """True if the text starts with a ``CITY (Reuters) -`` dateline."""
    return bool(_DATELINE_RE.match(text or ""))


def strip_reuters_dateline(text: str) -> str:
    """Remove a leading ``CITY (Reuters) -`` dateline if present."""
    return _DATELINE_RE.sub("", text or "", count=1).strip()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the full metric set for binary label 1=REAL / 0=FAKE.

    ``probs`` must be P(real). Prediction = prob > threshold.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel().astype(float)
    y_pred = (probs > threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0, sample_weight=sample_weight
    )
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    try:
        if np.unique(y_true).size < 2:
            auc = None
        else:
            auc = float(roc_auc_score(y_true, probs))
    except Exception:  # noqa: BLE001 - ROC undefined for single-class folds
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    n = len(y_true)
    return {
        "n": int(n),
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1)),
        "roc_auc": auc,
        "precision_FAKE": float(prec[0]),
        "recall_FAKE": float(rec[0]),
        "f1_FAKE": float(f1[0]),
        "precision_REAL": float(prec[1]),
        "recall_REAL": float(rec[1]),
        "f1_REAL": float(f1[1]),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]] with labels [FAKE, REAL]
        "actual_REAL_pct": float(y_true.mean() * 100.0),
        "actual_FAKE_pct": float((1.0 - y_true.mean()) * 100.0),
        "predicted_REAL_pct": float(y_pred.mean() * 100.0),
        "predicted_FAKE_pct": float((1.0 - y_pred.mean()) * 100.0),
        "threshold": float(threshold),
        "n_real": int((y_true == 1).sum()),
        "n_fake": int((y_true == 0).sum()),
    }


def evaluate_split(
    model: Any,
    vectorizer: Any,
    texts: list[str],
    y_true: np.ndarray,
    preprocessing_fn: Any = None,
) -> dict[str, Any]:
    """Predict and evaluate on a list of raw texts using (model, vectorizer).

    ``preprocessing_fn`` maps raw text -> cleaned text (vectorizer input).
    If None, raw text is fed straight into the vectorizer.
    """
    y_true = np.asarray(y_true).ravel()
    probs: list[float] = []
    vectors = vectorizer.transform([preprocessing_fn(t) if preprocessing_fn else t for t in texts])
    preds = model.predict(vectors, batch_size=512, verbose=0)
    if preds.ndim > 1:
        preds = preds[:, 0]
    probs = np.clip(preds.ravel().astype(float), 0.0, 1.0)
    report = metrics_report(y_true, probs)
    report["mean_p_real_REAL"] = float(probs[y_true == 1].mean()) if (y_true == 1).any() else None
    report["mean_p_real_FAKE"] = float(probs[y_true == 0].mean()) if (y_true == 0).any() else None
    report["probs"] = probs.tolist()
    return report


def format_confusion(cm: list[list[int]], heading: str = "Confusion (rows=true, cols=pred)") -> str:
    tn, fp = cm[0]
    fn, tp = cm[1]
    lines = [
        heading,
        f"              predicted FAKE   predicted REAL",
        f"true FAKE     {tn:>10} {fp:>14}",
        f"true REAL     {fn:>10} {tp:>14}",
    ]
    return "\n".join(lines)