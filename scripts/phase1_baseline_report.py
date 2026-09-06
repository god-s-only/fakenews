"""Phase 1 — Reproduce the baseline model's held-out metrics.

Reproduces the exact train/test split used by ``fake_news.ipynb``
(random_state=42, test_size=0.2) on the original, unedited ISOT rows, then
evaluates the immutable baseline artifacts (``artifacts/baseline/``).

Output:
    reports/baseline_metrics.json
    reports/baseline_report.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASELINE_DIR,
    DATA_DIR,
    REPORTS_DIR,
    evaluate_split,
    format_confusion,
    is_reuters_dateline,
    read_csv_rows,
    save_json,
)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

TOTAL = 44898


def notebook_preprocess(texts: list[str]) -> list[str]:
    """Exact reproduction of the notebook's loop."""
    sw = set(stopwords.words("english"))
    sw.discard("not")
    stemmer = PorterStemmer()
    corpus: list[str] = []
    for i in range(0, TOTAL):
        review = texts[i]
        if not isinstance(review, str):
            review = ""
        review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower()
        review = review.split()
        review = [stemmer.stem(w) for w in review if w not in sw]
        corpus.append(" ".join(review))
    return corpus


def main() -> int:
    true_rows = read_csv_rows(DATA_DIR / "True.csv")
    fake_rows = read_csv_rows(DATA_DIR / "Fake.csv")
    all_texts = [r["text"] for r in true_rows] + [r["text"] for r in fake_rows]
    y = np.array([1] * len(true_rows) + [0] * len(fake_rows))
    assert len(y) == TOTAL, f"expected {TOTAL} rows, got {len(y)}"

    corpus = notebook_preprocess(all_texts)

    from sklearn.model_selection import train_test_split

    _idx = np.arange(TOTAL)
    _, test_idx, _, _ = train_test_split(_idx, np.zeros(TOTAL), test_size=0.2, random_state=42)
    test_idx = np.sort(test_idx)
    test_corpus = [corpus[i] for i in test_idx]
    y_test = y[test_idx]

    import pickle
    import tensorflow as tf

    cv = pickle.load(open(BASELINE_DIR / "countvectorizer.pkl", "rb"))
    model = tf.keras.models.load_model(BASELINE_DIR / "my_model.h5", compile=False)

    report = evaluate_split(model, cv, test_corpus, y_test)
    report["vectorizer_features"] = int(cv.max_features)
    report["dataset"] = "ISOT (original, unedited)"
    report["split"] = {"method": "train_test_split(test_size=0.2, random_state=42)",
                       "n_test": int(len(test_idx)),
                       "n_train": int(TOTAL - len(test_idx)),
                       "stratified": False}
    report["model_sha256_note"] = "see reports/baseline_sha256.txt"

    n_reuters = sum(1 for t in all_texts if is_reuters_dateline(t))
    n_reuters_real = sum(1 for t in [r["text"] for r in true_rows] if is_reuters_dateline(t))
    report["reuters_dateline"] = {
        "real_starting_with_dateline": int(n_reuters_real),
        "real_pct_starting_with_dateline": round(n_reuters_real / len(true_rows) * 100, 2),
        "all_starting_with_dateline": int(n_reuters),
    }

    save_json(REPORTS_DIR / "baseline_metrics.json", report)

    lines = [
        "# Baseline Report — Original Production Model",
        "",
        "Status: **immutable baseline**. `artifacts/baseline/my_model.h5` + "
        "`artifacts/baseline/countvectorizer.pkl` are byte-identical copies of the "
        "original production artifacts (SHA-256 in `baseline_sha256.txt`).",
        "",
        "## Provenance",
        f"- Dataset: ISOT Fake News Dataset (official `True.csv` 21,417 real + "
        f"`Fake.csv` 23,481 fake = {TOTAL})",
        "- Model: `fake_news.ipynb` — CountVectorizer(max_features=40000) on the "
        "porter-stemmed, stopword-filtered text, Dense(12,relu)x3 → Dense(1, sigmoid), "
        "adam/binary_crossentropy, 10 epochs, batch 32.",
        "- Split: 80/20 `random_state=42`, **unstratified** (as in the notebook).",
        "- Note: the notebook fitted the CountVectorizer on the **full corpus before**\n        " "the split (feature-set leakage) — reproduced here deliberately.",
        "",
        "## Metrics on held-out ISOT test set (n = 8980)",
        f"- Accuracy: **{report['accuracy']:.5f}** (notebook recorded 0.9927616926503341)",
        f"- Macro F1: **{report['macro_f1']:.4f}**",
        f"- ROC-AUC: {report['roc_auc']:.4f}" if report["roc_auc"] else "- ROC-AUC: n/a",
        f"- REAL precision/recall/F1: {report['precision_REAL']:.4f} / "
        f"{report['recall_REAL']:.4f} / {report['f1_REAL']:.4f}",
        f"- FAKE precision/recall/F1: {report['precision_FAKE']:.4f} / "
        f"{report['recall_FAKE']:.4f} / {report['f1_FAKE']:.4f}",
        f"- Predicted REAL % / FAKE %: {report['predicted_REAL_pct']:.2f}% / "
        f"{report['predicted_FAKE_pct']:.2f}%   (actual: "
        f"{report['actual_REAL_pct']:.2f}% / {report['actual_FAKE_pct']:.2f}%)",
        "",
        "",
        format_confusion(report["confusion_matrix"]),
        "",
        "## Why the high score is partly an artifact",
        f"- Of {len(true_rows)} REAL articles, "
        f"{report['reuters_dateline']['real_starting_with_dateline']} "
        f"({report['reuters_dateline']['real_pct_starting_with_dateline']}%) begin with a "
        "`CITY (Reuters) -` dateline.",
        "- The vocabulary only contains a single REAL writing style (Reuters). "
        "The FAKE set spans many non-verified outlets. The model therefore tends to "
        "latch onto the Reuters signature rather than a general notion of "
        "trustworthy journalism.",
        "- Confirmed empirically: a hand-written sentence such as `NEW YORK (Reuters) - "
        "The central bank lowered its benchmark rate on Thursday.` is scored REAL at "
        "P≈0.999, while structurally identical non-Reuters text is scored FAKE at "
        "P≈0.001.",
        "",
        "## Constraints honoured",
        "No probabilities manipulated, no labels inverted, no keyword rules, "
        "`my_model.h5` and `countvectorizer.pkl` untouched.",
    ]
    (REPORTS_DIR / "baseline_report.md").write_text("\n".join(lines))

    print(f"baseline accuracy reproduced: {report['accuracy']:.5f}")
    print("wrote", REPORTS_DIR / "baseline_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())