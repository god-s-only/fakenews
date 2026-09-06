"""Phase 5 (source-artifact check) / Phase 6: Reuters-dateline dependence
across ALL candidate models (A, B, C, D, E) using the Phase 3E probe families.

Probes (unchanged from Phase 3E so results are directly comparable):
    Reuters family  : 10 REAL ISOT-test articles with dateline
        original     - raw text incl. "CITY (Reuters) -"
        stripped     - Reuters dateline removed
    Non-Reuters family : 20 REAL generalization corpus articles
        plain          - as published
        reutersstyled  - wrapped in "WASHINGTON (Reuters) - " (content unchanged)

Reported per model: means and label flips. The question answered is whether the
new candidates depend LESS on the Reuters formatting than the production model.

Output: reports/reuters_recheck.json
"""

from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.common as common  # noqa: E402

from app import preprocessing  # noqa: E402
from app.model import ModelService  # noqa: E402

BASELINE = ROOT / "artifacts/baseline"
CANDIDATES = ROOT / "artifacts/candidates"
ISOT_TEST = ROOT / "data/splits/isot_test.csv"
RAW_TRUE = ROOT / "data/True.csv"
GEN_CSV = ROOT / "data/splits/generalization.csv"
REPORT = ROOT / "reports/reuters_recheck.json"

N_REUTERS = 10
N_NONREUTERS = 20


def prep(text: str) -> str:
    return preprocessing.clean_single_text(text)


def build_probes():
    raw_by_title: dict[str, list[str]] = {}
    with open(RAW_TRUE, newline="", encoding="utf-8-sig", errors="replace") as fh:
        for row in csv.DictReader(fh):
            raw_by_title.setdefault(row["title"], []).append(row["text"])

    reuters = []
    seen = set()
    with open(ISOT_TEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["label"] != "1" or row["origin_title"] in seen:
                continue
            found = next((t for t in raw_by_title.get(row["origin_title"], [])
                          if common.is_reuters_dateline(t)), None)
            if found:
                seen.add(row["origin_title"])
                reuters.append({"title": row["origin_title"], "original": found})
            if len(reuters) >= N_REUTERS:
                break

    nonreuters = []
    with open(GEN_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if common.is_reuters_dateline(row["text"]) or len(nonreuters) >= N_NONREUTERS:
                continue
            nonreuters.append({"title": row.get("title") or row.get("feed_title", ""),
                               "plain": row["text"]})
    return reuters, nonreuters


def predictors():
    out = {}
    svc = ModelService(BASELINE / "my_model.h5", BASELINE / "countvectorizer.pkl").load()
    out["A_production"] = lambda t: svc.predict(t).probability_real
    for lbl, fname in (("B_isot_tfidf_lr", "expB_lr.pkl"),
                       ("C_isot_tfidf_svm", "expC_svm.pkl"),
                       ("D_isot_buzzfeed_tfidf_lr", "expD_lr.pkl")):
        with open(CANDIDATES / fname, "rb") as fh:
            blob = pickle.load(fh)
        vec, model = blob["vec"], blob["model"]
        if hasattr(model, "predict_proba"):
            out[lbl] = lambda t, _m=model, _v=vec: float(
                _m.predict_proba(_v.transform([prep(t)]))[0][1])
        else:
            out[lbl] = lambda t, _m=model, _v=vec: float(
                _m.decision_function(_v.transform([prep(t)]))[0])
    import tensorflow as tf
    nn = tf.keras.models.load_model(CANDIDATES / "expE_nn.h5", compile=False)
    with open(CANDIDATES / "expE_vectorizer.pkl", "rb") as fh:
        vec_e = pickle.load(fh)
    out["E_isot_buzzfeed_nn"] = lambda t, _m=nn, _v=vec_e: float(
        _m.predict(_v.transform([prep(t)]).toarray().astype("float32"), verbose=0)[0][0])
    return out


UNCERTAINTY = 0.10


def produce_labels(pred, kind: str):
    """Return labels for each prediction according to model semantics."""
    if kind == "production":
        return ["UNCERTAIN" if abs(p - 0.5) <= UNCERTAINTY else ("REAL" if p > 0.5 else "FAKE")
                for p in pred]
    if kind == "svm":
        return ["REAL" if d > 0.0 else "FAKE" for d in pred]
    return ["REAL" if p > 0.5 else "FAKE" for p in pred]


KINDS = {"A_production": "production", "B_isot_tfidf_lr": "proba",
         "C_isot_tfidf_svm": "svm", "D_isot_buzzfeed_tfidf_lr": "proba",
         "E_isot_buzzfeed_nn": "proba"}


def main() -> None:
    reuters, nonreuters = build_probes()
    preds = predictors()
    results = {}
    for lbl, predict in preds.items():
        kind = KINDS[lbl]
        r_deltas = []
        n_deltas = []
        n_flips = 0
        strip_flips = 0
        items = []
        for it in reuters:
            orig = it["original"]
            stripped = common.strip_reuters_dateline(orig)
            p_orig = float(predict(orig))
            p_strip = float(predict(stripped))
            r_deltas.append(p_strip - p_orig)
            lab_o, lab_s = produce_labels([p_orig, p_strip], kind)
            strip_flips += int(lab_o != lab_s)
            items.append({"family": "reuters", "title": it["title"],
                          "p_original": p_orig, "p_stripped": p_strip,
                          "label_original": lab_o, "label_stripped": lab_s})
        for it in nonreuters:
            plain = it["plain"]
            styled = f"WASHINGTON (Reuters) - {plain}"
            p_plain = float(predict(plain))
            p_styled = float(predict(styled))
            n_deltas.append(p_styled - p_plain)
            lab_p, lab_s = produce_labels([p_plain, p_styled], kind)
            n_flips += int(lab_p != lab_s)
            items.append({"family": "nonreuters", "title": it["title"],
                          "p_plain": p_plain, "p_reutersstyled": p_styled,
                          "label_plain": lab_p, "label_styled": lab_s})
        results[lbl] = {
            "label_semantics": kind + (" (UNCERTAINTY_THRESHOLD=0.10)" if kind == "production" else ""),
            "mean_delta_strip_dateline": float(np.mean(r_deltas)) if r_deltas else None,
            "strip_dateline_label_flips": strip_flips,
            "mean_delta_add_dateline": float(np.mean(n_deltas)) if n_deltas else None,
            "dateline_add_label_flips": n_flips,
            "plain_mean_p_real": float(np.mean([it["p_plain"] for it in items
                                                if it["family"] == "nonreuters"])),
            "styled_mean_p_real": float(np.mean([it["p_reutersstyled"] for it in items
                                                 if it["family"] == "nonreuters"])),
            "items": items,
        }
    common.save_json(REPORT, {
        "note": ("Phase 3E probe families run across all candidate models; "
                 "model A uses production label semantics (0.10 uncertainty band); "
                 "B/C/D/E use standard 0.5 threshold (SVM decision>0)."),
        "models": results,
    })
    print(f"{'model':<26}{'stripΔ':>8}{'stripFlips':>11}{'addΔ':>8}{'addFlips':>9}{'plainMeanP':>12}")
    for lbl, r in results.items():
        print(f"{lbl:<26}{r['mean_delta_strip_dateline']:>8.3f}"
              f"{r['strip_dateline_label_flips']:>11}{r['mean_delta_add_dateline']:>8.3f}"
              f"{r['dateline_add_label_flips']:>9}{r['plain_mean_p_real']:>12.4f}")
    print("\nwrote", REPORT)


if __name__ == "__main__":
    main()