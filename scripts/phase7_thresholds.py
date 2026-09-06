"""Phase 7: uncertainty / decision-threshold analysis (VALIDATION ONLY).

The sweep is run ONLY on validation splits (ISOT val, BuzzFeed val, combined).
No test set is touched here; the fixed production threshold stays 0.5 for the
final numerical claims in the report. Purpose: see whether a threshold other
than 0.5 materially changes metrics on out-of-training data, and to quantify how
the 0.10 uncertainty band (production label semantics) behaves.

Output: reports/threshold_analysis.json
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.common as common  # noqa: E402

from app import preprocessing  # noqa: E402
from app.model import ModelService  # noqa: E402
from scripts.common import metrics_report  # noqa: E402

BASELINE = ROOT / "artifacts/baseline"
CANDIDATES = ROOT / "artifacts/candidates"
ISOT_VAL = ROOT / "data/splits/isot_val.csv"
BF_CLEANED = ROOT / "data/processed/buzzfeed_cleaned.csv"
REPORT = ROOT / "reports/threshold_analysis.json"

THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(17)]  # 0.30..0.70
UNCERTAINTY = 0.10


def prep(text: str) -> str:
    return preprocessing.clean_single_text(text)


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def get_d_predictor():
    with open(CANDIDATES / "expD_lr.pkl", "rb") as fh:
        blob = pickle.load(fh)
    vec, model = blob["vec"], blob["model"]
    return lambda t: float(model.predict_proba(vec.transform([prep(t)]))[0][1])


def get_a_predictor():
    svc = ModelService(BASELINE / "my_model.h5", BASELINE / "countvectorizer.pkl").load()
    return lambda t: svc.predict(t).probability_real


def sweep(predict, texts, ys):
    out = {"n": int(len(texts))}
    probs = np.array([predict(t) for t in texts], dtype=float)
    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)
        acc = (preds == ys).mean()
        reported = metrics_report(ys, probs, threshold=t)
        out[f"{t:.3f}"] = {
            "accuracy": float(acc),
            "macro_f1": reported["macro_f1"],
            "predicted_REAL_pct": reported["predicted_REAL_pct"],
            "predicted_FAKE_pct": reported["predicted_FAKE_pct"],
        }
        if abs(t - 0.5) < 1e-9:
            out["default_0.500"] = out[f"{t:.3f}"]
    # production-style uncertainty band
    band = [("UNCERTAIN" if abs(p - 0.5) <= UNCERTAINTY
             else ("REAL" if p > 0.5 else "FAKE")) for p in probs]
    real_frac = float((ys == 1).mean())
    certain = np.array([l != "UNCERTAIN" for l in band])
    if certain.any():
        out["uncertain_band"] = {
            "uncertain_rate": float((~certain).mean()),
            "real_frac_among_certain": float(ys[certain].mean()) if certain.any() else None,
            "uncertain_rate_given_real": float((~certain[ys == 1]).mean()),
            "uncertain_rate_given_fake": float((~certain[ys == 0]).mean()),
        }
    out["real_frac_overall"] = real_frac
    return out


def main() -> None:
    isot_val = load_csv(ISOT_VAL)
    isot_t = [r["text"] for r in isot_val]
    isot_y = np.array([int(r["label"]) for r in isot_val])

    bf_rows = load_csv(BF_CLEANED)
    bf_t = [r["text"] for r in bf_rows]
    bf_y = np.array([int(r["label"]) for r in bf_rows])

    results = {
        "discipline": "validation splits ONLY; test sets untouched; default remains 0.5",
        "A_production": {
            "isot_val": sweep(get_a_predictor(), isot_t, isot_y),
            "bf_all": sweep(get_a_predictor(), bf_t, bf_y),
        },
        "D_isot_buzzfeed_tfidf_lr": {
            "isot_val": sweep(get_d_predictor(), isot_t, isot_y),
            "bf_all": sweep(get_d_predictor(), bf_t, bf_y),
            "combined_val": sweep(get_d_predictor(), isot_t + bf_t,
                                  np.concatenate([isot_y, bf_y])),
        },
    }
    common.save_json(REPORT, results)

    for model, sets in results.items():
        if model == "discipline":
            continue
        print(f"\n## {model}")
        for sname, sdata in sets.items():
            keys = [f"{t:.3f}" for t in THRESHOLDS]
            accs = {t: round(sdata[t]["accuracy"], 4) for t in keys}
            best = max(accs, key=lambda k: accs[k])
            print(f"  {sname:12} n={sdata['n']:5d} default=0.500 acc={sdata['default_0.500']['accuracy']:.4f} "
                  f"best@{best} acc={accs[best]:.4f}")
    print("\nwrote", REPORT)


if __name__ == "__main__":
    main()