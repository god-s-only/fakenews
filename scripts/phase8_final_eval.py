"""Phase 8 final evaluation — Candidate D, completely untouched.

Loads the frozen Candidate D artifact only (NO re-training, NO tuning, NO
threshold selection against the final sets — decision rule fixed at 0.5 with
the 0.10 verdict band, exactly as frozen). Eval sets: ISOT test, BuzzFeed-v02
test, mixed, OOD generalization corpus, and the Phase 5/6 Reuters-dateline
probe families.

Additional metrics beyond Phase 5: log loss, Brier score, ECE, and the full
per-class breakdowns the regression suite should lock in.

Output: reports/candidate_d_final_eval.json
"""

from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.common as common  # noqa: E402
from app import preprocessing  # noqa: E402
from app.config import settings  # noqa: E402
from scripts.common import metrics_report, save_json  # noqa: E402

CANDIDATE = ROOT / "artifacts/candidates/expD_lr.pkl"
ISOT_TEST = ROOT / "data/splits/isot_test.csv"
BF_CLEANED = ROOT / "data/processed/buzzfeed_cleaned.csv"
GEN_CSV = ROOT / "data/splits/generalization.csv"
RAW_TRUE = ROOT / "data/True.csv"
REPORT = ROOT / "reports/candidate_d_final_eval.json"

UNCERTAINTY = settings.UNCERTAINTY_THRESHOLD


def prep(text: str) -> str:
    return preprocessing.clean_single_text(text)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ece(y_true, probs, n_bins=10):
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(probs[m].mean() - y_true[m].mean())
    return float(total)


def brier(y_true, probs):
    return float(np.mean((np.asarray(probs) - np.asarray(y_true)) ** 2))


def log_loss(y_true, probs):
    from sklearn.metrics import log_loss as sk_log_loss
    clipped = np.clip(np.asarray(probs), 1e-12, 1.0 - 1e-12)
    return float(sk_log_loss(y_true, clipped, labels=[0, 1], sample_weight=None))


def verdict_label(p_real):
    winner = max(p_real, 1.0 - p_real)
    if winner - 0.5 < UNCERTAINTY:
        return "uncertain"
    return "real" if p_real >= 0.5 else "fake"


def label_eval(name, y_true, probs):
    m = metrics_report(np.asarray(y_true), np.asarray(probs), threshold=0.5)
    out = {k: m[k] for k in (
        "n", "accuracy", "macro_f1", "roc_auc",
        "precision_FAKE", "recall_FAKE", "f1_FAKE",
        "precision_REAL", "recall_REAL", "f1_REAL",
        "confusion_matrix", "actual_REAL_pct", "actual_FAKE_pct",
        "predicted_REAL_pct", "predicted_FAKE_pct", "threshold",
        "n_real", "n_fake")}
    out["log_loss"] = log_loss(y_true, probs)
    out["brier"] = brier(y_true, probs)
    out["ece"] = ece(y_true, probs)
    out["mean_p_real_REAL"] = float(np.asarray(probs)[np.asarray(y_true) == 1].mean())
    out["mean_p_real_FAKE"] = float(np.asarray(probs)[np.asarray(y_true) == 0].mean())
    return out


def main() -> None:
    with open(CANDIDATE, "rb") as fh:
        blob = pickle.load(fh)
    vec, model = blob["vec"], blob["model"]

    def predict(text: str) -> float:
        return float(model.predict_proba(vec.transform([prep(text)]))[0][1])

    results: dict = {
        "frozen_candidate": "artifacts/candidates/expD_lr.pkl",
        "decision_rule": "P(real)>0.5 -> REAL; <0.5 -> FAKE; verdict band 0.10",
        "no_tuning_note": "all thresholds/probs fixed; no fitting on these sets",
    }

    # ---- labeled sets ----
    isot = load_csv(ISOT_TEST)
    isot_y = np.array([int(r["label"]) for r in isot])
    isot_p = np.array([predict(r["text"]) for r in isot])
    results["isot_test"] = label_eval("isot_test", isot_y, isot_p)

    bf = load_csv(BF_CLEANED)
    bf_y = np.array([int(r["label"]) for r in bf])
    bf_p = np.array([predict(r["text"]) for r in bf])
    results["buzzfeed_all"] = label_eval("buzzfeed_all", bf_y, bf_p)

    # BuzzFeed-v02 TEST group split (reuse Phase 5 split determinism)
    from scripts.phase5_experiments import buzzfeed_splits
    bf_splits, _, _ = buzzfeed_splits()
    bf_test = bf_splits["test"]
    bf_test_y = np.array([int(r["label"]) for r in bf_test])
    bf_test_p = np.array([predict(r["text"]) for r in bf_test])
    results["buzzfeed_test"] = label_eval("buzzfeed_test", bf_test_y, bf_test_p)

    mixed_y = np.concatenate([isot_y, bf_test_y])
    mixed_p = np.concatenate([isot_p, bf_test_p])
    results["mixed_test"] = label_eval("mixed_test", mixed_y, mixed_p)

    # ---- OOD generalization (all REAL) ----
    gen = load_csv(GEN_CSV)
    gen_p = np.array([predict(r["text"]) for r in gen])
    n = len(gen_p)
    pred_real = int((gen_p > 0.5).sum())
    results["ood_generalization"] = {
        "n": n,
        "REAL_recall": float(pred_real / n),
        "REAL_precision": float(pred_real / n),
        "REAL_f1": float(pred_real / n),
        "predicted_REAL_pct": float(pred_real / n * 100.0),
        "predicted_FAKE_pct": float((n - pred_real) / n * 100.0),
        "label_counts": dict(Counter("REAL" if p > 0.5 else "FAKE" for p in gen_p)),
        "mean_p_real": float(gen_p.mean()),
        "median_p_real": float(np.median(gen_p)),
        "min_p_real": float(gen_p.min()),
        "max_p_real": float(gen_p.max()),
        "outlets": sorted({r["source"] for r in gen}),
    }

    # ---- Reuters-dateline robustness (Phase 3E/6 families) ----
    raw_by_title: dict[str, list[str]] = defaultdict(list)
    with open(RAW_TRUE, newline="", encoding="utf-8-sig", errors="replace") as fh:
        for row in csv.DictReader(fh):
            raw_by_title[row["title"]].append(row["text"])

    reuters, nonreuters = [], []
    seen = set()
    for row in isot:
        if row["label"] != "1" or row["origin_title"] in seen:
            continue
        found = next((t for t in raw_by_title.get(row["origin_title"], [])
                      if common.is_reuters_dateline(t)), None)
        if found:
            seen.add(row["origin_title"])
            reuters.append(found)
        if len(reuters) >= 10:
            break
    nonreuters = [r["text"] for r in gen if not common.is_reuters_dateline(r["text"])][:20]

    strip_d, add_d, strip_flips, add_flips = [], [], 0, 0
    strip_flips_bin, add_flips_bin = 0, 0
    for raw in reuters:
        p = predict(raw)
        ps = predict(common.strip_reuters_dateline(raw))
        strip_d.append(ps - p)
        strip_flips += int(verdict_label(p) != verdict_label(ps))
        strip_flips_bin += int((p > 0.5) != (ps > 0.5))
    for plain in nonreuters:
        styled = f"WASHINGTON (Reuters) - {plain}"
        p = predict(plain)
        ps = predict(styled)
        add_d.append(ps - p)
        add_flips += int(verdict_label(p) != verdict_label(ps))
        add_flips_bin += int((p > 0.5) != (ps > 0.5))

    results["reuters_robustness"] = {
        "reuters_family_n": len(reuters),
        "mean_delta_strip_dateline": float(np.mean(strip_d)) if strip_d else None,
        "strip_verdict_flips": strip_flips,
        "strip_label_flips_binary": strip_flips_bin,
        "nonreuters_family_n": len(nonreuters),
        "mean_delta_add_dateline": float(np.mean(add_d)) if add_d else None,
        "add_verdict_flips": add_flips,
        "add_label_flips_binary": add_flips_bin,
        "plain_mean_p_real": float(np.mean([predict(t) for t in nonreuters])),
        "styled_mean_p_real": float(np.mean([predict(f"WASHINGTON (Reuters) - {t}") for t in nonreuters])),
    }

    save_json(REPORT, results)

    print("=== Candidate D — final (frozen) evaluation ===")
    for key in ("isot_test", "buzzfeed_test", "buzzfeed_all", "mixed_test"):
        r = results[key]
        print(f"{key:14} n={r['n']:5d} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
              f"F1_REAL={r['f1_REAL']:.4f} F1_FAKE={r['f1_FAKE']:.4f} ROC={r['roc_auc'] and round(r['roc_auc'],4)} "
              f"logloss={r['log_loss']:.4f} brier={r['brier']:.4f} ece={r['ece']:.4f}")
    g = results["ood_generalization"]
    print(f"OOD gen: recall={g['REAL_recall']:.4f} labels={g['label_counts']} meanP={g['mean_p_real']:.4f}")
    rt = results["reuters_robustness"]
    print(f"Reuters: stripΔ={rt['mean_delta_strip_dateline']:.4f} (verdict flips {rt['strip_verdict_flips']}/10, "
          f"binary {rt['strip_label_flips_binary']}/10) | addΔ={rt['mean_delta_add_dateline']:.4f} "
          f"(verdict flips {rt['add_verdict_flips']}/20, binary {rt['add_label_flips_binary']}/20)")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()