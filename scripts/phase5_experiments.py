"""Phase 5: controlled experiments A-E with dataset-aware evaluation.

Experiments:
    A  production model (baseline artifacts) - no training
    B  cleaned ISOT + TF-IDF + Logistic Regression
    C  cleaned ISOT + TF-IDF + Linear SVM
    D  cleaned ISOT + BuzzFeed-v02 + TF-IDF + Logistic Regression
    E  cleaned ISOT + BuzzFeed-v02 + CountVectorizer + improved neural net

Discipline:
    * vectorizer fitted ONLY on the training split of each experiment
    * final eval sets (ISOT test, BuzzFeed test, mixed eval set) are never
      used for hyperparameter selection or threshold tuning (threshold fixed
      0.5; SVM threshold at decision=0)
    * validation split is used ONLY for cheap HP selection / early stopping
    * every experiment is reported per eval set (ISOT test, BuzzFeed test,
      mixed, and the all-REAL generalization corpus)

No probabilities are manipulated. No production artifact is overwritten.

Outputs:
    artifacts/candidates/expB_lr.pkl, expC_svm.pkl, expD_lr.pkl
    artifacts/candidates/expE_nn.h5, expE_vectorizer.pkl
    reports/experiments_results.json
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import preprocessing  # noqa: E402
from app.model import ModelService  # noqa: E402
from scripts.common import read_csv_rows, save_json, metrics_report  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASELINE = ROOT / "artifacts/baseline"
CANDIDATES = ROOT / "artifacts/candidates"
CANDIDATES.mkdir(parents=True, exist_ok=True)

ISOT_TRAIN = ROOT / "data/splits/isot_train.csv"
ISOT_VAL = ROOT / "data/splits/isot_val.csv"
ISOT_TEST = ROOT / "data/splits/isot_test.csv"
BF_CLEANED = ROOT / "data/processed/buzzfeed_cleaned.csv"
GEN_CSV = ROOT / "data/splits/generalization.csv"
REPORT = ROOT / "reports/experiments_results.json"

NN_MAX_FEATURES = 8_000


def prep(text: str) -> str:
    """Production-parity preprocessing (tokenize + stopword removal + stem)."""
    return preprocessing.clean_single_text(text)


# --------------------------------------------------------------------------- #
# BuzzFeed dataset-aware split (group by content_key, stratified, 80/10/10)
# --------------------------------------------------------------------------- #
def buzzfeed_splits():
    rows = read_csv_rows(BF_CLEANED)
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["content_key"], []).append(r)
    groups = {k: by_key[k][0] for k in by_key}
    keys = list(groups)
    counts = Counter(str(groups[k]["label"]) for k in keys)
    rng = random.Random(SEED)
    rng.shuffle(keys)

    targets = {s: {lab: 0.8 * counts[lab] for lab in counts} for s in ("train", "val", "test")}
    load = {s: {lab: 0 for lab in counts} for s in ("train", "val", "test")}

    def deficit(s, lab):
        return targets[s][lab] - load[s][lab]

    split_order = ("train", "val", "test")
    assigned = {s: [] for s in split_order}
    for k in keys:
        lab = str(groups[k]["label"])
        best = max(split_order, key=lambda s: deficit(s, lab))
        assigned[best].append(k)
        load[best][lab] += 1
    splits = {s: [r for k in assigned[s] for r in by_key[k]] for s in split_order}
    groups_done = {s: len(assigned[s]) for s in split_order}
    for s in split_order:
        stats = Counter(int(r["label"]) for r in splits[s])
        if any(v == 0 for v in stats.values()):
            raise RuntimeError(f"BuzzFeed {s} split lacks a class: {stats}")
    return splits, groups_done, {s: Counter(int(r["label"]) for r in splits[s]) for s in split_order}


# --------------------------------------------------------------------------- #
# Metrics / calibration
# --------------------------------------------------------------------------- #
def ece(y_true, probs, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins)."""
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


def generalize_report(probs) -> dict:
    probs = np.asarray(probs).ravel()
    n = len(probs)
    pred_real = int((probs > 0.5).sum())
    pred_fake = n - pred_real
    return {
        "n": int(n),
        "REAL_recall": float(pred_real / n) if n else None,
        "REAL_precision": float(pred_real / n) if n else None,
        "predicted_REAL_pct": float(pred_real / n * 100) if n else None,
        "predicted_FAKE_pct": float(pred_fake / n * 100) if n else None,
        "mean_p_real": float(probs.mean()) if n else None,
        "median_p_real": float(np.median(probs)) if n else None,
        "label_counts": dict(Counter(("REAL" if p > 0.5 else "FAKE") for p in probs)),
        "roc_auc": None,
    }


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def fit_vectorizer(texts, kind, max_features):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    if kind == "count":
        vec = CountVectorizer(min_df=2, max_features=max_features)
    else:
        vec = TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=max_features)
    vec.fit(texts)
    return vec


def train_lr(vec, Xtrain, ytrain, Xval, yval, Cs=(0.1, 1.0, 10.0)):
    from sklearn.linear_model import LogisticRegression
    best_c, best_acc, best_model = Cs[0], -1.0, None
    for c in Cs:
        m = LogisticRegression(C=c, max_iter=3000, solver="liblinear")
        m.fit(Xtrain, ytrain)
        acc = float((m.predict(Xval) == yval).mean())
        if acc > best_acc:
            best_c, best_acc, best_model = c, acc, m
    return best_model, best_c, best_acc


def train_svm(vec, Xtrain, ytrain, Xval, yval, Cs=(0.1, 1.0, 10.0)):
    from sklearn.svm import LinearSVC
    best_c, best_acc, best_model = Cs[0], -1.0, None
    for c in Cs:
        m = LinearSVC(C=c, max_iter=5000, dual="auto")
        m.fit(Xtrain, ytrain)
        acc = float((m.predict(Xval) == yval).mean())
        if acc > best_acc:
            best_c, best_acc, best_model = c, acc, m
    return best_model, best_c, best_acc


def train_nn(Xtrain, ytrain, Xval, yval, input_dim, class_weight=None):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        Xtrain, ytrain, epochs=30, batch_size=256, verbose=0,
        validation_data=(Xval, yval), class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    return model


# --------------------------------------------------------------------------- #
# Eval sets
# --------------------------------------------------------------------------- #
def load_sets():
    def load(path):
        rows = read_csv_rows(path)
        return [r["text"] for r in rows], np.array([int(r["label"]) for r in rows])

    isot_train_t, isot_train_y = load(ISOT_TRAIN)
    isot_val_t, isot_val_y = load(ISOT_VAL)
    isot_test_t, isot_test_y = load(ISOT_TEST)

    bf, bf_groups, bf_stats = buzzfeed_splits()
    bf_train_t = [r["text"] for r in bf["train"]]
    bf_train_y = np.array([int(r["label"]) for r in bf["train"]])
    bf_val_t = [r["text"] for r in bf["val"]]
    bf_val_y = np.array([int(r["label"]) for r in bf["val"]])
    bf_test_t = [r["text"] for r in bf["test"]]
    bf_test_y = np.array([int(r["label"]) for r in bf["test"]])

    gen_rows = read_csv_rows(GEN_CSV)
    gen_t = [r["text"] for r in gen_rows]
    gen_y = np.ones(len(gen_t), dtype=int)

    return dict(
        isot_train=(isot_train_t, isot_train_y), isot_val=(isot_val_t, isot_val_y),
        isot_test=(isot_test_t, isot_test_y),
        bf_train=(bf_train_t, bf_train_y), bf_val=(bf_val_t, bf_val_y),
        bf_test=(bf_test_t, bf_test_y),
        mixed_test=(isot_test_t + bf_test_t, np.concatenate([isot_test_y, bf_test_y])),
        gen=(gen_t, gen_y),
        buzzfeed_split_stats=bf_stats,
    )


def eval_sets_for(model_predict, sets, threshold: float = 0.5,
                  names=("isot_test", "bf_test", "mixed_test", "gen")):
    out = {}
    for name in names:
        texts, y = sets[name]
        probs = np.array([model_predict(t) for t in texts], dtype=float)
        if name == "gen":
            out[name] = generalize_report(probs)
        else:
            m = metrics_report(y, probs, threshold=threshold)
            sample = y if name != "mixed_test" else None
            out[name] = {k: m[k] for k in (
                "n", "accuracy", "macro_f1", "roc_auc",
                "precision_FAKE", "recall_FAKE", "f1_FAKE",
                "precision_REAL", "recall_REAL", "f1_REAL",
                "confusion_matrix", "predicted_REAL_pct", "predicted_FAKE_pct")}
            out[name]["ece"] = ece(y, probs)
    return out


def main() -> None:
    import sklearn.metrics

    sets = load_sets()
    isot_train_t, isot_train_y = sets["isot_train"]
    isot_val_t, isot_val_y = sets["isot_val"]
    bf_train_t, bf_train_y = sets["bf_train"]
    bf_val_t, bf_val_y = sets["bf_val"]

    c_isot_train = [prep(t) for t in isot_train_t]
    c_isot_val = [prep(t) for t in isot_val_t]
    c_bf_train = [prep(t) for t in bf_train_t]
    c_bf_val = [prep(t) for t in bf_val_t]
    c_comb_train = c_isot_train + c_bf_train
    y_comb_train = np.concatenate([isot_train_y, bf_train_y])
    c_comb_val = c_isot_val + c_bf_val
    y_comb_val = np.concatenate([isot_val_y, bf_val_y])

    results = {}

    # ---------------- A: production baseline ----------------
    svc = ModelService(BASELINE / "my_model.h5", BASELINE / "countvectorizer.pkl").load()
    results["A_production"] = eval_sets_for(lambda t: svc.predict(t).probability_real, sets)
    results["A_production"]["model"] = "artifacts/baseline/my_model.h5 (untouched)"

    # ---------------- B: ISOT TF-IDF + LR ----------------
    vec_b = fit_vectorizer(c_isot_train, "tfidf", 80_000)
    Xb_tr, Xb_va = vec_b.transform(c_isot_train), vec_b.transform(c_isot_val)
    lr_b, c_b, acc_b = train_lr(vec_b, Xb_tr, isot_train_y, Xb_va, isot_val_y)

    def _lr_predict(t, _m=lr_b, _v=vec_b):
        return float(_m.predict_proba(_v.transform([prep(t)]))[0][1])

    results["B_isot_tfidf_lr"] = eval_sets_for(_lr_predict, sets)
    results["B_isot_tfidf_lr"].update({"model": "LR", "C": c_b, "val_acc": acc_b,
                                       "vectorizer_fit": "ISOT train only"})

    # ---------------- C: ISOT TF-IDF + LinearSVC ----------------
    svm_c, c_c, acc_c = train_svm(vec_b, Xb_tr, isot_train_y, Xb_va, isot_val_y)

    def _svm_predict(t, _m=svm_c, _v=vec_b):
        return float(_m.decision_function(_v.transform([prep(t)]))[0])

    m_c = eval_sets_for(_svm_predict, sets, threshold=0.0)
    for name in ("isot_test", "bf_test", "mixed_test"):
        y = sets[name][1]
        d = np.array([_svm_predict(t) for t in sets[name][0]])
        m_c[name]["roc_auc"] = float(sklearn.metrics.roc_auc_score(y, d))
        m_c[name]["ece"] = None
    m_c["gen"]["note"] = "metrics via decision>0; P(real) unavailable for SVC"
    results["C_isot_tfidf_svm"] = m_c
    results["C_isot_tfidf_svm"].update({"model": "LinearSVC", "C": c_c, "val_acc": acc_c,
                                        "decision_threshold": 0})

    # ---------------- D: ISOT+BuzzFeed TF-IDF + LR ----------------
    vec_d = fit_vectorizer(c_comb_train, "tfidf", 80_000)
    Xd_tr, Xd_va = vec_d.transform(c_comb_train), vec_d.transform(c_comb_val)
    lr_d, c_d, acc_d = train_lr(vec_d, Xd_tr, y_comb_train, Xd_va, y_comb_val)

    def _lr_d_predict(t, _m=lr_d, _v=vec_d):
        return float(_m.predict_proba(_v.transform([prep(t)]))[0][1])

    results["D_isot_buzzfeed_tfidf_lr"] = eval_sets_for(_lr_d_predict, sets)
    results["D_isot_buzzfeed_tfidf_lr"].update({"model": "LR", "C": c_d, "val_acc": acc_d,
                                                "vectorizer_fit": "combined train only"})

    # ---------------- E: ISOT+BuzzFeed CountVectorizer + improved NN ----------------
    vec_e = fit_vectorizer(c_comb_train, "count", NN_MAX_FEATURES)
    Xe_tr = vec_e.transform(c_comb_train).toarray().astype("float32")
    Xe_va = vec_e.transform(c_comb_val).toarray().astype("float32")
    counts = Counter(int(v) for v in y_comb_train)
    class_weight = {0: len(y_comb_train) / (2 * counts[0]),
                    1: len(y_comb_train) / (2 * counts[1])}
    nn_e = train_nn(Xe_tr, y_comb_train, Xe_va, y_comb_val, Xe_tr.shape[1],
                    class_weight=class_weight)

    def _nn_predict(t, _m=nn_e, _v=vec_e):
        x = _v.transform([prep(t)]).toarray().astype("float32")
        return float(_m.predict(x, verbose=0)[0][0])

    results["E_isot_buzzfeed_nn"] = eval_sets_for(_nn_predict, sets)
    results["E_isot_buzzfeed_nn"].update(
        {"model": "improved Dense MLP (128-64 relu, dropout 0.3)",
         "class_weight": class_weight, "nn_max_features": NN_MAX_FEATURES})
    nn_e.save(str(CANDIDATES / "expE_nn.h5"))
    with open(CANDIDATES / "expE_vectorizer.pkl", "wb") as fh:
        pickle.dump(vec_e, fh)

    # persist sklearn artifacts
    for name, obj in (("expB_lr.pkl", {"vec": vec_b, "model": lr_b}),
                      ("expC_svm.pkl", {"vec": vec_b, "model": svm_c}),
                      ("expD_lr.pkl", {"vec": vec_d, "model": lr_d})):
        with open(CANDIDATES / name, "wb") as fh:
            pickle.dump(obj, fh)

    summary = {
        "seed": SEED,
        "eval_sets_note": "final eval sets never used for training/hp/threshold tuning",
        "buzzfeed_split_stats": sets["buzzfeed_split_stats"],
        "model_artifacts": [p.name for p in CANDIDATES.iterdir()],
    }
    save_json(REPORT, {"summary": summary, "experiments": results})

    order = ["A_production", "B_isot_tfidf_lr", "C_isot_tfidf_svm",
             "D_isot_buzzfeed_tfidf_lr", "E_isot_buzzfeed_nn"]
    print(f"{'exp':<26}{'ISOT_ACC':>9}{'BF_ACC':>8}{'MIX_ACC':>9}{'GEN_REALrec':>12}{'GEN_meanP':>10}")
    for k in order:
        r = results[k]
        print(f"{k:<26}"
              f"{r['isot_test']['accuracy']:>9.4f}{r['bf_test']['accuracy']:>8.4f}"
              f"{r['mixed_test']['accuracy']:>9.4f}"
              f"{r['gen']['REAL_recall']:>12.4f}{r['gen']['mean_p_real']:>10.4f}")
    print()
    for k in order:
        g = results[k]["gen"]
        print(f"{k}: GEN label_counts={g['label_counts']} | "
              f"REAL recall={g['REAL_recall']:0.3f} | mean P(real)={g['mean_p_real']:0.4f}")
    print("\nwrote", REPORT)


if __name__ == "__main__":
    main()