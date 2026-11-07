import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support


def eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def verification_metrics(labels, scores, threshold=None):
    e, t = eer(labels, scores)
    threshold = t if threshold is None else threshold
    p, r, f, _ = precision_recall_fscore_support(labels, scores >= threshold, average="binary",
                                                 zero_division=0)
    return {"EER": e, "AUC": float(roc_auc_score(labels, scores)), "P": float(p), "R": float(r),
            "F1": float(f), "thr": float(threshold)}


def fit_platt(labels, scores):
    lr = LogisticRegression()
    lr.fit(np.asarray(scores).reshape(-1, 1), labels)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def bootstrap_eer(labels, scores, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        vals.append(eer(labels[idx], scores[idx])[0])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
