"""Point metrics for binary P(dropback) predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_recall_fscore_support)


def all_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p >= threshold).astype(int)
    prec, rec, _, support = precision_recall_fscore_support(y, yhat, labels=[0, 1],
                                                            zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "dropback_rate": float(np.mean(y)),
        "accuracy": float(np.mean(yhat == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "precision_run": float(prec[0]), "recall_run": float(rec[0]),
        "precision_dropback": float(prec[1]), "recall_dropback": float(rec[1]),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
