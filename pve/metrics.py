"""Metrics, with the degenerate cases handled explicitly."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

log = logging.getLogger(__name__)


def safe_spearman(a, b) -> float:
    """Spearman rho, returning NaN instead of raising on a constant input.

    A model that predicts one value for every variant -- which happens when
    heavy regularisation wins a hyper-parameter search on a weak signal -- has
    no rank correlation at all, and that should surface as NaN rather than a
    warning buried in the log.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    res = spearmanr(a, b)
    # `.correlation` is the pre-1.9 SciPy spelling of `.statistic`.
    return float(getattr(res, "statistic", getattr(res, "correlation", np.nan)))


def safe_pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(pearsonr(a, b)[0])


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "spearman": safe_spearman(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def classification_metrics(y_true, proba, n_classes: int) -> dict:
    """Binary metrics treat encoded class 1 as positive.

    ``LabelEncoder`` sorts class names alphabetically, so for ProteinGym's
    Benign/Pathogenic labels class 1 is Pathogenic. The run records the mapping
    in ``results.json`` rather than leaving it implicit -- F1 is not symmetric
    in the choice, so it matters which one is called positive.
    """
    y_true = np.asarray(y_true)
    if n_classes == 2:
        scores = proba[:, 1]
        y_pred = (scores >= 0.5).astype(int)
        both_present = len(np.unique(y_true)) == 2
        return {
            "roc_auc": float(roc_auc_score(y_true, scores)) if both_present else float("nan"),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    y_pred = proba.argmax(1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def score_from_ranking(y_true, scores, task: str, n_classes: int) -> float:
    """Primary metric for a bare score vector (no fitted model).

    Used for the zero-shot baselines, which produce a ranking rather than
    calibrated predictions: Spearman against a continuous target, ROC-AUC
    against a binary one.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    if ok.sum() < 2:
        return float("nan")
    if task == "regression":
        return safe_spearman(y_true[ok], scores[ok])
    if n_classes != 2 or len(np.unique(y_true[ok])) != 2:
        return float("nan")
    return float(roc_auc_score(y_true[ok], scores[ok]))


def primary_metric(task: str, m: dict):
    if task == "regression":
        return "Spearman", m["spearman"]
    return ("ROC-AUC", m["roc_auc"]) if "roc_auc" in m else ("Accuracy", m["accuracy"])
