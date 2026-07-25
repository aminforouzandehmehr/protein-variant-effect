"""Cross-validation, held-out evaluation, and run provenance.

Two changes matter here beyond bookkeeping:

1. The test set is carved off **first** and cross-validation runs on the
   remainder. Previously both were computed over the whole array, so the "CV"
   and "held-out" numbers shared rows -- harmless while nothing was tuned, and
   straightforward leakage the moment anything was.
2. Hyper-parameter search happens in an *inner* loop on each training fold, so
   the reported score is a nested-CV estimate rather than the best of several
   grid points measured on the data that chose them.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from .metrics import classification_metrics, regression_metrics, safe_spearman
from .models import build_model, param_grid
from .splits import make_cv_splitter

log = logging.getLogger(__name__)

INNER_FOLDS = 3


def _scoring(task: str, n_classes: int):
    if task == "regression":
        return make_scorer(safe_spearman, greater_is_better=True)
    return "roc_auc" if n_classes == 2 else "accuracy"


def _make_estimator(model_name, seed, task, n_classes, sparse_input, tune, groups, y, folds):
    """Return ``(estimator, needs_groups)`` — a bare pipeline or a grid search."""
    grid = param_grid(model_name, task, tune)
    pipe = build_model(model_name, seed, task, sparse_input)
    if not grid:
        return pipe, False
    inner_cv, _ = make_cv_splitter(task, groups, min(INNER_FOLDS, folds), seed, y)
    search = GridSearchCV(
        pipe, grid, cv=inner_cv, scoring=_scoring(task, n_classes),
        n_jobs=-1, error_score=np.nan, refit=True,
    )
    return search, groups is not None


def _fit(estimator, X, y, groups, needs_groups):
    if needs_groups and groups is not None:
        estimator.fit(X, y, groups=groups)
    else:
        estimator.fit(X, y)
    return estimator


def _fold_score(estimator, X_te, y_te, task, n_classes) -> float:
    if task == "regression":
        return safe_spearman(y_te, estimator.predict(X_te))
    m = classification_metrics(y_te, estimator.predict_proba(X_te), n_classes)
    return m.get("roc_auc", m["accuracy"])


def cross_validate(
    X, y, groups, model_name, seed, task, n_classes, folds=5, tune="auto", sparse_input=False
) -> dict:
    """Nested cross-validation over the training data only."""
    splitter, folds = make_cv_splitter(task, groups, folds, seed, y)
    split_args = (X, y, groups) if groups is not None else (X, y)

    scores, chosen = [], []
    for k, (tr, te) in enumerate(splitter.split(*split_args), 1):
        g_tr = np.asarray(groups)[tr] if groups is not None else None
        est, needs_groups = _make_estimator(
            model_name, seed, task, n_classes, sparse_input, tune, g_tr, y[tr], folds
        )
        _fit(est, X[tr], y[tr], g_tr, needs_groups)
        if isinstance(est, GridSearchCV):
            chosen.append({k_: _jsonable(v) for k_, v in est.best_params_.items()})
        s = _fold_score(est, X[te], y[te], task, n_classes)
        scores.append(s)
        log.info("  fold %d/%d  score = %.3f  (n_train=%d, n_test=%d)", k, folds, s, len(tr), len(te))

    label = ("cv_spearman" if task == "regression"
             else "cv_roc_auc" if n_classes == 2 else "cv_accuracy")
    out = {
        f"{label}_mean": float(np.nanmean(scores)),
        f"{label}_std": float(np.nanstd(scores)),
        "cv_folds": folds,
        "cv_fold_scores": [float(s) for s in scores],
    }
    if chosen:
        out["cv_selected_params"] = chosen
    return out


def fit_and_evaluate(
    X, y, groups, train_idx, test_idx, model_name, seed, task, n_classes,
    tune="auto", sparse_input=False, folds=5,
):
    """Tune and fit on the training split, then score once on the held-out split."""
    g_tr = np.asarray(groups)[train_idx] if groups is not None else None
    est, needs_groups = _make_estimator(
        model_name, seed, task, n_classes, sparse_input, tune, g_tr, y[train_idx], folds
    )
    _fit(est, X[train_idx], y[train_idx], g_tr, needs_groups)

    best = None
    if isinstance(est, GridSearchCV):
        best = {k: _jsonable(v) for k, v in est.best_params_.items()}
        log.info("Selected hyper-parameters: %s", best)

    y_te = y[test_idx]
    if task == "regression":
        y_hat = est.predict(X[test_idx])
        return regression_metrics(y_te, y_hat), y_te, y_hat, best
    proba = est.predict_proba(X[test_idx])
    metrics = classification_metrics(y_te, proba, n_classes)
    score = proba[:, 1] if n_classes == 2 else proba
    return metrics, y_te, score, best


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def environment_info(repo_root: Path | None = None) -> dict:
    """Record what actually produced the numbers.

    A pipeline whose selling point is reproducibility should not leave the
    reader guessing which scikit-learn wrote ``results.json``.
    """
    import importlib

    versions = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "esm"):
        try:
            versions[mod] = getattr(importlib.import_module(mod), "__version__", "unknown")
        except Exception:
            versions[mod] = None

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }

    root = repo_root or Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
    except Exception:
        pass
    return info
