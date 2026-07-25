"""Estimator pipelines and their hyper-parameter grids.

Every model is a pipeline that starts with a ``VarianceThreshold``. That matters
most for ``onehot_seq``: in a DMS assay nearly every column is constant (only a
few hundred of the ``L x 20`` positions are ever mutated), and dropping those
columns *inside* the pipeline means the decision is refit on each training fold
instead of being made once over the whole dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

MODEL_NAMES = ("ridge", "gbm", "mlp")


def build_model(name: str, seed: int, task: str, sparse_input: bool = False) -> Pipeline:
    """Return the estimator pipeline for ``name`` under ``task``."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}' (choose from {MODEL_NAMES}).")

    steps = [("variancethreshold", VarianceThreshold(threshold=0.0))]
    # Centring densifies a sparse matrix, so it is only enabled for dense input.
    scaler = StandardScaler(with_mean=not sparse_input)

    if task == "regression":
        if name == "ridge":
            steps += [("standardscaler", scaler), ("model", Ridge(alpha=10.0))]
        elif name == "gbm":
            steps += [("model", GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed))]
        else:
            steps += [("standardscaler", scaler), ("model", MLPRegressor(
                hidden_layer_sizes=(128, 32), max_iter=400, early_stopping=True,
                random_state=seed))]
    else:
        if name == "ridge":  # logistic regression is the linear classifier analogue
            steps += [("standardscaler", scaler), ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed))]
        elif name == "gbm":
            steps += [("model", GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed))]
        else:
            steps += [("standardscaler", scaler), ("model", MLPClassifier(
                hidden_layer_sizes=(128, 32), max_iter=400, early_stopping=True,
                random_state=seed))]

    return Pipeline(steps)


def param_grid(name: str, task: str, tune: str) -> dict:
    """Hyper-parameter grid for ``name``, or ``{}`` for no search.

    ``tune='auto'`` searches the linear models only. Their regularisation
    strength genuinely matters (an arbitrary ``alpha=10`` can cost a lot of
    Spearman on one-hot features) and the search is cheap; tuning the tree and
    network models is left opt-in because it is not.
    """
    if tune == "off":
        return {}
    linear_only = tune == "auto"

    if name == "ridge":
        return ({"model__alpha": np.logspace(-2, 4, 7)} if task == "regression"
                else {"model__C": np.logspace(-3, 3, 7)})
    if linear_only:
        return {}
    if name == "gbm":
        return {"model__learning_rate": [0.02, 0.05, 0.1], "model__max_depth": [2, 3]}
    return {"model__alpha": [1e-5, 1e-4, 1e-3, 1e-2]}
