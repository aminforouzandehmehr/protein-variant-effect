"""Train/test and cross-validation splitting strategies.

A random split over the variants of one assay measures *interpolation*: for a
held-out variant at position 128, the training set almost certainly contains
other substitutions at position 128, so the model can memorise how tolerant that
site is. That is not the question anyone actually has about a variant-effect
predictor.

``position`` holds out whole positions, and ``protein`` holds out whole
proteins/assays, so the reported number is extrapolation to sites (or targets)
the model has never seen.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

log = logging.getLogger(__name__)

SPLIT_STRATEGIES = ("random", "position", "protein")


def make_groups(df: pd.DataFrame, parsed, strategy: str):
    """Return the grouping key per row, or ``None`` for a random split."""
    if strategy == "random":
        return None

    if strategy == "position":
        groups = np.array(
            [",".join(str(p) for _, p, _ in sorted(subs, key=lambda s: s[1])) if subs else ""
             for subs in parsed]
        )
    elif strategy == "protein":
        if "group" in df.columns:
            groups = df["group"].astype(str).to_numpy()
        else:
            log.warning(
                "No --group-col given; grouping by sequence length as a proxy for "
                "protein identity."
            )
            groups = df["sequence"].astype(str).str.len().astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'; choose from {SPLIT_STRATEGIES}.")

    n_groups = len(np.unique(groups))
    log.info("Split strategy '%s': %d distinct group(s) over %d variants.",
             strategy, n_groups, len(groups))
    if n_groups < 2:
        raise ValueError(
            f"Split strategy '{strategy}' produced {n_groups} group(s), so no split is "
            f"possible. A 'protein' split needs a CSV covering several proteins "
            f"(and --group-col naming the column that identifies them)."
        )
    return groups


def _effective_folds(folds: int, y, groups) -> int:
    """Shrink the fold count when the data cannot support it, rather than crash."""
    limit = len(np.unique(groups)) if groups is not None else len(y)
    if limit < folds:
        log.warning("Only %d splittable unit(s); reducing folds %d -> %d.", limit, folds, limit)
        return max(2, limit)
    return folds


def make_cv_splitter(task: str, groups, folds: int, seed: int, y=None):
    """Build the CV splitter matching the task and grouping."""
    folds = _effective_folds(folds, y if y is not None else [], groups)
    if groups is None:
        return (StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                if task == "classification"
                else KFold(n_splits=folds, shuffle=True, random_state=seed)), folds
    if task == "classification":
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed), folds
    # GroupKFold gained `shuffle` only in scikit-learn 1.6; it is deterministic
    # without it, which is fine here -- the grouping already fixes the folds.
    return GroupKFold(n_splits=folds), folds


def make_test_split(y, groups, task: str, test_size: float, seed: int):
    """Carve off the held-out test set *before* any cross-validation.

    Returns ``(train_idx, test_idx)``. Grouping is respected, so a position (or
    protein) never appears on both sides; classification splits stay stratified
    wherever the splitter allows it.
    """
    idx = np.arange(len(y))
    if groups is None:
        strat = y if task == "classification" else None
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)

    n_splits = max(2, int(round(1.0 / test_size)))
    n_splits = _effective_folds(n_splits, y, groups)
    splitter_cls = StratifiedGroupKFold if task == "classification" else GroupKFold
    splitter = (splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed)
                if splitter_cls is StratifiedGroupKFold else splitter_cls(n_splits=n_splits))
    train_idx, test_idx = next(iter(splitter.split(idx, y, groups)))
    log.info("Held-out test: %d variants over %d group(s), disjoint from the %d training variants.",
             len(test_idx), len(np.unique(np.asarray(groups)[test_idx])), len(train_idx))
    return train_idx, test_idx
