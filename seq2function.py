#!/usr/bin/env python3
"""
seq2function.py — A compact, reproducible baseline for sequence-to-function
prediction on protein variant-effect data.

Given a set of protein variants and a per-variant label, this trains a supervised
model to predict the label from sequence and reports held-out performance. It
supports two task types and auto-detects which one applies:

  * regression     — continuous functional/fitness scores (e.g. ProteinGym DMS
                     assays). Primary metric: Spearman correlation.
  * classification — categorical labels such as clinical pathogenicity
                     (e.g. ProteinGym clinical / ClinVar: Benign vs Pathogenic).
                     Primary metric: ROC-AUC (binary) or accuracy / macro-F1.

It is intended as a clean, honest baseline and a demonstration of an end-to-end,
leakage-aware modeling pipeline — not a state-of-the-art predictor.

Data format: a CSV with a mutated-sequence column and a label column. Defaults
match ProteinGym (`mutated_sequence`; score column overridable with --score-col).

Examples:
    python seq2function.py --csv DMS_ASSAY.csv --score-col DMS_score
    python seq2function.py --csv CLINICAL.csv --score-col DMS_bin_score --features mutation

Author: M. Amin Forouzandehmehr
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(csv: Path, seq_col: str, score_col: str, mutant_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    missing = [c for c in (seq_col, score_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found in {csv.name}. "
            f"Available: {list(df.columns)}. Use --seq-col / --score-col to override."
        )
    keep = {seq_col: "sequence", score_col: "label"}
    if mutant_col in df.columns:
        keep[mutant_col] = "mutant"
    df = df[list(keep)].rename(columns=keep).dropna(subset=["sequence", "label"])
    df = df[df["sequence"].str.len() > 0].reset_index(drop=True)
    log.info("Loaded %d variants from %s", len(df), csv.name)
    return df


def detect_task(labels: pd.Series, override: str) -> str:
    """Return 'regression' or 'classification'. Auto: non-numeric -> classification;
    numeric with <=2 unique values -> classification; else regression."""
    if override != "auto":
        return override
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.isna().any():
        return "classification"
    if numeric.nunique() <= 2:
        return "classification"
    return "regression"


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #
def featurize_onehot_seq(sequences: pd.Series) -> np.ndarray:
    lengths = sequences.str.len().unique()
    if len(lengths) != 1:
        raise ValueError(
            f"onehot_seq requires equal-length sequences; found lengths {sorted(lengths)}. "
            f"Use --features mutation for variable-length inputs."
        )
    L = int(lengths[0])
    X = np.zeros((len(sequences), L * len(AMINO_ACIDS)), dtype=np.float32)
    for row, seq in enumerate(sequences):
        for pos, aa in enumerate(seq):
            j = AA_INDEX.get(aa)
            if j is not None:
                X[row, pos * len(AMINO_ACIDS) + j] = 1.0
    log.info("one-hot sequence features: %s", X.shape)
    return X


def _parse_single_mutant(mutant: str):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", str(mutant).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def featurize_mutation(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compact features: normalized position + WT one-hot + mutant one-hot.
    Returns (X, mask) where mask selects parseable single substitutions."""
    if "mutant" not in df.columns:
        raise ValueError("--features mutation needs a mutant column (--mutant-col).")
    parsed = df["mutant"].map(_parse_single_mutant)
    mask = parsed.notna().to_numpy()
    if mask.sum() == 0:
        raise ValueError("No single-substitution mutants could be parsed (expected e.g. 'G128R').")
    parsed = parsed[mask]
    positions = np.array([p for _, p, _ in parsed], dtype=np.float32)
    positions = (positions - positions.mean()) / (positions.std() + 1e-8)
    n = len(parsed)
    wt = np.zeros((n, len(AMINO_ACIDS)), dtype=np.float32)
    mt = np.zeros((n, len(AMINO_ACIDS)), dtype=np.float32)
    for i, (w, _, m) in enumerate(parsed):
        if w in AA_INDEX:
            wt[i, AA_INDEX[w]] = 1.0
        if m in AA_INDEX:
            mt[i, AA_INDEX[m]] = 1.0
    X = np.hstack([positions.reshape(-1, 1), wt, mt])
    log.info("mutation-level features: %s (%d/%d variants parseable)", X.shape, n, len(df))
    return X, mask


def featurize_esm(sequences: pd.Series, model_name: str = "esm2_t12_35M_UR50D") -> np.ndarray:
    """Optional: mean-pooled ESM-2 embeddings. Requires `fair-esm` and `torch`.
    On Apple Silicon, set device to 'mps' for GPU acceleration."""
    import torch
    import esm

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("ESM device: %s", device)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    layer = model.num_layers
    feats = []
    for i, seq in enumerate(sequences):
        _, _, toks = bc([(str(i), seq)])
        with torch.no_grad():
            out = model(toks.to(device), repr_layers=[layer])["representations"][layer]
        feats.append(out[0, 1 : len(seq) + 1].mean(0).cpu().numpy())
        if (i + 1) % 200 == 0:
            log.info("  embedded %d/%d", i + 1, len(sequences))
    X = np.vstack(feats).astype(np.float32)
    log.info("ESM features: %s", X.shape)
    return X


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_model(name: str, seed: int, task: str):
    if task == "regression":
        if name == "ridge":
            return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0, random_state=seed))
        if name == "gbm":
            return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
        if name == "mlp":
            return make_pipeline(StandardScaler(with_mean=False),
                                 MLPRegressor(hidden_layer_sizes=(128, 32), max_iter=400,
                                              early_stopping=True, random_state=seed))
    else:  # classification
        if name == "ridge":  # logistic regression is the linear classifier analogue
            return make_pipeline(StandardScaler(with_mean=False),
                                 LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        if name == "gbm":
            return GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
        if name == "mlp":
            return make_pipeline(StandardScaler(with_mean=False),
                                 MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=400,
                                               early_stopping=True, random_state=seed))
    raise ValueError(f"Unknown model '{name}' (choose ridge, gbm, or mlp).")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "spearman": float(spearmanr(y_true, y_pred).correlation),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def classification_metrics(y_true, proba, n_classes) -> dict:
    if n_classes == 2:
        y_pred = (proba[:, 1] >= 0.5).astype(int)
        auroc = float(roc_auc_score(y_true, proba[:, 1])) if len(np.unique(y_true)) == 2 else float("nan")
        return {"roc_auc": auroc, "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred))}
    y_pred = proba.argmax(1)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def primary_metric(task: str, m: dict):
    return ("Spearman", m["spearman"]) if task == "regression" else \
           ("ROC-AUC", m["roc_auc"]) if "roc_auc" in m else ("Accuracy", m["accuracy"])


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def cross_validate(X, y, model_name, seed, task, n_classes, folds=5) -> dict:
    """K-fold CV; scaling/fitting happen inside each training fold only (no leakage).
    Stratified for classification."""
    splitter = (StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed) if task == "classification"
                else KFold(n_splits=folds, shuffle=True, random_state=seed))
    scores = []
    for k, (tr, te) in enumerate(splitter.split(X, y), 1):
        model = build_model(model_name, seed, task)
        model.fit(X[tr], y[tr])
        if task == "regression":
            s = float(spearmanr(y[te], model.predict(X[te])).correlation)
        else:
            proba = model.predict_proba(X[te])
            m = classification_metrics(y[te], proba, n_classes)
            s = m.get("roc_auc", m["accuracy"])
        scores.append(s)
        log.info("  fold %d/%d  score = %.3f", k, folds, s)
    label = "cv_spearman" if task == "regression" else ("cv_roc_auc" if n_classes == 2 else "cv_accuracy")
    return {f"{label}_mean": float(np.nanmean(scores)),
            f"{label}_std": float(np.nanstd(scores)), "cv_folds": folds}


def held_out_eval(X, y, model_name, seed, task, n_classes, test_size=0.2):
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    model = build_model(model_name, seed, task)
    model.fit(Xtr, ytr)
    if task == "regression":
        yhat = model.predict(Xte)
        return regression_metrics(yte, yhat), yte, yhat
    proba = model.predict_proba(Xte)
    return classification_metrics(yte, proba, n_classes), yte, (proba[:, 1] if n_classes == 2 else proba)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def regression_plot(y_true, y_pred, rho, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, edgecolor="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1)
    ax.set_xlabel("Measured score"); ax.set_ylabel("Predicted score")
    ax.set_title(f"Held-out variants (Spearman \u03c1 = {rho:.3f})")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    log.info("Saved plot -> %s", out)


def roc_plot(y_true, scores, auroc, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2); ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Held-out ROC (AUC = {auroc:.3f})")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    log.info("Saved plot -> %s", out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--features", choices=["onehot_seq", "mutation", "esm"], default="onehot_seq")
    p.add_argument("--model", choices=["ridge", "gbm", "mlp"], default="ridge")
    p.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    p.add_argument("--seq-col", default="mutated_sequence")
    p.add_argument("--score-col", default="DMS_score")
    p.add_argument("--mutant-col", default="mutant")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("results"))
    args = p.parse_args()

    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv, args.seq_col, args.score_col, args.mutant_col)
    task = detect_task(df["label"], args.task)
    log.info("Task: %s", task)

    # Features
    if args.features == "onehot_seq":
        X = featurize_onehot_seq(df["sequence"]); row_mask = np.ones(len(df), dtype=bool)
    elif args.features == "mutation":
        X, row_mask = featurize_mutation(df)
    else:
        X = featurize_esm(df["sequence"]); row_mask = np.ones(len(df), dtype=bool)

    raw_labels = df["label"].to_numpy()[row_mask]

    # Targets
    classes = None
    if task == "regression":
        y = pd.to_numeric(pd.Series(raw_labels), errors="coerce").to_numpy(dtype=np.float32)
    else:
        le = LabelEncoder()
        y = le.fit_transform(raw_labels)
        classes = list(le.classes_)
        log.info("Classes: %s  (counts: %s)", classes, np.bincount(y).tolist())
    n_classes = len(classes) if classes is not None else 0

    if len(y) < 2 * args.folds:
        log.warning("Only %d samples — results will be noisy; interpret with caution.", len(y))

    # Cross-validation
    log.info("Cross-validating (%s features, %s model, %s)...", args.features, args.model, task)
    cv = cross_validate(X, y, args.model, args.seed, task, n_classes, args.folds)
    cv_key = next(k for k in cv if k.endswith("_mean"))
    log.info("CV %s = %.3f +/- %.3f", cv_key, cv[cv_key], cv[cv_key.replace("_mean", "_std")])

    # Held-out test
    test_metrics, y_true, y_score = held_out_eval(X, y, args.model, args.seed, task, n_classes, args.test_size)
    name, val = primary_metric(task, test_metrics)
    log.info("Held-out test (%s = %.3f): %s", name, val, json.dumps({k: round(v, 3) for k, v in test_metrics.items()}))

    # Plot
    if task == "regression":
        regression_plot(y_true, y_score, test_metrics["spearman"], args.outdir / "plot.png")
    elif n_classes == 2:
        roc_plot(y_true, y_score, test_metrics["roc_auc"], args.outdir / "plot.png")

    summary = {
        "dataset": args.csv.name, "task": task, "n_variants": int(len(y)),
        "classes": classes, "features": args.features, "model": args.model, "seed": args.seed,
        "cross_validation": cv, "held_out_test": test_metrics,
    }
    (args.outdir / "results.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved summary -> %s", args.outdir / "results.json")


if __name__ == "__main__":
    main()
