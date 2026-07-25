#!/usr/bin/env python3
"""
seq2function.py — A compact, reproducible baseline for sequence-to-function
prediction on protein variant-effect data.

Given a set of protein variants and a per-variant label, this trains a supervised
model to predict the label from sequence and reports held-out performance against
zero-shot baselines. It supports two task types and auto-detects which applies:

  * regression     — continuous functional/fitness scores (e.g. ProteinGym DMS
                     assays). Primary metric: Spearman correlation.
  * classification — categorical labels such as clinical pathogenicity
                     (e.g. ProteinGym clinical / ClinVar: Benign vs Pathogenic).
                     Primary metric: ROC-AUC (binary) or accuracy / macro-F1.

It is intended as a clean, honest baseline and a demonstration of an end-to-end,
leakage-aware modeling pipeline — not a state-of-the-art predictor. Every run
reports the supervised score *next to* training-free baselines (BLOSUM62,
Grantham, optionally ESM-2 masked marginals), so "did the model learn anything?"
has an answer rather than an assumption.

Data format: a CSV with a mutated-sequence column and a label column. Defaults
match ProteinGym (`mutated_sequence`; score column overridable with --score-col).

Examples:
    python seq2function.py --csv DMS_ASSAY.csv --score-col DMS_score
    python seq2function.py --csv DMS_ASSAY.csv --features mutation --split position
    python seq2function.py --csv CLINICAL.csv --score-col DMS_bin_score --features mutation
    python seq2function.py --csv DMS_ASSAY.csv --features esm --zeroshot blosum,grantham,esm

Author: M. Amin Forouzandehmehr
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from pve import __version__
from pve.baselines import ALL_BASELINES, classification_orientation, compute_baselines
from pve.data import (
    coerce_regression_targets,
    detect_task,
    load_data,
    parse_mutant,
    reconstruct_wt_sequence,
)
from pve.esm_features import POOLING_MODES, featurize_esm
from pve.evaluate import cross_validate, environment_info, fit_and_evaluate
from pve.features import featurize_mutation, featurize_onehot_seq
from pve.metrics import primary_metric, score_from_ranking
from pve.models import MODEL_NAMES
from pve.plots import comparison_plot, regression_plot, roc_plot
from pve.splits import SPLIT_STRATEGIES, make_groups, make_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("seq2function")


def _take_rows(X, idx):
    """Row-select from a dense or sparse matrix with an index array."""
    return X[np.asarray(idx)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--features", choices=["onehot_seq", "mutation", "esm"], default="onehot_seq")
    p.add_argument("--model", choices=list(MODEL_NAMES), default="ridge")
    p.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    p.add_argument("--seq-col", default="mutated_sequence")
    p.add_argument("--score-col", default="DMS_score")
    p.add_argument("--mutant-col", default="mutant")

    g = p.add_argument_group("evaluation")
    g.add_argument("--split", choices=list(SPLIT_STRATEGIES), default="random",
                   help="random: interpolation within an assay. position: hold out whole "
                        "positions. protein: hold out whole proteins (needs --group-col).")
    g.add_argument("--group-col", default=None,
                   help="Column identifying the protein/assay, for --split protein.")
    g.add_argument("--tune", choices=["auto", "off", "on"], default="auto",
                   help="Hyper-parameter search inside each training fold. "
                        "auto: linear models only.")
    g.add_argument("--zeroshot", default="blosum,grantham",
                   help=f"Comma-separated training-free baselines from {ALL_BASELINES}, "
                        f"or 'none'. 'esm' needs torch + fair-esm.")
    g.add_argument("--folds", type=int, default=5)
    g.add_argument("--test-size", type=float, default=0.2)
    g.add_argument("--seed", type=int, default=42)

    e = p.add_argument_group("ESM")
    e.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    e.add_argument("--esm-pool", choices=list(POOLING_MODES), default="both",
                   help="both: mutated-position embedding + its contrast with wild type "
                        "(default). mean: whole-sequence mean pooling.")
    e.add_argument("--esm-device", default="auto")
    e.add_argument("--esm-batch", type=int, default=8)
    # Deliberately a str, not a Path: Path("") normalises to Path("."), which would
    # make the documented "pass '' to disable" silently point the cache at the cwd.
    e.add_argument("--esm-cache", type=str, default=".esm_cache/embeddings.npz",
                   help="Pooled-embedding cache; pass '' to disable.")

    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------- #
    df = load_data(args.csv, args.seq_col, args.score_col, args.mutant_col, args.group_col)
    task = detect_task(df["label"], args.task)

    parsed = (df["mutant"].map(parse_mutant) if "mutant" in df.columns
              else pd.Series([None] * len(df)))
    n_parsed = int(parsed.map(bool).sum())
    log.info("Parsed %d/%d mutant strings.", n_parsed, len(df))
    wt_sequence = reconstruct_wt_sequence(df["sequence"], parsed) if n_parsed else None

    # ---- features ----------------------------------------------------------- #
    if args.features == "onehot_seq":
        X, keep, feat_names = featurize_onehot_seq(df["sequence"])
    elif args.features == "mutation":
        X, keep, feat_names = featurize_mutation(df["sequence"], parsed)
    else:
        X, keep, feat_names = featurize_esm(
            df["sequence"], list(parsed), wt_sequence,
            mode=args.esm_pool, model_name=args.esm_model, device=args.esm_device,
            batch_size=args.esm_batch,
            cache_path=Path(args.esm_cache) if args.esm_cache else None,
        )

    rows = np.where(keep)[0]
    df = df.iloc[rows].reset_index(drop=True)
    parsed = parsed.iloc[rows].reset_index(drop=True)

    # ---- targets ------------------------------------------------------------ #
    raw_labels = df["label"].to_numpy()
    classes = None
    if task == "regression":
        y, num_mask = coerce_regression_targets(raw_labels)
        if not num_mask.all():
            X = _take_rows(X, np.where(num_mask)[0])
            df = df.iloc[np.where(num_mask)[0]].reset_index(drop=True)
            parsed = parsed.iloc[np.where(num_mask)[0]].reset_index(drop=True)
    else:
        le = LabelEncoder()
        y = le.fit_transform(raw_labels)
        classes = [c.item() if hasattr(c, "item") else c for c in le.classes_]
        log.info("Classes: %s  (counts: %s)", classes, np.bincount(y).tolist())
        if len(classes) == 2:
            log.info("Binary metrics treat '%s' as the positive class.", classes[1])
    n_classes = len(classes) if classes is not None else 0
    is_sparse = sparse.issparse(X)

    if len(y) < 2 * args.folds:
        log.warning("Only %d samples — results will be noisy; interpret with caution.", len(y))

    # ---- splits ------------------------------------------------------------- #
    groups = make_groups(df, list(parsed), args.split)
    train_idx, test_idx = make_test_split(y, groups, task, args.test_size, args.seed)

    # ---- cross-validation, on the training split only ------------------------ #
    log.info("Cross-validating (%s features, %s model, %s, %s split)...",
             args.features, args.model, task, args.split)
    g_tr = np.asarray(groups)[train_idx] if groups is not None else None
    cv = cross_validate(
        _take_rows(X, train_idx), y[train_idx], g_tr, args.model, args.seed, task,
        n_classes, args.folds, args.tune, is_sparse,
    )
    cv_key = next(k for k in cv if k.endswith("_mean"))
    log.info("CV %s = %.3f +/- %.3f", cv_key, cv[cv_key], cv[cv_key.replace("_mean", "_std")])

    # ---- held-out test ------------------------------------------------------- #
    test_metrics, y_true, y_score, best_params = fit_and_evaluate(
        X, y, groups, train_idx, test_idx, args.model, args.seed, task, n_classes,
        args.tune, is_sparse, args.folds,
    )
    metric_name, metric_val = primary_metric(task, test_metrics)
    log.info("Held-out test (%s = %.3f): %s", metric_name, metric_val,
             json.dumps({k: round(v, 3) for k, v in test_metrics.items()}))

    # ---- zero-shot baselines, scored on the same held-out rows --------------- #
    wanted = [b for b in (args.zeroshot or "").split(",") if b.strip() and b.strip() != "none"]
    baseline_scores = compute_baselines(
        [b.strip() for b in wanted], list(parsed), wt_sequence,
        esm_kwargs={"model_name": args.esm_model, "device": args.esm_device,
                    "batch_size": args.esm_batch},
    )
    sign, orientation = (classification_orientation(classes) if task == "classification"
                         else (1, "higher score = higher measured fitness"))
    baselines = {}
    for name, scores in baseline_scores.items():
        baselines[name] = score_from_ranking(
            y[test_idx], sign * np.asarray(scores)[test_idx], task, n_classes
        )
        log.info("Zero-shot %-9s %s = %.3f", name, metric_name, baselines[name])
    if baselines:
        baselines["_orientation"] = orientation

    # ---- output -------------------------------------------------------------- #
    if task == "regression":
        regression_plot(y_true, y_score, test_metrics["spearman"], args.outdir / "plot.png")
    elif n_classes == 2:
        roc_plot(y_true, y_score, test_metrics["roc_auc"], args.outdir / "plot.png")

    comparison_plot(
        [(f"{args.model} + {args.features}", metric_val, True)]
        + [(f"zero-shot {n}", v, False) for n, v in baselines.items() if n != "_orientation"],
        metric_name, args.outdir / "comparison.png",
    )

    summary = {
        "pipeline_version": __version__,
        "dataset": args.csv.name,
        "task": task,
        "n_variants": int(len(y)),
        "n_features": int(X.shape[1]),
        "classes": classes,
        "positive_class": classes[1] if classes and len(classes) == 2 else None,
        "features": args.features,
        "esm_pool": args.esm_pool if args.features == "esm" else None,
        "esm_model": args.esm_model if args.features == "esm" or "esm" in baselines else None,
        "model": args.model,
        "split": args.split,
        "n_groups": int(len(np.unique(groups))) if groups is not None else None,
        "tune": args.tune,
        "selected_params": best_params,
        "seed": args.seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "cross_validation": cv,
        "held_out_test": test_metrics,
        "zero_shot_baselines": baselines,
        "environment": environment_info(),
    }
    (args.outdir / "results.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved summary -> %s", args.outdir / "results.json")


if __name__ == "__main__":
    main()
