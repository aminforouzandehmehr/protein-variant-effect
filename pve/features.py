"""Featurizers.

Each returns ``(X, keep_mask, feature_names)`` where ``keep_mask`` selects the
input rows that could be featurized. No featurizer fits anything on the full
dataset: every quantity is computed per-variant, so nothing about the test rows
can reach a training fold. (Column scaling and variance filtering are pipeline
steps instead -- see :mod:`pve.models` -- so they are refit inside each fold.)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from .aaindex import (
    AA_INDEX,
    AMINO_ACIDS,
    N_AA,
    PROPERTY_NAMES,
    aa_properties,
    blosum62,
    grantham_distance,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Full-sequence one-hot
# --------------------------------------------------------------------------- #
def featurize_onehot_seq(sequences: pd.Series) -> tuple[sparse.csr_matrix, np.ndarray, list]:
    """One-hot encode the full mutated sequence as a sparse ``L x 20`` matrix.

    Sparse because the dense form is ``n_variants * L * 20`` float32 -- roughly
    2.4 GB for a 300-residue protein with 100k variants. Constant columns (every
    position that is never mutated in this assay, i.e. nearly all of them) are
    dropped by the ``VarianceThreshold`` step inside the model pipeline, where
    the fit sees training rows only.
    """
    seqs = sequences.astype(str)
    lengths = seqs.str.len().unique()
    if len(lengths) != 1:
        raise ValueError(
            f"onehot_seq requires equal-length sequences; found lengths {sorted(lengths)}. "
            f"Use --features mutation for variable-length inputs."
        )
    L = int(lengths[0])

    rows, cols = [], []
    for r, seq in enumerate(seqs):
        for pos, aa in enumerate(seq):
            j = AA_INDEX.get(aa)
            if j is not None:
                rows.append(r)
                cols.append(pos * N_AA + j)
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(seqs), L * N_AA),
        dtype=np.float32,
    )
    names = [f"pos{p + 1}_{aa}" for p in range(L) for aa in AMINO_ACIDS]
    keep = np.ones(len(seqs), dtype=bool)
    log.info("one-hot sequence features: %s (sparse, %.2f%% nonzero)",
             X.shape, 100 * X.nnz / (X.shape[0] * X.shape[1]))
    return X, keep, names


# --------------------------------------------------------------------------- #
# Mutation-level features
# --------------------------------------------------------------------------- #
def mutation_feature_names() -> list:
    """Column names produced by :func:`featurize_mutation`, in order."""
    names = ["rel_pos", "term_dist", "n_subs"]
    names += [f"wt_{aa}" for aa in AMINO_ACIDS]
    names += [f"mut_{aa}" for aa in AMINO_ACIDS]
    names += ["blosum62", "grantham"]
    names += [f"wt_{p}" for p in PROPERTY_NAMES]
    names += [f"mut_{p}" for p in PROPERTY_NAMES]
    names += [f"delta_{p}" for p in PROPERTY_NAMES]
    return names


def _substitution_block(wt: str, mut: str) -> np.ndarray:
    """Per-substitution features: identity one-hots, exchangeability, chemistry."""
    wt_oh = np.zeros(N_AA, dtype=np.float32)
    mut_oh = np.zeros(N_AA, dtype=np.float32)
    if wt in AA_INDEX:
        wt_oh[AA_INDEX[wt]] = 1.0
    if mut in AA_INDEX:
        mut_oh[AA_INDEX[mut]] = 1.0
    wt_p = aa_properties(wt)
    mut_p = aa_properties(mut)
    scalars = np.array([blosum62(wt, mut), grantham_distance(wt, mut)], dtype=np.float32)
    return np.concatenate([wt_oh, mut_oh, scalars, wt_p, mut_p, mut_p - wt_p])


def featurize_mutation(
    sequences: pd.Series, parsed: pd.Series
) -> tuple[np.ndarray, np.ndarray, list]:
    """Compact substitution features.

    Beyond the original (position, WT one-hot, mutant one-hot) encoding this adds
    the two scalars that carry most of the classical signal -- BLOSUM62
    exchangeability and Grantham chemical distance -- plus the physicochemical
    properties of both residues and their difference. Without them a linear model
    on 41 columns can only learn an additive amino-acid preference and a single
    monotonic trend in position.

    Multi-substitution variants are averaged over their substitutions (an
    explicitly additive assumption) and carry an ``n_subs`` column.

    Positional features are ``rel_pos`` and ``term_dist``, both normalised by the
    variant's own sequence length -- unlike a dataset-wide z-score, they involve
    no statistic pooled across train and test rows.
    """
    block = N_AA * 2 + 2 + 3 * len(PROPERTY_NAMES)
    feats, keep = [], np.zeros(len(parsed), dtype=bool)

    for i, (seq, subs) in enumerate(zip(sequences, parsed)):
        if not subs:
            continue
        L = max(len(str(seq)), 1)
        per_sub = np.zeros(block, dtype=np.float32)
        rel_pos, term_dist = 0.0, 0.0
        for wt, pos, mut in subs:
            per_sub += _substitution_block(wt, mut)
            rel_pos += pos / L
            term_dist += min(pos, L - pos + 1) / L
        n = len(subs)
        feats.append(
            np.concatenate(
                [
                    np.array([rel_pos / n, term_dist / n, float(n)], dtype=np.float32),
                    per_sub / n,
                ]
            )
        )
        keep[i] = True

    if not feats:
        raise ValueError(
            "No mutants could be parsed (expected e.g. 'G128R' or 'G128R:A200V'). "
            "Check --mutant-col."
        )
    X = np.vstack(feats).astype(np.float32)
    log.info("mutation-level features: %s (%d/%d variants parseable)",
             X.shape, int(keep.sum()), len(parsed))
    return X, keep, mutation_feature_names()
