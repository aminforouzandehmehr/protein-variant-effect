"""Zero-shot baselines — scores that involve no training at all.

Without these the pipeline reports a number with nothing to compare it against.
A supervised model that cannot beat a BLOSUM62 lookup has learned nothing about
the assay, and until now the run had no way to say so.

Every baseline returns a **tolerance** score: higher = predicted more wild-type-
like = higher DMS fitness / more likely benign.
"""

from __future__ import annotations

import logging

import numpy as np

from .aaindex import blosum62, grantham_distance

log = logging.getLogger(__name__)

FREE_BASELINES = ("blosum", "grantham")
ALL_BASELINES = FREE_BASELINES + ("esm",)

#: Class-name tokens that identify the *tolerated* side of a binary label.
_BENIGN_TOKENS = ("benign", "neutral", "tolerated", "functional", "wt", "normal")


def blosum_scores(parsed) -> np.ndarray:
    """Summed BLOSUM62 exchangeability over each variant's substitutions."""
    out = np.full(len(parsed), np.nan)
    for i, subs in enumerate(parsed):
        if subs:
            out[i] = sum(blosum62(wt, mut) for wt, _, mut in subs)
    return out


def grantham_scores(parsed) -> np.ndarray:
    """Negated Grantham distance, so that higher still means more tolerated."""
    out = np.full(len(parsed), np.nan)
    for i, subs in enumerate(parsed):
        if subs:
            out[i] = -sum(grantham_distance(wt, mut) for wt, _, mut in subs)
    return out


def classification_orientation(classes) -> tuple[int, str]:
    """Map a tolerance score onto "probability of class 1".

    Returns ``(sign, explanation)``. Class 1 is whichever label sorts second
    alphabetically. If its name looks benign the tolerance score points at it
    directly; otherwise class 1 is the damaging one and the score is negated.
    The rule is fixed in advance from the class *names* -- never chosen by
    whichever direction happens to score better, which would be fitting the
    baseline to the test set.
    """
    if not classes or len(classes) != 2:
        return -1, "assumed class 1 is the damaging class"
    name = str(classes[1]).strip().lower()
    if any(tok in name for tok in _BENIGN_TOKENS):
        return 1, f"class 1 ('{classes[1]}') read as the tolerated class"
    return -1, f"class 1 ('{classes[1]}') read as the damaging class"


def compute_baselines(
    names,
    parsed,
    wt_sequence: str | None,
    esm_kwargs: dict | None = None,
) -> dict:
    """Compute the requested baselines, skipping any that cannot run.

    Returns ``{name: tolerance_score_array}``.
    """
    out: dict = {}
    for name in names:
        if name == "blosum":
            out["blosum"] = blosum_scores(parsed)
        elif name == "grantham":
            out["grantham"] = grantham_scores(parsed)
        elif name == "esm":
            if wt_sequence is None:
                log.warning("Skipping the ESM zero-shot baseline: no wild-type sequence.")
                continue
            try:
                from .esm_features import esm_zeroshot

                out["esm"] = esm_zeroshot(wt_sequence, parsed, **(esm_kwargs or {}))
            except ImportError as exc:
                log.warning("Skipping the ESM zero-shot baseline: %s", exc)
            except Exception as exc:  # a failed baseline must not sink the run
                log.warning("ESM zero-shot baseline failed (%s); continuing.", exc)
        else:
            raise ValueError(f"Unknown baseline '{name}'; choose from {ALL_BASELINES}.")
    return out
