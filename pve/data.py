"""Loading, task detection, and mutant-string parsing."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: One substitution, e.g. ``G128R`` -> ``("G", 128, "R")``. Positions are 1-based.
Substitution = "tuple[str, int, str]"

_SUB_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mutant(mutant) -> list | None:
    """Parse a ProteinGym mutant string into a list of substitutions.

    Handles both single (``G128R``) and multiple (``G128R:A200V``) substitutions.
    Returns ``None`` if the string does not parse.
    """
    text = str(mutant).strip()
    if not text or text.lower() in {"nan", "wt", "none"}:
        return None
    subs = []
    for token in re.split(r"[:;,]", text):
        m = _SUB_RE.match(token.strip())
        if m is None:
            return None
        subs.append((m.group(1).upper(), int(m.group(2)), m.group(3).upper()))
    return subs or None


def load_data(
    csv: Path,
    seq_col: str,
    score_col: str,
    mutant_col: str,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Read the variant CSV down to the columns the pipeline uses.

    Returns a frame with ``sequence`` and ``label``, plus ``mutant`` and
    ``group`` when those columns are present in the source.
    """
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
    if group_col and group_col in df.columns:
        keep[group_col] = "group"
    elif group_col:
        log.warning("Group column '%s' not in %s; falling back to sequence identity.",
                    group_col, csv.name)

    n_raw = len(df)
    df = df[list(keep)].rename(columns=keep).dropna(subset=["sequence", "label"])
    df = df[df["sequence"].astype(str).str.len() > 0].reset_index(drop=True)
    if len(df) < n_raw:
        log.info("Dropped %d row(s) with a missing sequence or label.", n_raw - len(df))
    log.info("Loaded %d variants from %s", len(df), csv.name)
    return df


def detect_task(labels: pd.Series, override: str = "auto") -> str:
    """Decide between 'regression' and 'classification', and say why.

    Auto rules, in order:

    1. A label column that is mostly non-numeric is a classification target.
    2. A numeric column with <= 2 distinct values is a binary classification
       target (this is how ProteinGym ships ``DMS_bin_score``).
    3. Anything else is regression.

    A *few* unparseable values in an otherwise numeric column are treated as
    missing data, not as evidence of a classification task -- the original
    behaviour flipped the whole run to classification on a single stray value.
    """
    if override != "auto":
        log.info("Task: %s (forced with --task)", override)
        return override

    numeric = pd.to_numeric(labels, errors="coerce")
    n_bad = int(numeric.isna().sum())
    frac_bad = n_bad / max(len(numeric), 1)

    if frac_bad > 0.5:
        log.info("Task: classification (%.0f%% of labels are non-numeric)", 100 * frac_bad)
        return "classification"
    if n_bad:
        log.warning(
            "%d/%d label(s) (%.1f%%) are not numeric; treating them as missing and "
            "continuing as a numeric task. Force with --task if that is wrong.",
            n_bad, len(numeric), 100 * frac_bad,
        )
    if numeric.nunique() <= 2:
        log.info("Task: classification (numeric label takes only %d distinct values)",
                 numeric.nunique())
        return "classification"
    log.info("Task: regression (%d distinct numeric labels)", numeric.nunique())
    return "regression"


def reconstruct_wt_sequence(sequences: pd.Series, parsed: pd.Series) -> str | None:
    """Recover the wild-type sequence by reverting each variant's substitutions.

    Every parseable row votes for a wild-type sequence; the majority wins. That
    tolerates a minority of rows with off-by-one or otherwise inconsistent
    coordinates instead of silently trusting the first row.
    """
    votes: Counter = Counter()
    for seq, subs in zip(sequences, parsed):
        if not subs or not isinstance(seq, str):
            continue
        chars = list(seq)
        ok = True
        for wt, pos, mut in subs:
            if not (1 <= pos <= len(chars)) or chars[pos - 1] != mut:
                ok = False
                break
            chars[pos - 1] = wt
        if ok:
            votes["".join(chars)] += 1
        if sum(votes.values()) >= 200:  # a clear majority is settled long before this
            break

    if not votes:
        log.warning(
            "Could not reconstruct the wild-type sequence: no variant's mutant string "
            "matches its mutated sequence (check 1-based positions and --mutant-col)."
        )
        return None
    wt_seq, n = votes.most_common(1)[0]
    total = sum(votes.values())
    if n < 0.9 * total:
        log.warning("Wild-type reconstruction is ambiguous (%d/%d rows agree).", n, total)
    return wt_seq


def coerce_regression_targets(raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y, keep_mask)`` for a regression target, dropping non-numeric rows.

    Without this, ``to_numeric(errors="coerce")`` hands NaNs straight to
    scikit-learn, which fails much later with an opaque message.
    """
    y = pd.to_numeric(pd.Series(raw_labels), errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(y)
    n_drop = int((~keep).sum())
    if n_drop:
        log.warning("Dropped %d variant(s) with a non-numeric or infinite score.", n_drop)
    if not keep.any():
        raise ValueError("No numeric scores remain; is --score-col correct?")
    return y[keep], keep
