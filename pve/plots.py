"""Figures. Matplotlib is imported lazily so the package stays importable headless."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _axes(figsize=(5, 5)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.subplots(figsize=figsize)


def regression_plot(y_true, y_pred, rho: float, out: Path) -> None:
    plt, (fig, ax) = _axes()
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, edgecolor="none")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1)
    ax.set_xlabel("Measured score")
    ax.set_ylabel("Predicted score")
    ax.set_title(f"Held-out variants (Spearman ρ = {rho:.3f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved plot -> %s", out)


def roc_plot(y_true, scores, auroc: float, out: Path) -> None:
    from sklearn.metrics import roc_curve

    plt, (fig, ax) = _axes()
    fpr, tpr, _ = roc_curve(y_true, scores)
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Held-out ROC (AUC = {auroc:.3f})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved plot -> %s", out)


def comparison_plot(rows, metric_name: str, out: Path) -> None:
    """Supervised model against every zero-shot baseline, on the same test split.

    ``rows`` is a list of ``(label, score, is_supervised)``.
    """
    rows = [(lab, s, sup) for lab, s, sup in rows if np.isfinite(s)]
    if len(rows) < 2:
        return
    plt, (fig, ax) = _axes(figsize=(6, 0.6 * len(rows) + 1.8))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["#2b6cb0" if r[2] else "#a0aec0" for r in rows]
    ypos = np.arange(len(rows))
    ax.barh(ypos, values, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Held-out {metric_name}")
    if metric_name == "ROC-AUC":
        ax.axvline(0.5, color="0.4", ls="--", lw=1)
    else:
        ax.axvline(0.0, color="0.4", ls="--", lw=1)
    for y, v in zip(ypos, values):
        ax.text(v, y, f" {v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_title("Supervised model vs zero-shot baselines")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved plot -> %s", out)
