"""protein-variant-effect — a leakage-aware baseline for variant-effect prediction.

The public surface is deliberately small; ``seq2function.py`` is the CLI that
wires these pieces together.
"""

__version__ = "2.0.0"

__all__ = [
    "aaindex",
    "baselines",
    "data",
    "esm_features",
    "evaluate",
    "features",
    "metrics",
    "models",
    "plots",
    "splits",
]
