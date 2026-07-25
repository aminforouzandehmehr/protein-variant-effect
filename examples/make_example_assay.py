#!/usr/bin/env python3
"""Generate `example_assay.csv`, the bundled quickstart dataset.

The data is synthetic on purpose: it ships under this repository's MIT license
with no third-party terms attached, it works offline, and it is deterministic,
so the numbers in the README are reproducible from a clean clone.

The signal is not arbitrary. Each position gets a hidden "burial" weight, and a
substitution's cost scales that weight by how far the mutation moves hydropathy
and Grantham chemical distance:

    score = -burial[pos] * (|dHydropathy| + grantham/50) + noise

That gives the assay two properties worth demonstrating:

* chemistry matters, so BLOSUM62 and Grantham pick up real but partial signal;
* tolerance is *position-specific*, so a model that has seen other mutations at
  a position has an advantage a position-held-out split correctly removes.

Regenerate with:  python examples/make_example_assay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pve.aaindex import AMINO_ACIDS, HYDROPATHY, grantham_distance  # noqa: E402

SEED = 20240624
N_RESIDUES = 100
N_POSITIONS = 40  # positions actually mutagenised
OUT = Path(__file__).resolve().parent / "example_assay.csv"


def main() -> None:
    rng = np.random.RandomState(SEED)
    wt = "".join(rng.choice(list(AMINO_ACIDS), size=N_RESIDUES))
    positions = sorted(rng.choice(np.arange(1, N_RESIDUES + 1), size=N_POSITIONS, replace=False))
    burial = {p: rng.uniform(0.1, 1.0) for p in positions}

    rows = []
    for pos in positions:
        wt_aa = wt[pos - 1]
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            cost = abs(HYDROPATHY[mut_aa] - HYDROPATHY[wt_aa]) + grantham_distance(wt_aa, mut_aa) / 50
            score = -burial[pos] * cost + rng.normal(0, 0.8)
            rows.append(
                {
                    "mutant": f"{wt_aa}{pos}{mut_aa}",
                    "mutated_sequence": wt[: pos - 1] + mut_aa + wt[pos:],
                    "DMS_score": round(float(score), 4),
                }
            )

    df = pd.DataFrame(rows)
    # Bottom quartile is called damaging, mirroring how ProteinGym binarises assays.
    threshold = df["DMS_score"].quantile(0.25)
    df["DMS_bin_score"] = np.where(df["DMS_score"] <= threshold, "Pathogenic", "Benign")

    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} — {len(df)} variants over {len(positions)} positions, "
          f"{N_RESIDUES}-residue protein")
    print(df["DMS_bin_score"].value_counts().to_string())


if __name__ == "__main__":
    main()
