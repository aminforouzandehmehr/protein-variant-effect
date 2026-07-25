"""Amino-acid constants and substitution scores.

Everything here is a small, self-contained lookup table so the package has no
dependency on Biopython or a downloaded AAindex release. Sources:

* BLOSUM62 — Henikoff & Henikoff (1992) PNAS 89:10915, as distributed by NCBI.
* Grantham composition/polarity/volume and the distance formula —
  Grantham (1974) Science 185:862.
* Hydropathy — Kyte & Doolittle (1982) J Mol Biol 157:105.
"""

from __future__ import annotations

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
N_AA = len(AMINO_ACIDS)

# --------------------------------------------------------------------------- #
# BLOSUM62
# --------------------------------------------------------------------------- #
# Rows and columns are in the conventional BLOSUM ordering below, which is *not*
# alphabetical; _build_blosum reindexes into AMINO_ACIDS order.
_BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _build_blosum62() -> np.ndarray:
    """Return BLOSUM62 as a 20x20 array indexed by AMINO_ACIDS order."""
    rows = [r.split() for r in _BLOSUM62_ROWS.strip().splitlines()]
    raw = np.array([[int(v) for v in r] for r in rows], dtype=np.float32)
    if raw.shape != (N_AA, N_AA):
        raise RuntimeError(f"malformed BLOSUM62 table: {raw.shape}")
    order = [_BLOSUM_ORDER.index(aa) for aa in AMINO_ACIDS]
    return raw[np.ix_(order, order)]


BLOSUM62 = _build_blosum62()


def blosum62(wt: str, mut: str) -> float:
    """BLOSUM62 substitution score. Higher = more exchangeable = more tolerated."""
    i, j = AA_INDEX.get(wt), AA_INDEX.get(mut)
    return float(BLOSUM62[i, j]) if i is not None and j is not None else 0.0


# --------------------------------------------------------------------------- #
# Physicochemical properties
# --------------------------------------------------------------------------- #
# Kyte-Doolittle hydropathy.
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Grantham (1974) atomic composition, polarity and side-chain volume.
GRANTHAM_COMPOSITION = {
    "S": 1.42, "R": 0.65, "L": 0.00, "P": 0.39, "T": 0.71, "A": 0.00, "V": 0.00,
    "G": 0.74, "I": 0.00, "F": 0.00, "Y": 0.20, "C": 2.75, "H": 0.58, "Q": 0.89,
    "N": 1.33, "K": 0.33, "D": 1.38, "E": 0.92, "M": 0.00, "W": 0.13,
}
GRANTHAM_POLARITY = {
    "S": 9.2, "R": 10.5, "L": 4.9, "P": 8.0, "T": 8.6, "A": 8.1, "V": 5.9,
    "G": 9.0, "I": 5.2, "F": 5.2, "Y": 6.2, "C": 5.5, "H": 10.4, "Q": 10.5,
    "N": 11.6, "K": 11.3, "D": 13.0, "E": 12.3, "M": 5.7, "W": 5.4,
}
GRANTHAM_VOLUME = {
    "S": 32.0, "R": 124.0, "L": 111.0, "P": 32.5, "T": 61.0, "A": 31.0, "V": 84.0,
    "G": 3.0, "I": 111.0, "F": 132.0, "Y": 136.0, "C": 55.0, "H": 96.0, "Q": 85.0,
    "N": 56.0, "K": 119.0, "D": 54.0, "E": 83.0, "M": 105.0, "W": 170.0,
}

# Net side-chain charge at pH 7 (histidine is partially protonated).
CHARGE = {aa: 0.0 for aa in AMINO_ACIDS}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})

#: Property names in the order returned by :func:`aa_properties`.
PROPERTY_NAMES = ("hydropathy", "composition", "polarity", "volume", "charge")


def aa_properties(aa: str) -> np.ndarray:
    """Return the property vector for one amino acid (zeros for unknown codes)."""
    if aa not in AA_INDEX:
        return np.zeros(len(PROPERTY_NAMES), dtype=np.float32)
    return np.array(
        [
            HYDROPATHY[aa],
            GRANTHAM_COMPOSITION[aa],
            GRANTHAM_POLARITY[aa],
            GRANTHAM_VOLUME[aa],
            CHARGE[aa],
        ],
        dtype=np.float32,
    )


# Grantham distance constants (Grantham 1974, Table 2 caption).
_GRANTHAM_ALPHA = 1.833
_GRANTHAM_BETA = 0.1018
_GRANTHAM_GAMMA = 0.000399
_GRANTHAM_RHO = 50.723


def grantham_distance(wt: str, mut: str) -> float:
    """Grantham chemical distance between two residues.

    Computed from the composition/polarity/volume tables rather than a stored
    190-entry matrix; reproduces the published values (Ser->Arg = 110,
    Leu->Ile = 5). Higher = more chemically radical = less tolerated.
    """
    if wt not in AA_INDEX or mut not in AA_INDEX:
        return 0.0
    dc = GRANTHAM_COMPOSITION[wt] - GRANTHAM_COMPOSITION[mut]
    dp = GRANTHAM_POLARITY[wt] - GRANTHAM_POLARITY[mut]
    dv = GRANTHAM_VOLUME[wt] - GRANTHAM_VOLUME[mut]
    inner = _GRANTHAM_ALPHA * dc**2 + _GRANTHAM_BETA * dp**2 + _GRANTHAM_GAMMA * dv**2
    return float(_GRANTHAM_RHO * np.sqrt(inner))
