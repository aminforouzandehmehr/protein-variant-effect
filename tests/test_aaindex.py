"""The substitution tables are transcribed by hand, so they are checked against
published values rather than trusted."""

import numpy as np
import pytest

from pve.aaindex import (
    AMINO_ACIDS,
    BLOSUM62,
    aa_properties,
    blosum62,
    grantham_distance,
)


def test_blosum62_is_symmetric():
    assert np.array_equal(BLOSUM62, BLOSUM62.T)


@pytest.mark.parametrize(
    "wt,mut,expected",
    [
        ("A", "A", 4), ("W", "W", 11), ("C", "C", 9),   # diagonal
        ("A", "W", -3), ("W", "A", -3),                 # symmetric off-diagonal
        ("I", "L", 2), ("D", "E", 2), ("F", "Y", 3),    # conservative pairs
        ("G", "W", -2), ("D", "W", -4),                 # radical pairs
    ],
)
def test_blosum62_known_entries(wt, mut, expected):
    assert blosum62(wt, mut) == expected


def test_blosum62_conservative_beats_radical():
    # Ile->Leu is a classic tolerated swap; Ile->Pro is not.
    assert blosum62("I", "L") > blosum62("I", "P")


@pytest.mark.parametrize(
    "wt,mut,expected",
    [("S", "R", 110), ("L", "I", 5), ("C", "W", 215), ("G", "W", 184), ("D", "E", 45)],
)
def test_grantham_matches_published_distances(wt, mut, expected):
    # Grantham (1974) Table 2; the formula reproduces the table to within rounding.
    assert grantham_distance(wt, mut) == pytest.approx(expected, abs=1.0)


def test_grantham_is_symmetric_and_zero_on_diagonal():
    for aa in AMINO_ACIDS:
        assert grantham_distance(aa, aa) == 0.0
    assert grantham_distance("K", "D") == pytest.approx(grantham_distance("D", "K"))


def test_unknown_residues_degrade_quietly():
    assert blosum62("X", "A") == 0.0
    assert grantham_distance("A", "*") == 0.0
    assert np.all(aa_properties("X") == 0)


def test_charge_signs():
    # index 4 of the property vector is net charge at pH 7
    assert aa_properties("D")[4] < 0 and aa_properties("K")[4] > 0
    assert aa_properties("A")[4] == 0
