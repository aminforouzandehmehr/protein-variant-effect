import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pve.aaindex import AA_INDEX, N_AA, blosum62, grantham_distance
from pve.data import parse_mutant
from pve.features import featurize_mutation, featurize_onehot_seq, mutation_feature_names


def _frame():
    seqs = pd.Series(["MRVLA", "MKVLC", "MKPLA", "MKVLA"])
    parsed = pd.Series([parse_mutant(m) for m in ["K2R", "A5C", "V3P", "K2R:A5C"]])
    return seqs, parsed


# --------------------------------------------------------------------------- #
# one-hot
# --------------------------------------------------------------------------- #
def test_onehot_is_sparse_with_one_hit_per_position():
    seqs = pd.Series(["MKVLA", "MRVLA"])
    X, keep, names = featurize_onehot_seq(seqs)
    assert sparse.issparse(X)
    assert X.shape == (2, 5 * N_AA)
    assert keep.all()
    assert X.sum(axis=1).A1.tolist() == [5, 5]
    assert len(names) == X.shape[1]


def test_onehot_encodes_the_right_residues():
    X, _, _ = featurize_onehot_seq(pd.Series(["MK"]))
    dense = X.toarray()[0]
    assert dense[0 * N_AA + AA_INDEX["M"]] == 1
    assert dense[1 * N_AA + AA_INDEX["K"]] == 1
    assert dense.sum() == 2


def test_onehot_rejects_ragged_sequences():
    with pytest.raises(ValueError, match="equal-length"):
        featurize_onehot_seq(pd.Series(["MKV", "MK"]))


# --------------------------------------------------------------------------- #
# mutation-level
# --------------------------------------------------------------------------- #
def test_mutation_features_have_named_columns():
    seqs, parsed = _frame()
    X, keep, names = featurize_mutation(seqs, parsed)
    assert X.shape == (4, len(names)) == (4, len(mutation_feature_names()))
    assert keep.all()


def test_mutation_features_carry_blosum_and_grantham():
    """The scalars that make the encoding more than an additive AA preference."""
    seqs = pd.Series(["MRVLA"])
    parsed = pd.Series([parse_mutant("K2R")])
    X, _, names = featurize_mutation(seqs, parsed)
    row = X[0]
    assert row[names.index("blosum62")] == pytest.approx(blosum62("K", "R"))
    assert row[names.index("grantham")] == pytest.approx(grantham_distance("K", "R"), abs=1e-3)
    assert row[names.index("wt_K")] == 1 and row[names.index("mut_R")] == 1


def test_mutation_delta_properties_are_mut_minus_wt():
    seqs = pd.Series(["MRVLA"])
    parsed = pd.Series([parse_mutant("K2R")])
    X, _, names = featurize_mutation(seqs, parsed)
    row = X[0]
    for prop in ("hydropathy", "volume", "charge"):
        assert row[names.index(f"delta_{prop}")] == pytest.approx(
            row[names.index(f"mut_{prop}")] - row[names.index(f"wt_{prop}")], abs=1e-4
        )


def test_multi_substitution_variants_are_averaged_and_counted():
    seqs = pd.Series(["MRVLC"])
    parsed = pd.Series([parse_mutant("K2R:A5C")])
    X, keep, names = featurize_mutation(seqs, parsed)
    assert keep.all()
    assert X[0, names.index("n_subs")] == 2
    expected = (blosum62("K", "R") + blosum62("A", "C")) / 2
    assert X[0, names.index("blosum62")] == pytest.approx(expected)


def test_unparseable_rows_are_masked_out():
    seqs = pd.Series(["MRVLA", "MKVLC"])
    parsed = pd.Series([parse_mutant("K2R"), None])
    X, keep, _ = featurize_mutation(seqs, parsed)
    assert keep.tolist() == [True, False]
    assert X.shape[0] == 1


def test_mutation_features_raise_when_nothing_parses():
    with pytest.raises(ValueError, match="No mutants could be parsed"):
        featurize_mutation(pd.Series(["MKVLA"]), pd.Series([None]))


def test_positional_features_use_no_dataset_wide_statistics():
    """Featurizing a subset must give byte-identical rows to featurizing the whole set.

    The original encoding z-scored position using the mean and standard deviation
    of *every* variant, so a row's features depended on which other variants --
    including held-out ones -- were in the file.
    """
    seqs, parsed = _frame()
    full, _, _ = featurize_mutation(seqs, parsed)
    subset, _, _ = featurize_mutation(seqs.iloc[:2], parsed.iloc[:2])
    assert np.allclose(full[:2], subset)
