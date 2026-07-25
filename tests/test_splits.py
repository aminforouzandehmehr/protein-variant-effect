"""The splitting guarantees are the point of the `--split` work, so they are
asserted directly: a held-out group must never appear in training."""

import numpy as np
import pandas as pd
import pytest

from pve.data import parse_mutant
from pve.splits import make_cv_splitter, make_groups, make_test_split


def _assay(n_positions=20, per_position=6):
    rows, parsed = [], []
    for pos in range(1, n_positions + 1):
        for k in range(per_position):
            rows.append({"sequence": "A" * 50, "group": f"prot{pos % 3}"})
            parsed.append([("A", pos, "CDEFGHIK"[k % 8])])
    return pd.DataFrame(rows), parsed


def test_random_split_has_no_groups():
    df, parsed = _assay()
    assert make_groups(df, parsed, "random") is None


def test_position_groups_one_per_position():
    df, parsed = _assay(n_positions=20, per_position=6)
    groups = make_groups(df, parsed, "position")
    assert len(np.unique(groups)) == 20


def test_protein_groups_use_the_group_column():
    df, parsed = _assay()
    groups = make_groups(df, parsed, "protein")
    assert set(np.unique(groups)) == {"prot0", "prot1", "prot2"}


def test_single_group_is_an_error_not_a_crash_later():
    df = pd.DataFrame({"sequence": ["A" * 5] * 4, "group": ["p"] * 4})
    parsed = [[("A", 1, "C")]] * 4
    with pytest.raises(ValueError, match="no split is possible"):
        make_groups(df, parsed, "protein")


def test_position_test_split_shares_no_position_with_training():
    df, parsed = _assay()
    groups = make_groups(df, parsed, "position")
    y = np.random.RandomState(0).rand(len(df))
    train, test = make_test_split(y, groups, "regression", 0.2, seed=0)
    assert not (set(groups[train]) & set(groups[test]))
    assert len(train) + len(test) == len(df)


def test_random_test_split_partitions_the_data():
    y = np.random.RandomState(0).rand(100)
    train, test = make_test_split(y, None, "regression", 0.2, seed=0)
    assert sorted(np.concatenate([train, test])) == list(range(100))
    assert len(test) == 20


def test_stratified_test_split_keeps_both_classes():
    y = np.array([0] * 80 + [1] * 20)
    train, test = make_test_split(y, None, "classification", 0.2, seed=0)
    assert set(y[test]) == {0, 1}


def test_grouped_cv_folds_never_share_a_group():
    df, parsed = _assay()
    groups = make_groups(df, parsed, "position")
    y = np.random.RandomState(0).rand(len(df))
    splitter, folds = make_cv_splitter("regression", groups, 5, seed=0, y=y)
    assert folds == 5
    for tr, te in splitter.split(np.zeros((len(y), 2)), y, groups):
        assert not (set(groups[tr]) & set(groups[te]))


def test_fold_count_shrinks_instead_of_crashing():
    """Three positions cannot support five grouped folds."""
    df, parsed = _assay(n_positions=3, per_position=4)
    groups = make_groups(df, parsed, "position")
    y = np.random.RandomState(0).rand(len(df))
    _, folds = make_cv_splitter("regression", groups, 5, seed=0, y=y)
    assert folds == 3
