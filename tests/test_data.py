import numpy as np
import pandas as pd
import pytest

from pve.data import (
    coerce_regression_targets,
    detect_task,
    load_data,
    parse_mutant,
    reconstruct_wt_sequence,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("G128R", [("G", 128, "R")]),
        (" g128r ", [("G", 128, "R")]),
        ("G128R:A200V", [("G", 128, "R"), ("A", 200, "V")]),
        ("A1C;D2E", [("A", 1, "C"), ("D", 2, "E")]),
    ],
)
def test_parse_mutant_accepts_single_and_multiple(text, expected):
    assert parse_mutant(text) == expected


@pytest.mark.parametrize("text", ["", "WT", "nan", "junk", "128R", "G128", "GG128R"])
def test_parse_mutant_rejects_junk(text):
    assert parse_mutant(text) is None


def test_detect_task_continuous_is_regression():
    assert detect_task(pd.Series([0.1, 2.5, -3.0, 1.7, 0.4])) == "regression"


def test_detect_task_binary_numeric_is_classification():
    assert detect_task(pd.Series([0, 1, 1, 0, 1])) == "classification"


def test_detect_task_strings_are_classification():
    assert detect_task(pd.Series(["Benign", "Pathogenic", "Benign"])) == "classification"


def test_detect_task_survives_a_few_stray_values():
    """One unparseable score must not flip a whole DMS assay to classification.

    This was the original behaviour: `to_numeric(...).isna().any()` meant a
    single stray cell silently changed the task.
    """
    labels = pd.Series([0.1, 2.5, -3.0, 1.7, 0.4, 9.9, "n/a"])
    assert detect_task(labels) == "regression"


def test_detect_task_override_wins():
    assert detect_task(pd.Series([0.1, 0.2, 0.3]), "classification") == "classification"


def test_coerce_regression_targets_drops_non_numeric():
    y, keep = coerce_regression_targets(np.array(["1.0", "2.0", "bad", "4.0"], dtype=object))
    assert keep.tolist() == [True, True, False, True]
    assert y.tolist() == [1.0, 2.0, 4.0]


def test_coerce_regression_targets_raises_when_nothing_numeric():
    with pytest.raises(ValueError, match="No numeric scores"):
        coerce_regression_targets(np.array(["a", "b"], dtype=object))


def test_reconstruct_wt_reverts_substitutions():
    wt = "MKVLA"
    seqs = pd.Series(["MRVLA", "MKVLC", "MKPLA"])
    parsed = pd.Series([parse_mutant("K2R"), parse_mutant("A5C"), parse_mutant("V3P")])
    assert reconstruct_wt_sequence(seqs, parsed) == wt


def test_reconstruct_wt_tolerates_a_bad_row():
    """A minority of inconsistent rows must not decide the wild type."""
    seqs = pd.Series(["MRVLA", "MKVLC", "MKPLA", "QQQQQ"])
    parsed = pd.Series(
        [parse_mutant("K2R"), parse_mutant("A5C"), parse_mutant("V3P"), parse_mutant("Z9Y")]
    )
    assert reconstruct_wt_sequence(seqs, parsed) == "MKVLA"


def test_reconstruct_wt_returns_none_when_nothing_matches():
    seqs = pd.Series(["AAAA"])
    parsed = pd.Series([parse_mutant("K2R")])
    assert reconstruct_wt_sequence(seqs, parsed) is None


def test_load_data_reports_missing_columns(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"seq": ["AA"], "y": [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="not found"):
        load_data(csv, "mutated_sequence", "DMS_score", "mutant")


def test_load_data_drops_empty_rows(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {"mutated_sequence": ["AAA", None, "CCC"], "DMS_score": [1.0, 2.0, None]}
    ).to_csv(csv, index=False)
    df = load_data(csv, "mutated_sequence", "DMS_score", "mutant")
    assert len(df) == 1 and df.loc[0, "sequence"] == "AAA"
