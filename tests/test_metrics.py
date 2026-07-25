import numpy as np
import pytest

from pve.baselines import blosum_scores, classification_orientation, grantham_scores
from pve.data import parse_mutant
from pve.metrics import (
    classification_metrics,
    primary_metric,
    regression_metrics,
    safe_spearman,
    score_from_ranking,
)


def test_safe_spearman_returns_nan_for_a_constant_prediction():
    """A model that predicts one value has no rank correlation; it must not raise."""
    assert np.isnan(safe_spearman([1, 2, 3, 4], [7, 7, 7, 7]))


def test_safe_spearman_is_one_for_a_monotone_relation():
    assert safe_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert safe_spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_regression_metrics_are_perfect_on_an_exact_fit():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_classification_metrics_treat_class_one_as_positive():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    m = classification_metrics(y, proba, 2)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_roc_auc_is_nan_when_only_one_class_is_present():
    y = np.array([1, 1, 1])
    proba = np.tile([0.4, 0.6], (3, 1))
    assert np.isnan(classification_metrics(y, proba, 2)["roc_auc"])


def test_primary_metric_picks_the_right_headline():
    assert primary_metric("regression", {"spearman": 0.5})[0] == "Spearman"
    assert primary_metric("classification", {"roc_auc": 0.8})[0] == "ROC-AUC"
    assert primary_metric("classification", {"accuracy": 0.8})[0] == "Accuracy"


def test_score_from_ranking_ignores_nan_entries():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    scores = np.array([1.0, np.nan, 3.0, 4.0])
    assert score_from_ranking(y, scores, "regression", 0) == pytest.approx(1.0)


def test_score_from_ranking_is_nan_without_enough_finite_scores():
    y = np.array([1.0, 2.0, 3.0])
    assert np.isnan(score_from_ranking(y, np.full(3, np.nan), "regression", 0))


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def test_baselines_rank_conservative_above_radical():
    """Both baselines return a *tolerance* score: higher = more wild-type-like."""
    parsed = [parse_mutant("I100L"), parse_mutant("G100W")]
    assert blosum_scores(parsed)[0] > blosum_scores(parsed)[1]
    assert grantham_scores(parsed)[0] > grantham_scores(parsed)[1]


def test_baselines_are_nan_for_unparseable_variants():
    assert np.isnan(blosum_scores([None])[0])
    assert np.isnan(grantham_scores([None])[0])


def test_orientation_flips_for_a_damaging_positive_class():
    sign, _ = classification_orientation(["Benign", "Pathogenic"])
    assert sign == -1


def test_orientation_keeps_sign_for_a_tolerated_positive_class():
    sign, _ = classification_orientation(["Damaging", "Benign"])
    assert sign == 1


def test_orientation_is_decided_by_class_names_not_by_the_data():
    """Guards against the baseline being flipped to whichever direction scores better."""
    sign_a, _ = classification_orientation(["Benign", "Pathogenic"])
    sign_b, _ = classification_orientation(["Benign", "Pathogenic"])
    assert sign_a == sign_b == -1
