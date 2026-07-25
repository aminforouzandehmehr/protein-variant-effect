"""End-to-end runs on synthetic data with a known, learnable signal."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pve.aaindex import AMINO_ACIDS, HYDROPATHY
from pve.data import parse_mutant
from pve.evaluate import cross_validate, environment_info, fit_and_evaluate
from pve.features import featurize_mutation
from pve.splits import make_groups, make_test_split

REPO = Path(__file__).resolve().parent.parent


def synthetic_assay(n_positions=25, seed=0) -> pd.DataFrame:
    """Substitutions whose fitness depends on the hydropathy change, plus noise."""
    rng = np.random.RandomState(seed)
    wt = "".join(rng.choice(list(AMINO_ACIDS), size=60))
    rows = []
    for pos in range(1, n_positions + 1):
        for mut in AMINO_ACIDS:
            if mut == wt[pos - 1]:
                continue
            seq = wt[: pos - 1] + mut + wt[pos:]
            delta = HYDROPATHY[mut] - HYDROPATHY[wt[pos - 1]]
            score = -abs(delta) + rng.normal(0, 0.5)
            rows.append(
                {
                    "mutant": f"{wt[pos - 1]}{pos}{mut}",
                    "mutated_sequence": seq,
                    "DMS_score": score,
                    "DMS_bin_score": "Pathogenic" if score < -3 else "Benign",
                    "protein_id": f"P{pos % 4}",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def assay_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    synthetic_assay().to_csv(path, index=False)
    return path


def _run(csv, outdir, *extra):
    cmd = [sys.executable, "-W", "ignore", "seq2function.py",
           "--csv", str(csv), "--outdir", str(outdir), *extra]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads((Path(outdir) / "results.json").read_text())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_regression_run_learns_the_planted_signal(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "reg", "--features", "mutation")
    assert res["task"] == "regression"
    assert res["held_out_test"]["spearman"] > 0.4
    assert res["n_train"] + res["n_test"] == res["n_variants"]
    assert (tmp_path / "reg" / "plot.png").exists()


def test_classification_run(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "clf", "--features", "mutation",
               "--score-col", "DMS_bin_score")
    assert res["task"] == "classification"
    assert res["classes"] == ["Benign", "Pathogenic"]
    assert res["positive_class"] == "Pathogenic"
    assert res["held_out_test"]["roc_auc"] > 0.6


def test_onehot_run_uses_sparse_features(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "oh", "--features", "onehot_seq")
    assert res["n_features"] == 60 * len(AMINO_ACIDS)


def test_position_split_is_recorded_and_grouped(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "pos", "--features", "mutation", "--split", "position")
    assert res["split"] == "position"
    assert res["n_groups"] == 25


def test_protein_split_uses_the_group_column(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "prot", "--features", "mutation",
               "--split", "protein", "--group-col", "protein_id")
    assert res["n_groups"] == 4


def test_baselines_are_reported_alongside_the_model(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "base", "--features", "mutation")
    assert set(res["zero_shot_baselines"]) >= {"blosum", "grantham", "_orientation"}
    assert np.isfinite(res["zero_shot_baselines"]["blosum"])


def test_baselines_can_be_switched_off(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "nobase", "--features", "mutation", "--zeroshot", "none")
    assert res["zero_shot_baselines"] == {}


def test_run_records_its_environment(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "env", "--features", "mutation")
    env = res["environment"]
    assert env["packages"]["sklearn"] and env["python"]
    assert res["seed"] == 42 and res["tune"] == "auto"


def test_tuning_selects_and_records_a_hyperparameter(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "tuned", "--features", "mutation")
    assert "model__alpha" in res["selected_params"]
    assert len(res["cross_validation"]["cv_fold_scores"]) == res["cross_validation"]["cv_folds"]


def test_tuning_can_be_disabled(assay_csv, tmp_path):
    res = _run(assay_csv, tmp_path / "untuned", "--features", "mutation", "--tune", "off")
    assert res["selected_params"] is None


def test_runs_are_reproducible(assay_csv, tmp_path):
    a = _run(assay_csv, tmp_path / "r1", "--features", "mutation", "--seed", "7")
    b = _run(assay_csv, tmp_path / "r2", "--features", "mutation", "--seed", "7")
    assert a["held_out_test"] == b["held_out_test"]


def test_different_seeds_give_different_splits(assay_csv, tmp_path):
    a = _run(assay_csv, tmp_path / "s1", "--features", "mutation", "--seed", "1")
    b = _run(assay_csv, tmp_path / "s2", "--features", "mutation", "--seed", "2")
    assert a["held_out_test"] != b["held_out_test"]


# --------------------------------------------------------------------------- #
# library-level guarantee
# --------------------------------------------------------------------------- #
def test_cross_validation_never_sees_the_test_rows():
    """The test split is carved off first; CV runs on the remainder only.

    Previously both were computed over the full array, so the two reported
    numbers shared rows.
    """
    df = synthetic_assay()
    parsed = df["mutant"].map(parse_mutant)
    X, keep, _ = featurize_mutation(df["mutated_sequence"], parsed)
    y = df["DMS_score"].to_numpy(float)[keep]
    groups = make_groups(df[keep], list(parsed[keep]), "position")

    train_idx, test_idx = make_test_split(y, groups, "regression", 0.2, seed=0)
    assert not (set(train_idx) & set(test_idx))
    assert not (set(groups[train_idx]) & set(groups[test_idx]))

    cv = cross_validate(
        X[train_idx], y[train_idx], groups[train_idx], "ridge", 0, "regression", 0,
        folds=3, tune="auto",
    )
    assert len(cv["cv_fold_scores"]) == 3

    metrics, y_true, y_pred, best = fit_and_evaluate(
        X, y, groups, train_idx, test_idx, "ridge", 0, "regression", 0, "auto", False, 3
    )
    assert len(y_true) == len(test_idx)
    assert metrics["spearman"] > 0.3
    assert "model__alpha" in best


def test_environment_info_reports_versions():
    info = environment_info()
    assert info["packages"]["numpy"] and info["python"]


# --------------------------------------------------------------------------- #
# bundled quickstart
# --------------------------------------------------------------------------- #
EXAMPLE_CSV = REPO / "examples" / "example_assay.csv"


def test_bundled_example_exists():
    """The README quickstart must work from a clean clone."""
    assert EXAMPLE_CSV.exists(), "examples/example_assay.csv is missing from the repo"


def test_bundled_example_reproduces_the_readme_regression_numbers(tmp_path):
    res = _run(EXAMPLE_CSV, tmp_path / "ex", "--features", "mutation")
    assert res["task"] == "regression"
    assert res["n_variants"] == 760
    assert res["held_out_test"]["spearman"] == pytest.approx(0.583, abs=0.02)
    assert res["zero_shot_baselines"]["blosum"] == pytest.approx(0.535, abs=0.02)


def test_bundled_example_reproduces_the_readme_classification_numbers(tmp_path):
    res = _run(EXAMPLE_CSV, tmp_path / "exc", "--features", "mutation",
               "--score-col", "DMS_bin_score")
    assert res["classes"] == ["Benign", "Pathogenic"]
    assert res["held_out_test"]["roc_auc"] == pytest.approx(0.830, abs=0.02)


def test_bundled_example_position_split_is_harder(tmp_path):
    """The headline lesson of the example, asserted rather than asserted-in-prose."""
    rand = _run(EXAMPLE_CSV, tmp_path / "r", "--features", "mutation", "--split", "random")
    pos = _run(EXAMPLE_CSV, tmp_path / "p", "--features", "mutation", "--split", "position")
    assert pos["n_groups"] == 40
    assert pos["cross_validation"]["cv_spearman_mean"] < rand["cross_validation"]["cv_spearman_mean"]


def test_example_generator_is_deterministic(tmp_path):
    """Regenerating the CSV must not silently change the committed data."""
    proc = subprocess.run([sys.executable, "examples/make_example_assay.py"],
                          cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    proc2 = subprocess.run(["git", "diff", "--quiet", "--", "examples/example_assay.csv"],
                           cwd=REPO, capture_output=True, text=True)
    if proc2.returncode == 128:
        pytest.skip("not a git checkout")
    assert proc2.returncode == 0, "regenerating example_assay.csv changed the committed file"
