# protein-variant-effect

A compact, reproducible baseline for predicting **protein variant effects from sequence** —
built as a clean ML pipeline rather than a state-of-the-art predictor. It handles both
clinical pathogenicity (Benign/Pathogenic classification) and deep-mutational-scanning (DMS)
fitness (regression), with multiple feature modes and rigorous, leakage-aware evaluation.

Given a set of protein variants and a per-variant label, it trains a supervised model to
predict the label from sequence and reports held-out performance (Spearman correlation for
regression; ROC-AUC for classification).

This is intended as a clean baseline and a demonstration of an end-to-end,
leakage-aware modeling pipeline not a state-of-the-art predictor.


## Task types (auto-detected)

- **regression** — continuous functional/fitness scores (e.g. ProteinGym DMS assays). Primary metric: Spearman correlation.
- **classification** — categorical labels such as clinical pathogenicity (e.g. ProteinGym clinical / ClinVar: Benign vs Pathogenic). Primary metric: ROC-AUC (binary) or accuracy / macro-F1.

The task is inferred from the label column (non-numeric or ≤2 unique values → classification); override with `--task`.

## Methodology

1. **Input.** A DMS dataset (CSV) of protein variants, each with a full mutated amino-acid
   sequence and a continuous functional score. Column defaults match the
   [ProteinGym](https://proteingym.org) substitution format (`mutated_sequence`, `DMS_score`).

2. **Featurization** (`--features`):
   - `onehot_seq` *(default)* — one-hot encoding of the full mutated sequence
     (length `L` × 20 amino acids), the most direct sequence-to-function representation.
   - `mutation` — a compact, memory-light encoding of the single substitution
     (normalized position + wild-type one-hot + mutant one-hot); supports variable-length inputs.
   - `esm` *(optional, stronger)* — mean-pooled [ESM-2](https://github.com/facebookresearch/esm)
     protein-language-model embeddings. Requires `fair-esm` and `torch`; slow without a GPU.

3. **Models** (`--model`): ridge regression *(default)*, gradient-boosted trees, or a small MLP.

4. **Evaluation.** Two complementary estimates:
   - **K-fold cross-validation** (default 5-fold) reporting Spearman ρ per fold. All scaling
     and model fitting happen **inside each fold's training set**, so no information leaks
     from held-out variants.
   - A **held-out test split** reporting Spearman, Pearson, R², RMSE, and MAE, with a
     predicted-vs-measured scatter plot.

Results are written to `results/results.json` and `results/scatter.png`.

## Installation

```bash
pip install -r requirements.txt
```

The ESM feature mode additionally needs `pip install fair-esm torch` (optional).

## Usage

```bash
# default: one-hot sequence features + ridge regression
python seq2function.py --csv DATASET.csv

# compact mutation-level features + gradient-boosted trees
python seq2function.py --csv DATASET.csv --features mutation --model gbm

# clinical pathogenicity (classification); mutation features suit small datasets
python seq2function.py --csv CLINICAL.csv --score-col DMS_bin_score --features mutation

# stronger ESM-2 embeddings (slow without GPU; uses Apple-Silicon MPS if available)
python seq2function.py --csv DATASET.csv --features esm --model ridge
```

Override column names with `--seq-col`, `--score-col`, `--mutant-col` if your CSV differs.
Run `python seq2function.py --help` for all options.

## Getting data

Download a single DMS substitution assay from ProteinGym
(https://proteingym.org) or MaveDB (https://www.mavedb.org). Each is a CSV with a
mutated-sequence column and a functional score; point `--csv` at it.

## Example output

Classification on a single-gene clinical assay from the ProteinGym clinical benchmark
(`NP_000060.2`, 140 missense variants labeled Benign/Pathogenic), using mutation-level
features and the default logistic-regression model:

```
python seq2function.py --csv NP_000060.2.csv --score-col DMS_bin_score --features mutation

Task: classification   |   n_variants = 140   |   classes = [Benign, Pathogenic]
Cross-validation:  ROC-AUC = 0.74 +/- 0.13  (stratified 5-fold)
Held-out test:     ROC-AUC = 0.85 | accuracy = 0.75 | F1 = 0.36
```

The model recovers a reproducible sequence-to-pathogenicity signal (cross-validated
ROC-AUC ≈ 0.74). ROC-AUC is reported as the primary, threshold-independent metric; the
lower F1 at the default 0.5 threshold reflects class imbalance in the clinical labels
rather than absence of signal. With a single small gene the estimate is necessarily
modest — this is a baseline/learning project, not a state-of-the-art predictor.

## Limitations

- Linear / shallow baselines on one-hot or mutation features capture additive effects well
  but miss higher-order epistasis; ESM embeddings or deeper models narrow that gap.
- Random train/test splitting estimates interpolation within an assay, not extrapolation to
  unseen positions or proteins; position-held-out splits are a more stringent next step.
- Single-substitution focus for the `mutation` feature mode; `onehot_seq` handles
  multi-substitution variants of equal length.

## License

MIT
