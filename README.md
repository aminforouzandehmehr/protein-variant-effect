# protein-variant-effect

[![tests](https://github.com/aminforouzandehmehr/protein-variant-effect/actions/workflows/ci.yml/badge.svg)](https://github.com/aminforouzandehmehr/protein-variant-effect/actions/workflows/ci.yml)

A compact, reproducible baseline for predicting **protein variant effects from sequence**,
built as a clean ML pipeline. It handles both
clinical pathogenicity (Benign/Pathogenic classification) and deep-mutational-scanning (DMS)
fitness (regression), with multiple feature modes and rigorous, leakage-aware evaluation.

Given a set of protein variants and a per-variant label, it trains a supervised model to
predict the label from sequence and reports held-out performance **next to training-free
baselines**, so that "did the model actually learn anything?" is answered rather than
assumed.

This is intended as a clean baseline and a demonstration of an end-to-end, leakage-aware
modeling pipeline.

## Task types (auto-detected)

- **regression** — continuous functional/fitness scores (e.g. ProteinGym DMS assays). Primary metric: Spearman correlation.
- **classification** — categorical labels such as clinical pathogenicity (e.g. ProteinGym clinical / ClinVar: Benign vs Pathogenic). Primary metric: ROC-AUC (binary) or accuracy / macro-F1.

The task is inferred from the label column (mostly non-numeric, or ≤2 unique values →
classification); override with `--task`. A handful of unparseable scores in an otherwise
numeric column are treated as missing data rather than flipping the whole run.

## Methodology

1. **Input.** A DMS or clinical dataset (CSV) of protein variants, each with a full mutated
   amino-acid sequence and a label. Column defaults match the
   [ProteinGym](https://proteingym.org) substitution format (`mutated_sequence`, `DMS_score`,
   `mutant`).

2. **Featurization** (`--features`):
   - `onehot_seq` *(default)* — one-hot encoding of the full mutated sequence
     (length `L` × 20 amino acids), held **sparse**; the dense form is ~2.4 GB for a
     300-residue protein with 100k variants.
   - `mutation` — a compact, memory-light encoding of the substitution: relative position,
     wild-type and mutant one-hots, **BLOSUM62 exchangeability, Grantham chemical distance**,
     and the physicochemical properties of both residues plus their difference.
     Multi-substitution variants are averaged and carry an `n_subs` column.
   - `esm` *(optional, strongest)* — [ESM-2](https://github.com/facebookresearch/esm)
     embeddings **at the mutated position**, concatenated with their contrast against the
     wild type at the same position (`--esm-pool both`, the default).

3. **Models** (`--model`): ridge/logistic regression *(default)*, gradient-boosted trees, or
   a small MLP. Every model is a pipeline beginning with a `VarianceThreshold`, and
   regularisation strength is tuned by grid search **inside each training fold**.

4. **Evaluation.** The held-out test set is carved off **first**; cross-validation then runs
   on the remainder, with hyper-parameter search in an inner loop (nested CV). Splits can
   hold out random variants, whole **positions**, or whole **proteins** (`--split`).

5. **Baselines.** Every run also scores training-free predictors on the same held-out rows:
   BLOSUM62, Grantham, and optionally ESM-2 masked marginals (`--zeroshot`).

Results are written to `results/results.json`, `results/plot.png`, and
`results/comparison.png`.

## Installation

```bash
pip install -r requirements.txt
```

The ESM feature mode and the ESM zero-shot baseline additionally need
`pip install -r requirements-esm.txt` (torch + fair-esm).

## Quickstart

A small example assay ships with the repo, so this works immediately after cloning —
no download, no network:

```bash
python seq2function.py --csv examples/example_assay.csv --features mutation
```

```
Task: regression (760 distinct numeric labels)
mutation-level features: (760, 60) (760/760 variants parseable)
CV cv_spearman_mean = 0.529 +/- 0.042
Held-out test (Spearman = 0.583)
Zero-shot blosum    Spearman = 0.535
Zero-shot grantham  Spearman = 0.469
```

`examples/example_assay.csv` is synthetic — 760 substitutions across 40 positions of a
100-residue protein, regenerable with `python examples/make_example_assay.py`. It is
generated rather than borrowed so it carries this repo's license, runs offline, and stays
deterministic. Each position has a hidden burial weight and each substitution's cost scales
that weight by its hydropathy and Grantham distance, which is why the chemistry baselines
score well above chance and why position-held-out evaluation is genuinely harder:

| run | CV ρ | held-out ρ |
| --- | --- | --- |
| `--features mutation --split random` | 0.529 | 0.583 |
| `--features mutation --split position` | 0.209 | 0.534 |
| `--features onehot_seq --split random` | 0.572 | 0.627 |
| zero-shot BLOSUM62 | — | 0.535 |
| zero-shot Grantham | — | 0.469 |

The same file also carries a binary label for the classification path:

```bash
python seq2function.py --csv examples/example_assay.csv --features mutation \
    --score-col DMS_bin_score
```

```
Classes: ['Benign', 'Pathogenic']  (counts: [570, 190])
CV cv_roc_auc_mean = 0.785 +/- 0.041
Held-out test (ROC-AUC = 0.830 | accuracy = 0.757 | F1 = 0.611)
Zero-shot blosum    ROC-AUC = 0.781
```

Note how close the free BLOSUM62 baseline (0.781) runs to the trained model (0.830) — on
synthetic chemistry-driven data that is expected, and it is exactly the kind of comparison
this pipeline exists to surface. For results on real data, see
[Example output](#example-output) below.

## Usage

```bash
# default: one-hot sequence features + ridge regression
python seq2function.py --csv DATASET.csv
```

```bash
# substitution-level features, holding out whole positions (extrapolation)
python seq2function.py --csv DATASET.csv --features mutation --split position
```

```bash
# clinical pathogenicity (classification)
python seq2function.py --csv CLINICAL.csv --score-col DMS_bin_score --features mutation
```

```bash
# ESM-2 features plus the ESM zero-shot baseline
python seq2function.py --csv DATASET.csv --features esm --zeroshot blosum,grantham,esm
```

Override column names with `--seq-col`, `--score-col`, `--mutant-col`. Run
`python seq2function.py --help` for all options.

### Evaluation options that change what the number means

| flag | effect |
| --- | --- |
| `--split random` *(default)* | Interpolation within an assay: other substitutions at the same position are in the training set. |
| `--split position` | Extrapolation to unseen positions. Strictly harder, and the honest number for "how will this do at a site I have not assayed?" |
| `--split protein --group-col COL` | Extrapolation to unseen proteins. Needs a CSV spanning several targets. |
| `--zeroshot blosum,grantham,esm` | Training-free reference points scored on the same held-out rows. |
| `--tune auto/on/off` | Grid search inside each training fold. `auto` tunes the linear models only. |
| `--esm-pool both/mut_pos/delta/mean` | How ESM representations are reduced to a feature vector. |

## Example output

Real data, not the bundled synthetic file.
`BLAT_ECOLX_Stiffler_2015` from the ProteinGym DMS benchmark (4,996 β-lactamase variants,
continuous fitness scores), `--seed 42`, ESM-2 35M. Held-out Spearman ρ:

| approach | split | held-out ρ |
| --- | --- | --- |
| zero-shot Grantham distance | random | 0.222 |
| zero-shot BLOSUM62 | random | 0.324 |
| supervised `mutation` + ridge | random | 0.514 |
| **zero-shot ESM-2 masked marginals** | random | **0.550** |
| supervised `esm` + ridge, mean-pooled | random | 0.764 |
| supervised `esm` + ridge, position-aware | random | **0.867** |
| supervised `mutation` + ridge | position | 0.419 |
| supervised `esm` + ridge, position-aware | position | 0.660 |

Three things worth reading off that table:

- **The supervised substitution model loses to a free baseline.** ESM-2 masked marginals
  (0.550) beat `mutation` + ridge (0.514) with no training at all. That comparison is the
  reason the baselines are reported by default.
- **Where you pool the embedding matters more than the model.** For a single substitution in
  a length-`L` protein, mean-pooled wild-type and mutant embeddings differ by roughly `1/L`,
  so mean pooling averages away the signal. Reading the representation at the mutated
  position instead moves ρ from 0.764 to 0.867.
- **Random splits flatter every model.** Holding out whole positions costs the ESM model
  0.207 ρ (0.867 → 0.660) and the substitution model 0.095 (0.514 → 0.419). The random
  number is not wrong, it just answers an easier question.

Classification on a single-gene clinical assay (`NP_000060.2`, 140 missense variants labeled
Benign/Pathogenic), mutation features, default logistic regression:

```
Task: classification   |   n_variants = 140   |   classes = [Benign, Pathogenic]
Cross-validation:  ROC-AUC = 0.77 +/- 0.19  (stratified 5-fold, training split only)
Held-out test:     ROC-AUC = 0.88 | accuracy = 0.71 | F1 = 0.43
Zero-shot BLOSUM62:   ROC-AUC = 0.65
Zero-shot Grantham:   ROC-AUC = 0.64
```

ROC-AUC is reported as the primary, threshold-independent metric; the lower F1 at the
default 0.5 threshold reflects class imbalance rather than absence of signal. With a single
small gene the estimate is necessarily modest — this is a baseline/learning project, not a
state-of-the-art predictor.

## Reproducibility

`results.json` records the split strategy and group count, the selected hyper-parameters,
per-fold scores, the class→integer mapping and which class was treated as positive, the
seed, and the exact Python/NumPy/scikit-learn/torch versions plus the git commit that
produced the run.

ESM embeddings are cached in `.esm_cache/embeddings.npz` (keyed by model, pooling mode,
sequence hash and position), so re-running an assay does not re-embed it. Pass
`--esm-cache ''` to disable.

## Getting data

`examples/example_assay.csv` is bundled so the pipeline runs out of the box. For real
assays, download a substitution dataset from ProteinGym (https://proteingym.org) or MaveDB
(https://www.mavedb.org). Each is a CSV with a mutated-sequence column, a mutant column and
a score; point `--csv` at it. None are redistributed here — they carry the terms of their
~200 source publications.

## Development

```bash
pip install -r requirements-dev.txt && pytest -q
```

The suite covers the substitution tables against published values, the featurizers, the
grouped-split guarantees, the ESM indexing and scoring logic (no torch needed — that logic
is pure NumPy), and end-to-end CLI runs on synthetic data with a planted signal. CI runs it
on Python 3.9–3.12.

## Limitations

- Linear / shallow baselines capture additive effects well but miss higher-order epistasis;
  ESM embeddings narrow that gap without closing it.
- The supervised models are trained per assay. Nothing here transfers across proteins;
  `--split protein` will show that plainly on a multi-target CSV.
- ESM zero-shot uses masked marginals from a single model. The published protocol ensembles
  several ESM-1v checkpoints and does better than the numbers above.
- Sequences longer than 1022 residues are windowed around the mutated position for ESM,
  which drops long-range context.
- No MSA-derived features (conservation, PSSM) and no structural features (solvent
  accessibility, ΔΔG). Both are the obvious next additions to `--features mutation`.

### Note for macOS users

NumPy 2.0 built against Apple's Accelerate BLAS emits spurious `overflow`/`divide by zero
encountered in matmul` warnings — a bare `np.float32` matmul triggers them, and the results
are unaffected. Upgrading NumPy or running with `python -W ignore` silences them.

## License

MIT
