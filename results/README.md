# Results

Model outputs reported in the paper. Two independent evaluations: 5-fold
cross-validation on TCGA, and inference on the external CPTAC cohort.

## `fold_1/` … `fold_5/` — TCGA cross-validation

Held-out validation predictions, one directory per fold of `splits/folds.csv`.
Each case appears in the validation set of exactly one fold, so the five
directories together cover the full cohort without overlap:

| fold | cases |
|------|-------|
| 1 | 1,428 |
| 2 | 1,406 |
| 3 | 1,407 |
| 4 | 1,403 |
| 5 | 1,419 |
| **total** | **7,063** |

Each directory contains two files, with the same rows and the same six
metadata columns (`Case ID`, `Samples`, `File ID`, `Sample ID`, `Project ID`,
plus a stray `Unnamed: 0` index left over from the export):

- **`val_predictions_activities.csv`** (36 columns) — predicted exposures, one
  `pred_<SBS>` column for each of the 30 COSMIC SBS signatures.

- **`val_predictions_profiles.csv`** (102 columns) — predicted 96-channel
  mutational profile, one `pred_` column per trinucleotide context
  (`A[C>A]A`, `A[C>A]C`, …).

The ground truth (exposures from SigProfilerAssignment and mutational profiles
from the matched WGS) is derived from controlled-access data and is not
released; see *Data availability* in the main README.

## `cptac/cptac_exposures.csv` — external cohort

Predictions on CPTAC, produced with the final checkpoint. 3,333 rows, one per
image (`Image ID`, `Case ID`), followed by one column per signature. Like the
fold files it contains **predictions only**, and the evaluation reported in the paper is restricted to
the subset of samples with matched WGS after Mutect2-based filtering and WSI
matching.

## The model checkpoint

`checkpoints/hist2sig.pt` (outside this directory) is the model to use for
inference on new data. It is not one of the five CV models: it is a refit on
train + validation combined, which is also what produced the CPTAC
predictions above. The per-fold CV checkpoints are not tracked in this
repository.

Inference on a directory of extracted `.pt` feature files:

```bash
python src/predict_activities_nb.py --features <dir> --ckpt checkpoints/hist2sig.pt --out_csv predictions.csv
```

## How these were produced

The fold predictions come from 5-fold cross-validation on `splits/folds.csv`;
the CPTAC predictions come from `checkpoints/hist2sig.pt` through
`src/predict_activities_nb.py`. The metrics reported in the paper can be
recomputed from these files, together with the controlled-access ground
truth, with
[`Paper_reproducibility/Performance_Evaluation_summary.ipynb`](../Paper_reproducibility/Performance_Evaluation_summary.ipynb).
