# Signature-Level Patch Attribution (MIL Grad-CAM)

Post-hoc attribution of patch-level influence per mutational signature, without retraining.

## Overview

Standard MIL attention (`val_attention_patches.csv`) produces a single shared attention map per slide, agnostic to the target signature. This script adds **signature-specific** scores by combining attention weights with feature-space gradients — analogous to Grad-CAM adapted for MIL.

Supported model: `attention_mil`.

## Method

For a slide with patches `i = 1..L` and signature logit `y_k`:

1. Compute hidden embeddings `h_i` and MIL attention weights `a_i`.
2. Compute gradient of `y_k` w.r.t. each `h_i`: `g_i = ∂y_k / ∂h_i`.
3. Build attribution score: `grad_score_i = ⟨h_i, g_i⟩` (signed dot product).
4. Rank patches per signature using `combined_i = a_i · ReLU(grad_score_i)` (attention-weighted positive contribution).

`combined_i` drives top-k filtering (`--topk-patches`) but is not written to output.

## Output

Parquet file (Snappy-compressed). One row per `(patch, signature)`.

| Column | Type | Description |
|---|---|---|
| `feature_file` | category | Source `.pt` filename |
| `case_id` | category | Inferred from filename (first 3 dash-separated tokens) |
| `signature_name` | category | Signature label from checkpoint |
| `signature_idx` | int16 | Signature index |
| `coord_x`, `coord_y` | int32 | Patch tile coordinates |
| `original_index` | int32 | Patch index in the full (pre-sampling) feature array |
| `attention_weight` | float32 | MIL attention weight `a_i` |
| `grad_score` | float32 | Signed gradient attribution `⟨h_i, g_i⟩` |

## Usage

```bash
python grad_cam_signatures/export_signature_patch_attribution.py \
  --features-dir <PATH_TO_PT_FILES> \
  --ckpt <PATH_TO_best.pt> \
  --out-parquet <OUTPUT.parquet> \
  [--val-attention-csv <val_attention_patches.csv>] \
  [--bagsize 10000] \
  [--sampling energy_topk] \
  [--energy-mode l2_sq] \
  [--energy-tau 1.0] \
  [--sampling-seed 42] \
  [--topk-patches 200] \
  [--cuda cuda:0]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--val-attention-csv` | — | If provided, restricts processing to slides listed in that CSV |
| `--bagsize` | `0` (all) | Max patches per slide; `0` disables sampling |
| `--sampling` | `none` | Sampling strategy: `none`, `random`, `energy_topk`, `energy_gumbel` |
| `--topk-patches` | `0` (all) | Keep only top-k patches per signature by `combined_score` |

## Notes

- `energy_topk` sampling is deterministic; `random` and `energy_gumbel` are deterministic with a fixed `--sampling-seed`.
- Output size scales as `#slides × #patches × #signatures`. Use `--topk-patches` to bound it.
- This is gradient attribution in **feature space**, not pixel-space Grad-CAM.
