#!/usr/bin/env python3
"""Export signature-specific patch attributions for AttentionMIL checkpoints.

This script computes patch-level attribution scores per signature without retraining.
It supports filtering to validation slides listed in a val_attention_patches.csv file.

Main idea (MIL-adapted Grad-CAM style):
- Compute shared MIL attention over patches.
- For each signature logit y_k, compute gradient wrt per-patch hidden embedding h_i.
- Build per-patch scores from h_i * d(y_k)/d(h_i).

Scores exported per patch/signature:
- grad_score: signed contribution proxy
- grad_abs_score: absolute contribution magnitude
- gradcam_like_score: ReLU(grad_score)
- combined_score: attention_weight * gradcam_like_score
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "utils") not in sys.path:
    sys.path.append(str(_REPO_ROOT / "utils"))
if str(_REPO_ROOT / "models") not in sys.path:
    sys.path.append(str(_REPO_ROOT / "models"))

from models import build_model  # noqa: E402
from sampling_utils import BagSamplingConfig, sample_bag  # noqa: E402



def _as_bool_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((1, length), dtype=torch.bool, device=device)


def _infer_case_id(feature_filename: str) -> str:
    base = os.path.basename(feature_filename)
    parts = base.split("-")
    return "-".join(parts[:3])


def _namespace_from_ckpt(ckpt: dict) -> argparse.Namespace:
    args_dict = {}
    if isinstance(ckpt.get("args"), dict):
        args_dict.update(ckpt["args"])
    for key in ("model", "layers_nodes", "activation", "dropout", "attn_dim", "conv_kernel", "conv_blocks"):
        if key not in args_dict and key in ckpt:
            args_dict[key] = ckpt[key]
    return argparse.Namespace(**args_dict)


def _load_feature_payload(file_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(file_path, map_location="cpu")
    if not isinstance(payload, dict) or "feat" not in payload:
        raise RuntimeError("Unexpected feature payload format in %s" % str(file_path))
    if "coords" not in payload:
        raise RuntimeError("Missing 'coords' in feature payload %s" % str(file_path))
    features = torch.as_tensor(payload["feat"], dtype=torch.float32)
    coords = torch.as_tensor(payload["coords"], dtype=torch.int64)
    if features.ndim != 2:
        raise RuntimeError("Expected features [L,D], got shape %s in %s" % (tuple(features.shape), str(file_path)))
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise RuntimeError("Expected coords [L,2], got shape %s in %s" % (tuple(coords.shape), str(file_path)))
    if coords.shape[0] != features.shape[0]:
        raise RuntimeError("coords/features length mismatch in %s" % str(file_path))
    return features, coords[:, :2]


def _apply_sampling(
    features: torch.Tensor,
    coords: torch.Tensor,
    *,
    bagsize: int,
    sampling: str,
    energy_mode: str,
    energy_tau: float,
    sampling_seed: int | None,
    sample_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_len = int(features.shape[0])
    indices = torch.arange(full_len, dtype=torch.long)
    if bagsize <= 0 or sampling == "none" or full_len <= bagsize:
        return features, coords, indices

    cfg = BagSamplingConfig(
        bagsize=int(bagsize),
        strategy=str(sampling),
        energy_mode=str(energy_mode),
        energy_tau=float(energy_tau),
        seed=sampling_seed,
    )
    per_item_seed = None if sampling_seed is None else int(sampling_seed) + int(sample_idx)
    sampled_feat, meta = sample_bag(features, config=cfg, per_item_seed=per_item_seed)
    sampled_idx = meta["indices"].to(torch.long)
    sampled_coords = coords.index_select(0, sampled_idx)
    return sampled_feat, sampled_coords, sampled_idx


def _safe_softmax(attn_logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.softmax(attn_logits, dim=1)
    attn_logits = attn_logits.masked_fill(mask, torch.finfo(attn_logits.dtype).min)
    attn = torch.softmax(attn_logits, dim=1)
    attn = attn.masked_fill(mask, 0.0)
    denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return attn / denom


def _forward_attention_mil_with_hidden(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor):
    h = model.instance_encoder(x)
    v = torch.tanh(model.attn_V(h))
    u = torch.sigmoid(model.attn_U(h))
    logits_attn = model.attn_w(v * u).squeeze(-1)
    attn = _safe_softmax(logits_attn, mask)
    z = (attn.unsqueeze(-1) * h).sum(dim=1)
    z = model.out_norm(z)
    logits = model.head(z)
    return logits, attn, h


def _rank_and_percentile(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=torch.long, device=values.device)
    if values.numel() > 1:
        pct = 1.0 - (ranks.to(torch.float32) - 1.0) / (values.numel() - 1.0)
    else:
        pct = torch.ones_like(values, dtype=torch.float32)
    return ranks, pct


def _filter_feature_files_from_val_csv(val_csv: Path) -> set[str]:
    keep = set()
    with val_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "feature_file" not in reader.fieldnames:
            raise RuntimeError("val attention csv has no feature_file column: %s" % str(val_csv))
        for row in reader:
            ff = str(row.get("feature_file", "")).strip()
            if ff:
                keep.add(ff)
    return keep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export patch-level signature attributions (MIL Grad-CAM style)")
    p.add_argument("--features-dir", required=True, help="Directory with .pt feature payloads")
    p.add_argument("--ckpt", required=True, help="Checkpoint path (best.pt)")
    p.add_argument("--out-parquet", required=True, help="Output Parquet path")
    p.add_argument("--val-attention-csv", default=None, help="Optional val_attention_patches.csv to filter feature files")
    p.add_argument("--cuda", default="cuda:0", help="CUDA device, e.g. cuda:0")
    p.add_argument("--bagsize", type=int, default=0, help="Optional bag sampling size; 0 disables sampling")
    p.add_argument("--sampling", default="none", choices=["none", "random", "energy_topk", "energy_gumbel"])
    p.add_argument("--energy-mode", default="l2_sq", choices=["l2", "l2_sq", "abs_mean"])
    p.add_argument("--energy-tau", type=float, default=1.0)
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--topk-patches", type=int, default=0, help="Keep only top-k patches per signature; 0 = all")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unexpected checkpoint format")
    activity_columns = ckpt.get("activity_columns")
    if not activity_columns:
        raise RuntimeError("Checkpoint missing activity_columns")

    k = int(ckpt.get("k") or len(activity_columns))
    input_dim = int(ckpt.get("input_dim") or 0)
    if input_dim <= 0:
        raise RuntimeError("Checkpoint missing input_dim")

    ckpt_args = _namespace_from_ckpt(ckpt)
    if not hasattr(ckpt_args, "layers_nodes"):
        ckpt_args.layers_nodes = [256]
    if not hasattr(ckpt_args, "activation"):
        ckpt_args.activation = "gelu"
    if not hasattr(ckpt_args, "dropout"):
        ckpt_args.dropout = 0.0
    if str(getattr(ckpt_args, "model", "")).lower() != "attention_mil":
        raise RuntimeError("This script currently supports model=attention_mil only")

    model = build_model(ckpt_args, input_dim=input_dim, k=k, device=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    model_dtype = next(model.parameters()).dtype

    keep_feature_files = None
    if args.val_attention_csv:
        keep_feature_files = _filter_feature_files_from_val_csv(Path(args.val_attention_csv))

    feature_paths = sorted(Path(args.features_dir).glob("*.pt"))
    if not feature_paths:
        raise RuntimeError("No .pt feature files found in %s" % args.features_dir)

    if keep_feature_files is not None:
        feature_paths = [p for p in feature_paths if p.name in keep_feature_files]

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_idx, file_path in enumerate(feature_paths):
        feature_file = file_path.name
        case_id = _infer_case_id(feature_file)

        feat_cpu, coords_cpu = _load_feature_payload(file_path)
        feat_cpu, coords_cpu, sampled_idx = _apply_sampling(
            feat_cpu,
            coords_cpu,
            bagsize=int(args.bagsize),
            sampling=str(args.sampling),
            energy_mode=str(args.energy_mode),
            energy_tau=float(args.energy_tau),
            sampling_seed=int(args.sampling_seed),
            sample_idx=sample_idx,
        )

        x = feat_cpu.unsqueeze(0).to(device=device, dtype=model_dtype, non_blocking=True)
        mask = _as_bool_mask(int(x.shape[1]), device=device)

        logits, attn, h = _forward_attention_mil_with_hidden(model, x, mask)
        h_s = h[0]
        attn_s = attn[0].to(torch.float32)

        for sig_idx in range(k):
            target = logits[0, sig_idx]
            grad_h = torch.autograd.grad(target, h, retain_graph=True, create_graph=False)[0][0].to(torch.float32)
            h_f = h_s.to(torch.float32)

            grad_score = torch.sum(h_f * grad_h, dim=1)
            combined = attn_s * torch.relu(grad_score)

            keep_mask = torch.ones_like(combined, dtype=torch.bool)
            topk = int(args.topk_patches)
            if topk > 0 and topk < combined.numel():
                top_idx = torch.topk(combined, k=topk, largest=True, sorted=False).indices
                keep_mask = torch.zeros_like(keep_mask)
                keep_mask[top_idx] = True

            sig_name = str(activity_columns[sig_idx]) if sig_idx < len(activity_columns) else "sig_%d" % sig_idx

            kept = keep_mask.nonzero(as_tuple=True)[0].cpu()
            n = kept.numel()
            if n == 0:
                continue

            rows.append({
                "feature_file": [feature_file] * n,
                "case_id": [case_id] * n,
                "signature_name": [sig_name] * n,
                "signature_idx": kept.new_full((n,), sig_idx, dtype=torch.int16).numpy(),
                "coord_x": coords_cpu[kept, 0].numpy().astype("int32"),
                "coord_y": coords_cpu[kept, 1].numpy().astype("int32"),
                "original_index": sampled_idx[kept].numpy().astype("int32"),
                "attention_weight": attn_s[kept].numpy().astype("float32"),
                "grad_score": grad_score[kept].numpy().astype("float32"),
            })

    if not rows:
        raise RuntimeError("No rows collected — check feature files and val_attention_csv filter")

    df = pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)
    for col in ("feature_file", "case_id", "signature_name"):
        df[col] = df[col].astype("category")

    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    print("features_processed:", len(feature_paths))
    print("rows_written:", len(df))
    print("out_parquet:", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
