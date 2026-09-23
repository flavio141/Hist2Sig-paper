import argparse
import os
import sys
from pathlib import Path
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.amp import autocast

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("models", "utils"):
    if str(_REPO_ROOT / _sub) not in sys.path:
        sys.path.append(str(_REPO_ROOT / _sub))

from models import build_model  # noqa: E402


def _as_list_int(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def _infer_case_id(filename: str) -> str:
    base = os.path.basename(filename)
    parts = base.split("-")
    return "-".join(parts[:3])


class FeatureOnlyDataset(Dataset):
    def __init__(
        self,
        features_path: str,
        *,
        bagsize: int = 0,
        sampling: str = "none",
        energy_mode: str = "l2_sq",
        energy_tau: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.features_path = str(features_path)
        self.pt_files = sorted(glob.glob(os.path.join(self.features_path, "*.pt")))
        if not self.pt_files:
            raise RuntimeError(f"No .pt files found in: {self.features_path}")

        self.bagsize = int(bagsize)
        self.sampling = str(sampling)
        self.energy_mode = str(energy_mode)
        self.energy_tau = float(energy_tau)
        self.seed = seed

        self._use_sampling = self.bagsize > 0 and self.sampling.lower() != "none"

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, idx: int):
        file_path = self.pt_files[idx]
        filename = os.path.basename(file_path)
        image_id = os.path.splitext(filename)[0]
        case_id = _infer_case_id(filename)

        try:
            features_obj = torch.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to torch.load feature file: {file_path} (case_id={case_id})") from e

        if not isinstance(features_obj, dict) or "feat" not in features_obj:
            raise RuntimeError(
                f"Feature file has unexpected format (expected dict with key 'feat'): {file_path} (case_id={case_id})"
            )

        features = features_obj["feat"]  # [L, D]

        if self._use_sampling:
            from sampling_utils import BagSamplingConfig, sample_bag

            cfg = BagSamplingConfig(
                bagsize=self.bagsize,
                strategy=self.sampling,
                energy_mode=self.energy_mode,
                energy_tau=self.energy_tau,
                seed=self.seed,
            )
            per_item_seed = None if self.seed is None else int(self.seed) + int(idx)
            features, _meta = sample_bag(features, config=cfg, per_item_seed=per_item_seed)

        return case_id, image_id, features


def collate_fn_pred(batch):
    max_len = max(b[2].shape[0] for b in batch)
    batch = [
        (
            cid,
            img_id,
            torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])).to(torch.bfloat16),
            torch.cat(
                [
                    torch.zeros(x.shape[0], dtype=torch.bool),
                    torch.ones(max_len - x.shape[0], dtype=torch.bool),
                ],
                dim=0,
            ),
        )
        for cid, img_id, x in batch
    ]
    return default_collate(batch)


def _namespace_from_ckpt(ckpt: dict) -> argparse.Namespace:
    args_dict = {}
    if isinstance(ckpt.get("args"), dict):
        args_dict.update(ckpt["args"])

    # Backfill required fields from checkpoint top-level if missing
    for key in ("model", "layers_nodes", "activation", "dropout", "attn_dim", "conv_kernel", "conv_blocks"):
        if key not in args_dict and key in ckpt:
            args_dict[key] = ckpt[key]

    return argparse.Namespace(**args_dict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict exposures (activities) from .pt features using a trained NB model")
    parser.add_argument("--features", required=True, help="Path to features directory with .pt files")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt) (e.g. checkpoints/hist2sig.pt)")
    parser.add_argument("--out_csv", required=True, help="Output CSV with Image ID, Case ID and exposures")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--exp_clip", type=float, default=None, help="Override exp_clip for exp(log_rate)")

    parser.add_argument("--bagsize", type=int, default=None, help="Default: same bagsize the checkpoint was trained with")
    parser.add_argument("--sampling", type=str, default=None, choices=["none", "random", "energy_topk", "energy_gumbel"], help="Default: same sampling strategy used for training")
    parser.add_argument("--energy_mode", type=str, default=None, choices=["l2", "l2_sq", "abs_mean"], help="Default: same as training")
    parser.add_argument("--energy_tau", type=float, default=None, help="Default: same as training")
    parser.add_argument("--sampling_seed", type=int, default=None, help="Default: same as training")

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint has unexpected type: {type(ckpt)}")

    activity_columns = ckpt.get("activity_columns")
    if not activity_columns:
        raise RuntimeError("Checkpoint is missing activity_columns; cannot label exposures")

    input_dim = int(ckpt.get("input_dim") or 0)
    if input_dim <= 0:
        raise RuntimeError("Checkpoint missing input_dim; cannot build model")

    k = int(ckpt.get("k") or len(activity_columns))

    ckpt_args = _namespace_from_ckpt(ckpt)

    # Ensure defaults are set if missing in checkpoint args
    if not hasattr(ckpt_args, "layers_nodes"):
        ckpt_args.layers_nodes = [256]
    if not hasattr(ckpt_args, "activation"):
        ckpt_args.activation = "gelu"
    if not hasattr(ckpt_args, "dropout"):
        ckpt_args.dropout = 0.0

    model = build_model(ckpt_args, input_dim=input_dim, k=k, device=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    exp_clip = float(args.exp_clip) if args.exp_clip is not None else float(ckpt_args.__dict__.get("exp_clip", 20.0))

    # Sampling defaults: match the bagging the checkpoint was trained with, unless overridden on the CLI.
    bagsize = int(args.bagsize) if args.bagsize is not None else int(ckpt_args.__dict__.get("bagsize", 0))
    sampling = str(args.sampling) if args.sampling is not None else str(ckpt_args.__dict__.get("sampling", "none"))
    energy_mode = str(args.energy_mode) if args.energy_mode is not None else str(ckpt_args.__dict__.get("energy_mode", "l2_sq"))
    energy_tau = float(args.energy_tau) if args.energy_tau is not None else float(ckpt_args.__dict__.get("energy_tau", 1.0))
    sampling_seed = int(args.sampling_seed) if args.sampling_seed is not None else int(ckpt_args.__dict__.get("sampling_seed", 42))

    print(f"Sampling: bagsize={bagsize} sampling={sampling} energy_mode={energy_mode} energy_tau={energy_tau} seed={sampling_seed}")

    dataset = FeatureOnlyDataset(
        args.features,
        bagsize=bagsize,
        sampling=sampling,
        energy_mode=energy_mode,
        energy_tau=energy_tau,
        seed=sampling_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_fn_pred,
    )

    all_case_ids: list[str] = []
    all_image_ids: list[str] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for case_ids, image_ids, x, mask in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred_log_a = model(x, mask)

            pred_log_a = pred_log_a.to(torch.float32).clamp(min=-float(exp_clip), max=float(exp_clip))
            pred_a = torch.exp(pred_log_a).cpu().numpy()

            all_case_ids.extend([str(c) for c in case_ids])
            all_image_ids.extend([str(i) for i in image_ids])
            all_pred.append(pred_a)

    pred_arr = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, len(activity_columns)), dtype=np.float32)

    df = pd.DataFrame(pred_arr, columns=[str(c) for c in activity_columns])
    df.insert(0, "Case ID", all_case_ids)
    df.insert(0, "Image ID", all_image_ids)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
