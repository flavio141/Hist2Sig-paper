from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _metadata_frame(act_df: pd.DataFrame, *, activity_cols: list[str], case_ids: list[str]) -> pd.DataFrame:
    if "Case ID" not in act_df.columns:
        return pd.DataFrame({"Case ID": [str(c) for c in case_ids]})

    meta_cols = [c for c in act_df.columns if c not in set(activity_cols)]
    meta = act_df.drop_duplicates(subset=["Case ID"]).loc[:, meta_cols].copy()
    meta["Case ID"] = meta["Case ID"].astype(str)
    meta = meta.set_index("Case ID", drop=False)

    rows = []
    for cid in case_ids:
        cid = str(cid)
        if cid in meta.index:
            rows.append(meta.loc[cid])
        else:
            rows.append(pd.Series({"Case ID": cid}))

    out = pd.DataFrame(rows)
    out = out.loc[:, ~out.columns.duplicated()]
    if "Case ID" not in out.columns:
        out.insert(0, "Case ID", [str(c) for c in case_ids])
    else:
        out["Case ID"] = [str(c) for c in case_ids]
        cols = ["Case ID"] + [c for c in out.columns if c != "Case ID"]
        out = out.loc[:, cols]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Export refit test predictions NPZ to CSV with metadata.")
    p.add_argument("--npz", required=True, help="Path to *_refit_test_preds.npz")
    p.add_argument("--counts_dir", required=True, help="Directory containing Activities_Sig.csv and Mutational_profile.csv")
    p.add_argument("--out_dir", required=True, help="Output directory")
    args = p.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path, allow_pickle=True)
    case_ids = [str(x) for x in z["case_ids"].tolist()]

    activity_cols = [str(x) for x in z["activity_columns"].tolist()]
    y_act_true = z["y_true"].astype(np.float32, copy=False)
    y_act_pred = z["y_pred"].astype(np.float32, copy=False)

    profile_cols = [str(x) for x in z["profile_columns"].tolist()]
    y_prof_true = z["y_profile_true"].astype(np.float32, copy=False)
    y_prof_pred = z["y_profile_pred"].astype(np.float32, copy=False)

    counts_dir = Path(args.counts_dir)
    act_df = pd.read_csv(counts_dir / "Activities_Sig.csv")

    meta_df = _metadata_frame(act_df, activity_cols=activity_cols, case_ids=case_ids)

    act_out = meta_df.copy()
    for j, name in enumerate(activity_cols):
        act_out[f"true_{name}"] = y_act_true[:, j]
        act_out[f"pred_{name}"] = y_act_pred[:, j]
    act_out.to_csv(out_dir / "test_predictions_activities.csv", index=False)

    prof_out = meta_df.copy()
    for j, name in enumerate(profile_cols):
        prof_out[f"true_{name}"] = y_prof_true[:, j]
        prof_out[f"pred_{name}"] = y_prof_pred[:, j]
    prof_out.to_csv(out_dir / "test_predictions_profiles.csv", index=False)

    print(f"Wrote CSVs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
