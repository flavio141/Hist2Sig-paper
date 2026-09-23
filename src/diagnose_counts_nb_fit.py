import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_to_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.loc[:, cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(0.0)
    return out


def _summarize_counts(x: np.ndarray, *, name: str) -> dict[str, float | str]:
    # x is 1D float array of counts
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"name": name, "n": 0}

    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else 0.0
    zero_frac = float(np.mean(x == 0.0))

    # Dispersion index (Poisson would be ~1 when mean>0)
    vmr = float(var / mean) if mean > 0 else float("nan")

    # Method-of-moments NB2 theta: var = mean + mean^2/theta
    # => theta = mean^2 / (var - mean)
    if mean > 0 and var > mean:
        theta_mom = float((mean * mean) / (var - mean))
    else:
        theta_mom = float("inf")

    # Expected zeros under Poisson(mean)
    p0_pois = float(np.exp(-mean))

    # Expected zeros under NB(mean, theta)
    # P0 = (theta/(theta+mu))^theta; when theta->inf, tends to exp(-mu)
    if np.isfinite(theta_mom) and theta_mom > 0 and mean > 0:
        p0_nb = float((theta_mom / (theta_mom + mean)) ** theta_mom)
    else:
        p0_nb = p0_pois

    excess_zero_vs_pois = float(zero_frac - p0_pois)
    excess_zero_vs_nb = float(zero_frac - p0_nb)

    return {
        "name": name,
        "n": n,
        "mean": mean,
        "var": var,
        "vmr": vmr,
        "zero_frac": zero_frac,
        "p0_pois": p0_pois,
        "p0_nb_mom": p0_nb,
        "excess_zero_vs_pois": excess_zero_vs_pois,
        "excess_zero_vs_nb_mom": excess_zero_vs_nb,
        "theta_mom": theta_mom,
    }


def _pick_activity_columns(df: pd.DataFrame) -> list[str]:
    # Match repo logic: prefer SBS* columns.
    sbs_cols = [c for c in df.columns if str(c).startswith("SBS")]
    if sbs_cols:
        return sbs_cols

    # Fallback: everything after Project ID
    if "Project ID" in df.columns:
        idx = df.columns.get_loc("Project ID")
        cols = list(df.columns[idx + 1 :])
        if cols:
            return cols

    # Last resort: drop obvious metadata
    drop = {"Case ID", "Project ID"}
    return [c for c in df.columns if c not in drop]


def _pick_profile_columns(df: pd.DataFrame) -> list[str]:
    if "Project ID" in df.columns:
        idx = df.columns.get_loc("Project ID")
        cols = list(df.columns[idx + 1 :])
        if cols:
            return cols

    # Fallback: drop obvious metadata
    drop = {"Case ID", "Project ID"}
    return [c for c in df.columns if c not in drop]


def _summarize_table(
    df: pd.DataFrame,
    cols: list[str],
    *,
    label: str,
    min_mean: float,
) -> pd.DataFrame:
    xdf = _safe_to_numeric_frame(df, cols)

    rows: list[dict[str, float | str]] = []
    for c in cols:
        x = xdf[c].to_numpy(dtype=np.float64, copy=False)
        row = _summarize_counts(x, name=str(c))
        row["group"] = label
        rows.append(row)

    out = pd.DataFrame(rows)
    if "mean" in out.columns:
        out = out.loc[(out["mean"].fillna(0.0) >= float(min_mean))].copy()
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Diagnose whether Poisson vs Negative Binomial is appropriate for your counts by "
            "computing mean/variance, var-to-mean ratio, and zero inflation indicators per column."
        )
    )

    p.add_argument(
        "--counts_dir",
        type=str,
            default=str(Path("..") / "DB_Signatures"),
        help="Directory containing Activities_Sig.csv and Mutational_profile.csv",
    )
    p.add_argument("--activities_csv", type=str, default=None)
    p.add_argument("--mut_profile_csv", type=str, default=None)

    p.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["activities", "profiles", "both"],
        help="Which tables to analyze.",
    )

    p.add_argument(
        "--min_mean",
        type=float,
        default=0.0,
        help="Drop columns with mean < min_mean (useful to ignore nearly-all-zero columns).",
    )

    p.add_argument(
        "--out_csv",
        type=str,
        default=str(Path("results_rag") / "nb_diagnostics.csv"),
        help="Where to write the per-column summary CSV.",
    )

    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many top columns to print for a few rankings.",
    )

    args = p.parse_args()

    counts_dir = Path(args.counts_dir)
    act_path = Path(args.activities_csv) if args.activities_csv else (counts_dir / "Activities_Sig.csv")
    prof_path = Path(args.mut_profile_csv) if args.mut_profile_csv else (counts_dir / "Mutational_profile.csv")

    out_frames: list[pd.DataFrame] = []

    if args.which in {"activities", "both"}:
        act_df = pd.read_csv(act_path)
        if "Case ID" not in act_df.columns:
            raise ValueError(f"Expected 'Case ID' column in {act_path}")
        act_cols = _pick_activity_columns(act_df)
        out_frames.append(
            _summarize_table(act_df, act_cols, label="activities", min_mean=float(args.min_mean))
        )

    if args.which in {"profiles", "both"}:
        prof_df = pd.read_csv(prof_path)
        if "Case ID" not in prof_df.columns:
            raise ValueError(f"Expected 'Case ID' column in {prof_path}")
        prof_cols = _pick_profile_columns(prof_df)
        out_frames.append(
            _summarize_table(prof_df, prof_cols, label="profiles", min_mean=float(args.min_mean))
        )

    out = pd.concat(out_frames, axis=0, ignore_index=True) if out_frames else pd.DataFrame()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # Quick console summary
    top_k = int(args.top_k)
    if not out.empty:
        def _show(title: str, sort_col: str, ascending: bool):
            if sort_col not in out.columns:
                return
            tmp = out.sort_values(sort_col, ascending=ascending).head(top_k)
            cols = ["group", "name", "mean", "var", "vmr", "zero_frac", "theta_mom", sort_col]
            cols = [c for c in cols if c in tmp.columns]
            print("\n==", title, "==")
            print(tmp.loc[:, cols].to_string(index=False))

        _show("Most overdispersed (high var/mean)", "vmr", ascending=False)
        _show("Most excess zeros vs Poisson", "excess_zero_vs_pois", ascending=False)
        _show("Most excess zeros vs MOM-NB", "excess_zero_vs_nb_mom", ascending=False)

    print(f"\nWrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
