#!/usr/bin/env python3
"""Export Case IDs present in a features folder, excluding files listed in bad_files.txt.

Default behavior matches Hist2Sig conventions:
- Feature files are `*.pt` under `--features_dir`.
- Case ID is derived from filename as the first 3 '-' separated fields.
- bad_files.txt contains lines like: `BAD: /path/to/<filename>.pt`.

Outputs a newline-separated list of unique Case IDs.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Set


def _parse_bad_feature_basenames(bad_files_path: Path) -> Set[str]:
    bad: Set[str] = set()
    if not bad_files_path.exists():
        return bad

    for line in bad_files_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("BAD:"):
            continue
        # Format: BAD: /abs/path/to/file.pt
        p = line.split(":", 1)[1].strip()
        if not p:
            continue
        bad.add(os.path.basename(p))
    return bad


def _case_id_from_feature_filename(filename: str) -> str:
    # Convention used across the repo.
    return "-".join(os.path.basename(filename).split("-")[:3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features_dir",
        type=str,
        default="/leonardo_work/CNHPC_2116672/features",
        help="Directory containing *.pt feature files",
    )
    ap.add_argument(
        "--bad_files",
        type=str,
        default="bad_files.txt",
        help="Path to bad_files.txt (supports relative to repo root)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="features_case_ids_not_bad.txt",
        help="Output path for Case ID list",
    )
    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    bad_files_path = Path(args.bad_files)
    out_path = Path(args.out)

    if not features_dir.exists():
        raise SystemExit(f"features_dir does not exist: {features_dir}")

    bad_basenames = _parse_bad_feature_basenames(bad_files_path)

    pt_files = sorted(glob.glob(str(features_dir / "*.pt")))
    case_ids: Set[str] = set()

    for p in pt_files:
        bn = os.path.basename(p)
        if bn in bad_basenames:
            continue
        case_ids.add(_case_id_from_feature_filename(bn))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(case_ids)) + ("\n" if case_ids else ""))

    print(f"features_dir: {features_dir}")
    print(f"bad_files: {bad_files_path} (bad feature files: {len(bad_basenames)})")
    print(f"pt_files scanned: {len(pt_files)}")
    print(f"unique Case IDs written: {len(case_ids)}")
    print(f"out: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
