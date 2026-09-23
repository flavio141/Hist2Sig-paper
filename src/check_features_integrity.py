import argparse
import glob
import os
import sys

import torch


def iter_feature_files(features_dir: str):
    pattern = os.path.join(features_dir, "*.pt")
    for path in sorted(glob.glob(pattern)):
        yield path


def load_already_checked(log_path: str) -> set[str]:
    """Parses an existing log file and returns the set of feature file paths already checked.

    Expected line formats:
      OK  <path>
      BAD <path>\t<error>
    """
    checked: set[str] = set()
    if not log_path:
        return checked
    if not os.path.exists(log_path):
        return checked

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("OK "):
                checked.add(line[3:])
            elif line.startswith("BAD "):
                # "BAD <path>\t..."
                rest = line[4:]
                path = rest.split("\t", 1)[0]
                checked.add(path)
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Checks that .pt feature files can be loaded by torch.load and contain the expected key 'feat'."
    )
    parser.add_argument("--features", required=True, help="Path to directory containing .pt feature files")
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If > 0, only checks the first N files (sorted).",
    )
    parser.add_argument(
        "--log",
        default="",
        help=(
            "Optional log file. If provided, writes one line per checked file (OK/BAD). "
            "Re-running with the same --log will skip already checked files (resume)."
        ),
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=50,
        help="Flush log output every N checked files (0 disables periodic flush).",
    )
    args = parser.parse_args()

    features_dir = args.features
    if not os.path.isdir(features_dir):
        print(f"ERROR: features directory does not exist: {features_dir}")
        return 2

    total = 0
    ok = 0
    bad = 0

    checked_already = load_already_checked(args.log)
    if checked_already:
        print(f"Resuming: skipping {len(checked_already)} already checked files from {args.log}")

    log_fh = None
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        log_fh = open(args.log, "a", encoding="utf-8")

    for path in iter_feature_files(features_dir):
        if path in checked_already:
            continue

        total += 1
        if args.max_files and total > args.max_files:
            break

        try:
            obj = torch.load(path, map_location="cpu")
            if not isinstance(obj, dict) or "feat" not in obj:
                raise RuntimeError("unexpected format (expected dict with key 'feat')")
        except Exception as e:
            bad += 1
            msg = f"{type(e).__name__}: {e}"
            print(f"BAD: {path}\n  -> {msg}")
            if log_fh is not None:
                log_fh.write(f"BAD {path}\t{msg}\n")
            continue

        ok += 1
        if log_fh is not None:
            log_fh.write(f"OK {path}\n")

        if ok % 500 == 0:
            print(f"Checked {ok} OK files...")

        if log_fh is not None and args.flush_every and (total % args.flush_every == 0):
            log_fh.flush()

    if log_fh is not None:
        log_fh.flush()
        log_fh.close()

    print(f"DONE. total_checked={total}, ok={ok}, bad={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
