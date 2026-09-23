import os
import csv
import argparse

import sys
sys.path.append("utils")

from tqdm import tqdm
from check_utils import find_level_for_target_mpp

parser = argparse.ArgumentParser(description="Check slide properties and find appropriate level for target MPP.")
parser.add_argument("--local_dir", type=str, default="external_dataset", help="Directory containing the WSI files.")
parser.add_argument("--failed_dir", type=str, default="failed_WSI", help="Directory to save failed WSI files.")


if __name__ == "__main__":
    print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides")
    args = parser.parse_args()

    local_dir = args.local_dir
    target_mpp = 0.5
    output_csv = "slide_mpp_info.csv"
    failed_dir = args.failed_dir

    os.makedirs(failed_dir, exist_ok=True)

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_name", "resolution", "mpp_x", "mpp_y", "level_downsamples"])

        for wsi in tqdm(os.listdir(local_dir), desc='Analyzing slides'):
            if any(wsi.endswith(suffix) for suffix in ['.svs', '.tif', '.tiff', '.ndpi']):
                slide_path = os.path.join(local_dir, wsi)
                mpp_x, mpp_y, objective_power, level_downsamples, properties = find_level_for_target_mpp(slide_path, target_mpp)

                if objective_power is None:
                    fail_txt_path = os.path.join(failed_dir, f"{os.path.splitext(wsi)[0]}_properties.txt")
                    with open(fail_txt_path, "w") as pf:
                        for key, value in properties.items():
                            pf.write(f"{key}: {value}\n")
                    resolution = "N/A"
                else:
                    resolution = objective_power

                if level_downsamples is not None:
                    level_downsamples_str = ",".join([f"{ds:.4f}" for ds in level_downsamples])
                else:
                    level_downsamples_str = "N/A"

                writer.writerow([
                    wsi,
                    resolution,
                    #level if level is not None else "N/A",
                    mpp_x if mpp_x is not None else "N/A",
                    mpp_y if mpp_y is not None else "N/A",
                    level_downsamples_str
                ])
            else:
                print(f"Skipping non-slide file: {wsi}")
