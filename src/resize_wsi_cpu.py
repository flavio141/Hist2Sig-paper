import os
import pyvips
import argparse
import pandas as pd

import sys
sys.path.append("utils")

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from resize_utils import resize_wsi_to_20x, force_memory_cleanup

parser = argparse.ArgumentParser(description="Resize WSI images to 20x magnification.")
parser.add_argument("--dataset", type=str, default="dataset_corrupted", help="Dataset folder")
parser.add_argument("--reduce_memory", type=bool, default=False, help="Reduce memory usage during processing.")
parser.add_argument("--delete_original", type=bool, default=False, help="Delete original WSI files after resizing.")
parser.add_argument("--status_file", type=str, default="resizing_status.csv", help="Path to the status file.")

def process_slide(row_dict, args, delete_original=False):
    slide_name = row_dict['slide_name']
    mpp_x = float(row_dict['mpp_x'])
    mpp_y = float(row_dict['mpp_y'])
    resolution = float(row_dict['resolution'])

    input_file = os.path.join(args.dataset, slide_name)
    output_file = os.path.join("dataset_20x", slide_name.replace(".svs", ".tiff"))

    if not any(input_file.endswith(suffix) for suffix in ['.svs', '.tif', '.tiff', '.ndpi']):
        return (slide_name, False, "Unsupported format", row_dict)

    if resolution not in [20, 40]:
        return (slide_name, False, "Resolution not 20x or 40x", row_dict)

    if os.path.exists(output_file):
        return (slide_name, True, "Already resized", row_dict)

    success = resize_wsi_to_20x(resolution, mpp_x, mpp_y, input_file, output_file)

    if success and delete_original:
        try:
            os.remove(input_file)
        except Exception as e:
            return (slide_name, False, f"Delete failed: {e}", row_dict)

    return (slide_name, success, "Processed" if success else "Resize failed", row_dict)

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists("slide_mpp_info.csv"):
        raise FileNotFoundError("slide_mpp_info.csv not found. Please ensure it exists in the current directory. Try running python src/check_slide.py first.")

    if args.reduce_memory:
        pyvips.cache_set_max_mem(100 * 1024 * 1024)
        pyvips.cache_set_max(50)

    os.makedirs("dataset_20x", exist_ok=True)
    print("Starting resizing slides to 20x...")

    slide_info = pd.read_csv("slide_mpp_info.csv")
    slide_dicts = slide_info.to_dict(orient='records')

    if os.path.exists(args.status_file):
        status_df = pd.read_csv(args.status_file)
        processed_slides = set(status_df[status_df["status"] == True]["slide_name"])
    else:
        status_df = pd.DataFrame(columns=["slide_name", "status", "message"])
        processed_slides = set()

    slides_to_process = [row for row in slide_dicts if row["slide_name"] not in processed_slides]

    processed_count = 0

    with open("failed_WSI.txt", "w") as f_fail, open("not_resized_WSI.csv", "w") as f_not_resized:
        f_fail.write("Failed WSI files:\n")
        f_not_resized.write(",".join(slide_info.columns) + "\n")

        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(process_slide, row_dict, args, args.delete_original)
                for row_dict in slides_to_process
            ]

            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slides"):
                slide_name, success, status, row = future.result()
                results.append({"slide_name": slide_name, "status": success, "message": status})

                if success:
                    processed_count += 1
                    if processed_count % 5 == 0:
                        force_memory_cleanup()
                else:
                    if status == "Resolution not 20x or 40x":
                        f_not_resized.write(",".join(str(row[col]) for col in slide_info.columns) + "\n")
                    else:
                        f_fail.write(f"{slide_name} - {status}\n")

    results_df = pd.DataFrame(results)
    updated_status_df = pd.concat([status_df, results_df], ignore_index=True)
    updated_status_df.drop_duplicates(subset="slide_name", keep="last", inplace=True)
    updated_status_df.to_csv(args.status_file, index=False)

    force_memory_cleanup()
    print(f"Processing completed. Total new images processed: {processed_count}")
