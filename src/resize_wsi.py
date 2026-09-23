import os
import gc
import pyvips
import shutil
import psutil
import argparse
import pandas as pd

import sys
sys.path.append("utils")

from tqdm import tqdm
from resize_utils import resize_wsi_to_20x, force_memory_cleanup


parser = argparse.ArgumentParser(description="Resize WSI images to 20x magnification.")
parser.add_argument("--reduce_memory", type=bool, default=False, help="Reduce memory usage during processing.")
parser.add_argument("--delete_original", type=bool, default=False, help="Delete original WSI files after resizing.")


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

    processed_count = 0

    with open("failed_WSI.txt", "w") as f_fail, open("not_resized_WSI.csv", "w") as f_not_resized:
        f_fail.write("Failed WSI files:\n")
        f_not_resized.write(",".join(slide_info.columns) + "\n")

        for idx, row in tqdm(slide_info.iterrows(), total=len(slide_info), desc='Processing slides'):
            process = psutil.Process(os.getpid())

            slide_name = row['slide_name']
            mpp_x = row['mpp_x']
            mpp_y = row['mpp_y']
            resolution = row['resolution']

            input_file = os.path.join("dataset", slide_name)
            output_file = os.path.join("dataset_20x", slide_name.replace(".svs", ".tiff"))

            if not any(input_file.endswith(suffix) for suffix in ['.svs', '.tif', '.tiff', '.ndpi']):
                continue

            if resolution == 40 or resolution == 20:
                if os.path.exists(output_file):
                    continue
                
                output = resize_wsi_to_20x(resolution, mpp_x, mpp_y, input_file, output_file)
                if idx % 10 == 0:
                    mb = process.memory_info().rss / (1024 * 1024)
                    print(f"Image number {idx}, {mb} MB")
                
                if output:
                    if args.delete_original:
                        os.remove(input_file)
                    processed_count += 1
                    
                    if processed_count % 5 == 0:
                        force_memory_cleanup()
                        print(f"Memory cleanup performed after {processed_count} images")
                else:
                    f_fail.write(f"{input_file}\n")
                    
            else:
                f_not_resized.write(",".join(str(row[col]) for col in slide_info.columns) + "\n")
                #shutil.move(input_file, output_file.replace(".tiff", ".svs"))
            
            if idx % 2 == 0:
                gc.collect()
    
    force_memory_cleanup()
    print(f"Processing completed. Total images processed: {processed_count}")
