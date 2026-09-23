import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def copy_files(files, source_dir, dest_dir):
    for file in tqdm(files, desc='Copying Images'):
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        if os.path.exists(dest_path):
            continue
        shutil.copy(src_path, dest_path)


def copy_directories(dirs, source_dir, dest_dir):
    for dir_name in tqdm(dirs, desc='Copying Folders'):
        src_path = os.path.join(source_dir, dir_name)
        dest_path = os.path.join(dest_dir, dir_name)
        if os.path.exists(dest_path):
            continue
        shutil.copytree(src_path, dest_path)


if __name__ == '__main__':
    dataset_dir = 'dataset'
    processed_images_dir = 'processed_images'

    dataset_A_dir = 'dataset_A'
    dataset_B_dir = 'dataset_B'
    processed_images_A_dir = 'processed_images_A'
    processed_images_B_dir = 'processed_images_B'

    os.makedirs(dataset_A_dir, exist_ok=True)
    os.makedirs(dataset_B_dir, exist_ok=True)
    os.makedirs(processed_images_A_dir, exist_ok=True)
    os.makedirs(processed_images_B_dir, exist_ok=True)

    dataset_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    dataset_A_files, dataset_B_files = train_test_split(dataset_files, test_size=0.5, random_state=42)

    copy_files(dataset_A_files, dataset_dir, dataset_A_dir)
    copy_files(dataset_B_files, dataset_dir, dataset_B_dir)

    processed_dirs = [d for d in os.listdir(processed_images_dir) if os.path.isdir(os.path.join(processed_images_dir, d))]

    processed_images_A_dirs = [d for d in processed_dirs if d in [f.split('.')[0] for f in dataset_A_files]]
    processed_images_B_dirs = [d for d in processed_dirs if d in [f.split('.')[0] for f in dataset_B_files]]

    copy_directories(processed_images_A_dirs, processed_images_dir, processed_images_A_dir)
    copy_directories(processed_images_B_dirs, processed_images_dir, processed_images_B_dir)
