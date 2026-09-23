import os
import PIL
import time
import h5py
import shutil
import requests
import openslide
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from tqdm import tqdm
from PIL import Image
from datetime import timedelta
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

embed_sizes={"dinov2_vits14": 384,
             "dinov2_vitb14": 768,
             "dinov2_vitl14": 1024,
             "dinov2_vitg14": 1536}


def trasform_img(target_img_size=224, normalize = False):
    trsforms = []

    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
	
    trsforms.append(transforms.ToTensor())
    if normalize == True:
        trsforms.append(transforms.Normalize())
    trsforms = transforms.Compose(trsforms)

    return trsforms


def tensor_to_image(img, trsforms, normalize = False):
    tensor_predict = trsforms(img)

    if normalize == True:
        tensor = (tensor_predict - tensor_predict.min()) / (tensor_predict.max() - tensor_predict.min())
    
    image_array = np.transpose(tensor_predict.numpy(), (1, 2, 0))
    
    tensor = image_array*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor), tensor_predict


def get_dino_bloom(modelpath = "models/DinoBloom-L.pth", 
                   modelname = "dinov2_vitl14", 
                   url = "https://zenodo.org/records/10908163/files/DinoBloom-L.pth?download=1"):
    
    model = torch.hub.load('facebookresearch/dinov2', modelname)

    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(modelpath):
        response = requests.get(url)
        with open(modelpath, 'wb') as f:
            f.write(response.content)
    
    pretrained = torch.load(modelpath, map_location=torch.device('cpu'))
    new_state_dict = {}

    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    pos_embed = nn.Parameter(torch.zeros(1, 257, embed_sizes[modelname]))
    model.pos_embed = pos_embed

    model.load_state_dict(new_state_dict, strict=True)
    return model


def create_features_dino(args, slide, model, trsforms, h5_path):    
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Extract Features with DinoBloom')
        
        if os.path.exists(os.path.join(args.features, '.'.join(slide.split('.')[:-1]))) and len(os.listdir(os.path.join(args.features, '.'.join(slide.split('.')[:-1])))) == len(f['coords']):
                iterator.close()
                print(f"Slide {'.'.join(slide.split('.')[:-1])} already processed")
        else:
            for idx in iterator:
                coord = f['coords'][idx]
                if not os.path.exists(f"{os.path.join(args.process, '.'.join(slide.split('.')[:-1]))}/{coord[0]}x_{coord[1]}y.jpg"):
                    continue
                patch = Image.open(f"{os.path.join(args.process, '.'.join(slide.split('.')[:-1]))}/{coord[0]}x_{coord[1]}y.jpg")
                tensor_predict = trsforms(patch)

                features = model.forward_features(tensor_predict.unsqueeze(0).cuda(args.cuda))
                torch.save(features['x_norm_clstoken'].clone(), f"{args.features}/{'.'.join(slide.split('.')[:-1])}/tensor_{f['coords'][idx][0]}_{f['coords'][idx][1]}.pt")
    
    try:
        file_list = [file for file in os.listdir(os.path.join(args.features, '.'.join(slide.split('.')[:-1]))) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f"{args.features}/{'.'.join(slide.split('.')[:-1])}", file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f"No files found in {os.path.join(args.features, '.'.join(slide.split('.')[:-1]))}: {error}")

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features, '.'.join(slide.split('.')[:-1])))

    torch.save(stacked_tensor.squeeze(dim=1), f"{args.features}/{'.'.join(slide.split('.')[:-1])}.pt")

def create_features_dino_not_normalized(args, slide, wsi, model, trsforms, h5_path, size=(256, 256), level=-1):    
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Extract Features with DinoBloom')
        
        for idx in iterator:
            image = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB").resize(size)
            tensor_predict = trsforms(image)
            
            features = model.forward_features(tensor_predict.unsqueeze(0).cuda(args.cuda))
            torch.save(features['x_norm_clstoken'].clone(), f"{args.features}/{'.'.join(slide.split('.')[:-1])}/tensor_{f['coords'][idx][0]}x_{f['coords'][idx][1]}y.pt")
    
    try:
        file_list = [file for file in os.listdir(os.path.join(args.features, '.'.join(slide.split('.')[:-1]))) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f"{args.features}/{'.'.join(slide.split('.')[:-1])}", file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f"No files found in {os.path.join(args.features, '.'.join(slide.split('.')[:-1]))}: {error}")

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features, '.'.join(slide.split('.')[:-1])))

    torch.save(stacked_tensor.squeeze(dim=1), f"{args.features}/{'.'.join(slide.split('.')[:-1])}.pt")


def create_features_giga(args, slide, model, h5_path):    
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Creating Patches and Extracting Features')
        if os.path.exists(os.path.join(args.features, '.'.join(slide.split('.')[:-1]))) and len(os.listdir(os.path.join(args.features, ".".join(slide.split('.')[:-1])))) == len(f['coords']):
                iterator.close()
                print(f"Slide {'.'.join(slide.split('.')[:-1])} already processed")
        else:
            for idx in iterator:
                coord = f['coords'][idx]
                patch = Image.open(f"{os.path.join(args.process, '.'.join(slide.split('.')[:-1]))}/{coord[0]}x_{coord[1]}y.jpg")
                model.eval()
                with torch.no_grad():
                    features = model(v2.functional.to_tensor(patch).unsqueeze(0).cuda(args.cuda))
                torch.save(features.clone(), f"{args.features}/{'.'.join(slide.split('.')[:-1])}/tensor_{coord[0]}x_{coord[1]}y.pt")

    try:
        file_list = [file for file in os.listdir(os.path.join(args.features, '.'.join(slide.split('.')[:-1]))) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f"{args.features}/{'.'.join(slide.split('.')[:-1])}", file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f"No files found in {os.path.join(args.features, '.'.join(slide.split('.')[:-1]))}: {error}")

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features, '.'.join(slide.split('.')[:-1])))

    torch.save(stacked_tensor.squeeze(dim=1), f"{args.features}/{'.'.join(slide.split('.')[:-1])}.pt")


def normalize(args, wsi, h5_path, slide, stain_normalizer, level=0, size=256):
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Normalizing')
        
        for idx in iterator:
            coord = f['coords'][idx]
            if len(os.listdir(os.path.join(args.process, ".".join(slide.split('.')[:-1])))) == len(f['coords']):
                iterator.close()
                print(f"Slide {'.'.join(slide.split('.')[:-1])} already normalized")
                break
            
            if os.path.exists(f"{os.path.join(args.process, '.' .join(slide.split('.')[:-1]))}/{coord[0]}x_{coord[1]}y.jpg"):
                continue

            image = wsi.read_region(location=coord, level=level, size=[size, size])
            normed_sample = stain_normalizer.transform(image.copy())

            patch = Image.fromarray(normed_sample)
            patch.save(f"{os.path.join(args.process, '.'.join(slide.split('.')[:-1]))}/{coord[0]}x_{coord[1]}y.jpg")
            patch.save(f"{os.path.join(args.process, '.'.join(slide.split('.')[:-1]))}/{coord[0]}x_{coord[1]}y.jpg")


def normalize_gpu(args, wsi, h5_path, slide, stain_normalizer, level=0, size=256):
    device = torch.device(f"{args.cuda}" if torch.cuda.is_available() else "cpu")

    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Normalizing')

        slide_name = ".".join(slide.split('.')[:-1])
        slide_dir = os.path.join(args.process, slide_name)
        os.makedirs(slide_dir, exist_ok=True)

        for idx in iterator:
            coord = f['coords'][idx]
            save_path = os.path.join(slide_dir, f"{coord[0]}x_{coord[1]}y.jpg")

            if os.path.exists(save_path):
                continue

            if len(os.listdir(slide_dir)) == len(f['coords']):
                print(f"Slide {slide_name} already normalized")
                break

            image = wsi.read_region(location=coord, level=level, size=[size, size]).convert("RGB")
            img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
            norm_tensor = stain_normalizer(img_tensor)

            norm_img = transforms.ToPILImage()(norm_tensor.squeeze(0).cpu())
            norm_img.save(save_path)


def create_features_optim(args, slide, wsi, model, trsforms, h5_path):    
    with h5py.File(h5_path, "r") as f:
        iterator = tqdm(range(len(f['coords'])), desc='Extract Features with H-Optimus')
        
        if os.path.exists(os.path.join(args.features, '.'.join(slide.split('.')[:-1]))) and len(os.listdir(os.path.join(args.features, '.'.join(slide.split('.')[:-1])))) == len(f['coords']):
                iterator.close()
                print(f"Slide {'.'.join(slide.split('.')[:-1])} already processed")
        else:
            for idx in iterator:
                coord = f['coords'][idx]
                image = wsi.read_region(location=coord, level=args.level, size=[args.size, args.size]).convert("RGB")
                #image = transforms.ToPILImage()(image)

                with torch.autocast(device_type=args.cuda, dtype=torch.float16):
                    with torch.inference_mode():
                        features = model(trsforms(image).unsqueeze(0).to(args.cuda))
                torch.save(features.clone(), f"{args.features}/{'.'.join(slide.split('.')[:-1])}/tensor_{coord[0]}_{coord[1]}.pt")

    try:
        file_list = [file for file in os.listdir(os.path.join(args.features, '.'.join(slide.split('.')[:-1]))) if file.endswith('.pt')]
        stacked_tensor = torch.cat([torch.load(os.path.join(f"{args.features}/{'.'.join(slide.split('.')[:-1])}", file)) for file in file_list], dim=0)
    except FileNotFoundError as error:
        print(f"No files found in {os.path.join(args.features, '.'.join(slide.split('.')[:-1]))}: {error}")

    if not args.keep_feat:
        shutil.rmtree(os.path.join(args.features, '.'.join(slide.split('.')[:-1])))

    torch.save(stacked_tensor.squeeze(dim=1), f"{args.features}/{'.'.join(slide.split('.')[:-1])}.pt")


# 1. Prova di Parallelizzazione
class WSIPatchDataset(Dataset):
    def __init__(self, wsi_path, coords, level, size, transform=None):
        self.wsi_path = wsi_path
        self.coords = coords
        self.level = level
        self.size = size
        self.transform = transform
        self._wsi = None

    def _init_wsi(self):
        if self._wsi is None:
            self._wsi = openslide.OpenSlide(self.wsi_path)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        self._init_wsi()
        coord = self.coords[idx]
        image = self._wsi.read_region(location=coord, level=self.level, size=[self.size, self.size]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, tuple(coord)


def create_features_optim_cpu(args, slide, wsi, model, transform, h5_path):    
    with h5py.File(h5_path, "r") as f:
        coords = list(f['coords'])

    slide_name = '.'.join(slide.split('.')[:-1])
    out_dir = os.path.join(args.features, slide_name)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(f"{args.features}/{slide_name}.pt"):
        print(f"{slide_name}.pt already exists, skipping")
        return

    dataset = WSIPatchDataset(wsi, coords, args.level, args.size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    futures = []
    executor = ThreadPoolExecutor(max_workers=16)

    saved_files = []

    iterator = tqdm(dataloader, desc=f"[{slide_name}] Extracting features")

    for images, coords_batch in iterator:
        images = images.to(args.cuda, non_blocking=True)

        with torch.autocast(device_type=args.cuda, dtype=torch.float16):
            with torch.inference_mode():
                features = model(images)

        for feat, coord in zip(features, coords_batch):
            x, y = coord[0], coord[1]
            filename = f"{out_dir}/tensor_{x}_{y}.pt"
            saved_files.append(filename)
            futures.append(executor.submit(torch.save, feat.clone().cpu(), filename))

    [f.result() for f in futures]
    executor.shutdown()

    try:
        tensors = [torch.load(fp) for fp in saved_files if os.path.exists(fp)]
        stacked_tensor = torch.cat(tensors, dim=0)
    except Exception as e:
        print(f"Error loading individual tensors: {e}")
        return

    if not args.keep_feat:
        shutil.rmtree(out_dir)

    torch.save(stacked_tensor.squeeze(dim=1), f"{args.features}/{slide_name}.pt")



# 2. Prova di Parallelizzazione con mp.spawn
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def process_and_save_features(rank, wsi_path, chunks, level, size, transform, model, device, out_dir):
    coords = chunks[rank]
    wsi = openslide.OpenSlide(wsi_path)

    current_device = torch.device(device) if rank == 0 else torch.device("cpu")
    model = model.to(current_device).eval()

    total = len(coords)
    print(f"[Rank {rank}] Starting with {total} patches.")

    start_time = time.time()

    for idx, coord in enumerate(coords, 1):
        image = wsi.read_region(location=coord, level=level, size=[size, size]).convert("RGB")
        if transform:
            image = transform(image)
        image = image.unsqueeze(0).to(current_device)

        with torch.inference_mode():
            if current_device.type == device:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    feat = model(image)
            else:
                feat = model(image)

        x, y = coord[0], coord[1]
        filename = f"{out_dir}/tensor_{x}_{y}.pt"
        torch.save(feat.squeeze(0).cpu(), filename)

        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = elapsed / idx
            eta = rate * (total - idx)
            print(f"[Rank {rank}] {idx}/{total} patches | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")


def create_features_parallel(args, slide, path, model, transform, h5_path):
    with h5py.File(h5_path, "r") as f:
        coords = list(f['coords'])

    slide_name = '.'.join(slide.split('.')[:-1])
    out_dir = os.path.join(args.features, slide_name)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(f"{args.features}/{slide_name}.pt"):
        print(f"{slide_name}.pt already exists, skipping")
        return

    wsi_path = path

    num_processes = min(8, mp.cpu_count())
    chunks = [coords[i::num_processes] for i in range(num_processes)]

    model.share_memory()

    mp.spawn(
        process_and_save_features,
        args=(wsi_path, chunks, args.level, args.size, transform, model, args.cuda, out_dir),
        nprocs=num_processes
    )

    try:
        tensors = [torch.load(os.path.join(out_dir, fname))
                   for fname in os.listdir(out_dir) if fname.endswith(".pt")]
        stacked_tensor = torch.stack(tensors, dim=0)
    except Exception as e:
        print(f"Error loading individual tensors: {e}")
        return

    if not args.keep_feat:
        shutil.rmtree(out_dir)

    torch.save(stacked_tensor, f"{args.features}/{slide_name}.pt")


# 4. Ultima prova di parallelizzazione
class PatchDataset(Dataset):
    def __init__(self, wsi, coords, level, size, transform):
        self.wsi = wsi
        self.coords = coords
        self.level = level
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        image = self.wsi.read_region(location=coord, level=self.level, size=(self.size, self.size)).convert("RGB")
        image = self.transform(image)
        return image


class PatchDatasetLast(Dataset):
    def __init__(self, slide_path, coords, level, size, transform):
        self.slide_path = slide_path
        self.coords = coords
        self.level = level
        self.size = size
        self.transform = transform
        self.wsi = None

    def _init_wsi(self):
        if self.wsi is None:
            self.wsi = openslide.OpenSlide(self.slide_path)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        self._init_wsi()
        coord = self.coords[idx]
        image = self.wsi.read_region(location=coord, level=self.level, size=(self.size, self.size)).convert("RGB")
        image = self.transform(image)
        return image


def create_features_optim_last(args, slide, wsi, model, trsforms, h5_path):    
    slide_id = '.'.join(slide.split('.')[:-1])
    output_path = os.path.join(args.features, f"{slide_id}.pt")

    if os.path.exists(output_path):
        print(f"Slide {slide_id} already processed")
        return

    with h5py.File(h5_path, "r") as f:
        coords = f['coords'][:]
    
    #dataset = PatchDataset(wsi=wsi, coords=coords, level=args.level, size=args.size, transform=trsforms)
    dataset = PatchDatasetLast(slide_path=os.path.join(args.source, slide), coords=coords, level=args.level, size=args.size, transform=trsforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    all_features = []

    model.eval()
    with torch.autocast(device_type=args.cuda, dtype=torch.float16):
        with torch.inference_mode():
            for images in tqdm(dataloader, desc=f"Extracting features from {slide_id}"):
                images = images.to(args.cuda, non_blocking=True)
                features = model(images)

                all_features.append(features.cpu())

                del images, features

    stacked_tensor = torch.cat(all_features, dim=0).half()

    torch.save(
        {"coords": torch.as_tensor(coords, dtype=torch.int32),
         "feat":   stacked_tensor},
        output_path
    )
    print(f"Saved features for {slide_id} to {output_path}")

    del stacked_tensor, coords
    torch.cuda.empty_cache()