import os
import PIL
import h5py
import shutil
import requests
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2


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


"""
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
"""

from torch.utils.data import Dataset, DataLoader
class PatchDataset(Dataset):
    """A lightweight iterable over WSI patches"""
    def __init__(self, wsi, coords, level, size, transform):
        self.wsi, self.coords = wsi, coords
        self.level, self.size, self.transform = level, size, transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        img = self.wsi.read_region((int(x), int(y)),
                                   level=self.level,
                                   size=(self.size, self.size)).convert("RGB")
        return self.transform(img)

def create_features_optim(args, slide, wsi, model, transform, h5_path):
    """
    Batched, GPU-centric feature extraction:
      • loads patches lazily via a DataLoader
      • runs them in fp16 with autocast
      • concatenates everything in CPU RAM
      • *one* torch.save() per slide
    """
        
    out_file = os.path.join(args.features, '.'.join(slide.split('.')[:-1]) + '.pt')
    if os.path.exists(out_file):
        print(f"Slide {slide} already processed")
        return

    # 1. prepare dataset & dataloader
    with h5py.File(h5_path, "r") as h5f:
        coords = np.array(h5f["coords"])        # (N, 2)

    ds      = PatchDataset(wsi, coords,
                           level=args.level,
                           size=args.size,
                           transform=transform)
    loader  = DataLoader(ds,
                         batch_size=32,        # tune to fill your 15 GB GPU
                         shuffle=False,
                         num_workers=1,
                         pin_memory=True)

    # 2. forward pass in batches
    feats = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        for batch in tqdm(loader, desc=f"[{slide}] extracting"):
            batch = batch.to(args.cuda, non_blocking=True)
            emb   = model(batch).cpu()          # immediately back to CPU
            feats.append(emb)

    stacked = torch.cat(feats).half()     # shape (N, 1536)
    

    # ---------- save coords + features together ----------
    torch.save(
        {"coords": torch.as_tensor(coords, dtype=torch.int32),
         "feat":   stacked},
        out_file
    )


