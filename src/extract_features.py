import os
import cv2
import timm

import torch

import argparse
import openslide

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

from tiatoolbox import data
#from tiatoolbox.tools import stainnorm
#from tiatoolbox.wsicore import wsireader

from torchvision.transforms import ToTensor
from torch_staintools.normalizer import NormalizerBuilder


import sys
sys.path.append('utils')

from features_utils import get_dino_bloom, trasform_img, normalize_gpu, create_features_dino, create_features_dino_not_normalized, create_features_giga, normalize

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Extract Features')
# Folders
parser.add_argument('--source', type=str, default='dataset_20x', help='folder of the dataset')
parser.add_argument('--process', type=str, default='processed_images', help='folder for the processed features')
parser.add_argument('--features', type=str, default='features', help='folder for the features')
parser.add_argument('--h5', type=str, default='result_dir/patches', help='folder for the patches to read')

# Settings
parser.add_argument('--model', type=str, default='', help='select the model to extract features') # Default: GigaPath
parser.add_argument('--keep_feat', type=bool, default=True, help='keep features for each patch')
parser.add_argument('--normalize', type=str, default=True, help='tell if the images should be normalized before feature extraction')
parser.add_argument('--use_normalized', type=bool, default=True, help='use normalized images for feature extraction')
parser.add_argument('--cuda', type=str, default='cuda:0', help='select the cuda device to use')




if __name__ == '__main__':
    args = parser.parse_args()

    slides = sorted(os.listdir(args.source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(args.source, slide))]
    slides = [slide for slide in slides if slide.split(".")[-1] in ["tiff", "ndpi", "svs", "tif"]]

    print(f'Creating Folder -> {args.process}')
    
    os.makedirs(args.process, exist_ok=True)

    if args.model == 'dino':
        os.makedirs(args.features, exist_ok=True)
        model = get_dino_bloom()
        trsforms = trasform_img()
    else:
        os.makedirs(args.features, exist_ok=True)
        #model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    if args.normalize:
        target_image = data.stain_norm_target()
        device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
        # stain_normalizer = stainnorm.VahadaneNormalizer()
        # stain_normalizer.fit(target_image)

        target = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_tensor = ToTensor()(target).unsqueeze(0).to(device)

        stain_normalizer = NormalizerBuilder.build('reinhard', concentration_method='ista')
        stain_normalizer = stain_normalizer.to(device)
        stain_normalizer.fit(target_tensor)


    #if torch.cuda.is_available():
    #    model.cuda(args.cuda)


    for i, slide in enumerate(slides):
        print("\n\nProgress: {:.2f}, {}/{}".format((i + 1) / len(slides), i + 1, len(slides)))
        file_path = os.path.join(args.source, slide)
        h5_path = os.path.join(args.h5, ".".join(slide.split('.')[:-1]) + '.h5')

        if not os.path.exists(h5_path):
            print(f'File not present: {h5_path}')
            continue

        if args.normalize:
            print(file_path)
            wsi = openslide.OpenSlide(file_path)#wsireader.OpenSlideWSIReader.open(file_path)
            os.makedirs(os.path.join(args.process, ".".join(slide.split('.')[:-1])), exist_ok=True)
            normalize_gpu(args, wsi, h5_path, slide, stain_normalizer, level=0)
            

        # print(file_path)
        # if args.model == 'dino' and args.use_normalized:
        #     if os.path.exists(os.path.join(args.features, slide.split('.')[0] + '.pt')) or not os.path.exists(os.path.join(args.process, slide.split('.')[0])):
        #         continue
            
        #     os.makedirs(os.path.join(args.features, slide.split('.')[0]), exist_ok=True)
        #     create_features_dino(args, slide, model, trsforms, h5_path)
        # elif args.model == 'dino' and not args.use_normalized:
        #     if os.path.exists(os.path.join(args.features, slide.split('.')[0] + '.pt')) or not os.path.exists(h5_path):
        #         continue

        #     wsi = openslide.open_slide(file_path)
        #     os.makedirs(os.path.join(args.features, slide.split('.')[0]), exist_ok=True)
        #     create_features_dino_not_normalized(args, slide, wsi, model, trsforms, h5_path)
        # else:
        #     if os.path.exists(os.path.join(args.features, slide.split('.')[0] + '.pt')):
        #         continue

        #     os.makedirs(os.path.join(args.features, slide.split('.')[0]), exist_ok=True)
        #     create_features_giga(args, slide, model, h5_path)

