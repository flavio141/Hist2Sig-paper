import os
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

from torchvision import transforms

import sys
sys.path.append('utils')

from features_utils_optimized import create_features_optim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


parser = argparse.ArgumentParser(description='Extract Features')
# Folders
parser.add_argument('--source', type=str, default='external_dataset', help='folder of the dataset')
parser.add_argument('--features', type=str, default='features_external', help='folder for the features')
parser.add_argument('--h5', type=str, default='result_dir/patches', help='folder for the patches to read')

# Settings
parser.add_argument('--model', type=str, default='h-optim', help='select the model to extract features') # Default: H-Optimus
parser.add_argument('--keep_feat', type=bool, default=True, help='keep features for each patch')
parser.add_argument('--level', type=int, default=0, help='level of the slide to extract features from')
parser.add_argument('--size', type=int, default=224, help='size of the patch')
parser.add_argument('--cuda', type=str, default='cuda:0', help='select the cuda device to use')




if __name__ == '__main__':
    args = parser.parse_args()

    slides = sorted(os.listdir(args.source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(args.source, slide))]
    slides = [slide for slide in slides if slide.split(".")[-1] in ["tiff", "ndpi", "svs", "tif"]]

    os.makedirs(args.features, exist_ok=True)

    if args.model == 'h-optim':
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=False
        )
        model.to("cuda")
        model.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])


    if torch.cuda.is_available():
       model.cuda(args.cuda)


    for i, slide in enumerate(slides):
        print("\n\nProgress: {:.2f}, {}/{}".format((i + 1) / len(slides), i + 1, len(slides)))
        file_path = os.path.join(args.source, slide)
        h5_path = os.path.join(args.h5, ".".join(slide.split('.')[:-1]) + '.h5')

        if not os.path.exists(h5_path):
            print(f'File not present: {h5_path}')
            continue
            

        print(file_path)
        wsi = openslide.OpenSlide(file_path)
        if args.model == 'h-optim':
            if os.path.exists(os.path.join(args.features, slide.replace('.tiff', '.pt'))):
                continue
            
            #os.makedirs(os.path.join(args.features, '.'.join(slide.split('.')[:-1])), exist_ok=True)
            create_features_optim(args, slide, wsi, model, transform, h5_path)
        # else:
        #     if os.path.exists(os.path.join(args.features, slide.split('.')[0] + '.pt')):
        #         continue

        #     os.makedirs(os.path.join(args.features, slide.split('.')[0]), exist_ok=True)
        #     create_features_virchow2(args, slide, model, h5_path)
