#!/usr/bin/env python3
import os
import sys
import argparse

import numpy as np
from PIL import Image
import torch
from skimage import color

# allow import of ColorNet
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_ROOT)
from faces.model import ColorNet

def load_model(model_path, device):
    model = ColorNet().to(device)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def preprocess(img_path, size):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    m   = min(w,h)
    img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    img = img.resize((size, size), Image.BILINEAR)

    rgb = np.array(img) / 255.0
    lab = color.rgb2lab(rgb).astype(np.float32)
    L   = lab[:,:,0] / 100.0
    return img, torch.from_numpy(L).unsqueeze(0).unsqueeze(0)

def postprocess(L, ab):
    lab = torch.cat([L, ab], dim=1)[0].permute(1,2,0).cpu().numpy()
    lab[:,:,0]  *= 100
    lab[:,:,1:] *= 128
    rgb = color.lab2rgb(lab)
    return Image.fromarray((rgb*255).astype(np.uint8))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',  required=True, help='Input RGB image')
    p.add_argument('-o','--output', default='out.png',
                   help='Where to save colorized result')
    p.add_argument('-m','--model',
                   default=os.path.abspath(
                     os.path.join(SRC_ROOT, '..', 'models', 'faces.pt')
                   ),
                   help='Path to best_model.pt')
    p.add_argument('-s','--size', type=int, default=128,
                   help='Training image size (e.g. 128)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.model, device)

    orig, L = preprocess(args.input, args.size)
    L = L.to(device)
    with torch.no_grad():
        ab = model(L)

    out = postprocess(L, ab)
    out.save(args.output)
    print(f"✔ Saved colorized image to {args.output}")

if __name__ == '__main__':
    main()