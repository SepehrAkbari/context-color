#!/usr/bin/env python3
import os
import sys
import argparse

import numpy as np
from PIL import Image
import torch
from skimage import color
import cv2

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

def inpaint_ab(ab_pred, device, threshold=1e-3, radius=3, iterations=2):
    """
    ab_pred: torch.Tensor [1,2,H,W] on device
    Returns an inpainted ab tensor of same shape.
    """
    # move to CPU numpy HxWx2
    ab_np = ab_pred.squeeze(0).permute(1,2,0).cpu().numpy()
    H, W, _ = ab_np.shape

    # threshold mask where colors are essentially zero
    mask = (np.linalg.norm(ab_np, axis=2) < threshold).astype('uint8') * 255

    filled = ab_np.copy()
    for _ in range(iterations):
        # if no holes remain, break early
        if not mask.any():
            break
        for c in range(2):
            # scale channel from [-1,1] to [0,255]
            chan = ((filled[:,:,c] + 1) * 127.5).astype('uint8')
            # inpaint
            filled_chan = cv2.inpaint(chan, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
            # rescale back to [-1,1]
            filled[:,:,c] = filled_chan.astype('float32') / 127.5 - 1.0
        # recompute mask for next iteration
        mask = (np.linalg.norm(filled, axis=2) < threshold).astype('uint8') * 255

    # back to torch tensor
    ab_filled = torch.from_numpy(filled).permute(2,0,1).unsqueeze(0).to(device)
    return ab_filled

def postprocess(L, ab):
    lab = torch.cat([L, ab], dim=1)[0].permute(1,2,0).cpu().numpy()
    lab[:,:,0]  *= 100
    lab[:,:,1:] *= 128
    rgb = color.lab2rgb(lab)
    return Image.fromarray((rgb*255).astype(np.uint8))

def main():
    print("Starting...")
    p = argparse.ArgumentParser()
    p.add_argument('-i','--input',  required=True, help='Input RGB image or B&W')
    p.add_argument('-o','--output', default='out.png',
                   help='Where to save colorized result')
    p.add_argument('-m','--model',
                   default=os.path.abspath(
                     os.path.join(SRC_ROOT, '..', 'models', 'faces.pt')
                   ),
                   help='Path to faces.pt')
    p.add_argument('-s','--size', type=int, default=256,
                   help='Training image size (e.g. 128)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model  = load_model(args.model, device)
    print("Model loaded")

    _, L = preprocess(args.input, args.size)
    print("Image preprocessed")
    L = L.to(device)

    with torch.no_grad():
        ab_pred = model(L)

    # inpaint any zero-color holes
    ab = inpaint_ab(ab_pred, device)
    print("Colorization complete")

    out = postprocess(L, ab)
    print("Image postprocessed")
    out.save(args.output)
    print(f"✔ Saved fully‐colored image to {args.output}")

if __name__ == '__main__':
    main()