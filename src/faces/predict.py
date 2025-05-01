import os
import sys
import argparse

import numpy as np
from PIL import Image
import torch
from skimage import color
import cv2

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
    ab_np = ab_pred.squeeze(0).permute(1,2,0).cpu().numpy()
    H, W, _ = ab_np.shape

    mask = (np.linalg.norm(ab_np, axis=2) < threshold).astype('uint8') * 255

    filled = ab_np.copy()
    for _ in range(iterations):
        if not mask.any():
            break
        for c in range(2):
            chan = ((filled[:,:,c] + 1) * 127.5).astype('uint8')
            filled_chan = cv2.inpaint(chan, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
            filled[:,:,c] = filled_chan.astype('float32') / 127.5 - 1.0
        mask = (np.linalg.norm(filled, axis=2) < threshold).astype('uint8') * 255

    ab_filled = torch.from_numpy(filled).permute(2,0,1).unsqueeze(0).to(device)
    return ab_filled

def postprocess(L, ab):
    lab = torch.cat([L, ab], dim=1)[0].permute(1,2,0).cpu().numpy()
    lab[:,:,0]  *= 100
    lab[:,:,1:] *= 128
    rgb = color.lab2rgb(lab)
    return Image.fromarray((rgb*255).astype(np.uint8))

def main():
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

    _, L = preprocess(args.input, args.size)
    L = L.to(device)

    with torch.no_grad():
        ab_pred = model(L)

    # inpaint any zero-color holes
    ab = inpaint_ab(ab_pred, device)

    out = postprocess(L, ab)
    out.save(args.output)
    print(f"Saved output image to {args.output}")

if __name__ == '__main__':
    main()