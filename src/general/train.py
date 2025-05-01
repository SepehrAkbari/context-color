import os
import sys
import random
import numpy as np
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torchvision.utils as vutils

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_ROOT)

from general.datasets import COCOColorization
from general.model    import ColorNet

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

BASE_DIR     = os.path.abspath(os.path.join(SRC_ROOT, '..'))
COCO_IMG_DIR = os.path.join(BASE_DIR, 'data', 'coco', 'images', 'train2017')
COCO_ANN     = os.path.join(BASE_DIR, 'data', 'coco', 'annotations', 'instances_train2017.json')
MODEL_OUT    = os.path.join(BASE_DIR, 'models', 'general.pt')

IMG_SIZE     = 128
BATCH_SIZE   = 32
NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE_LR  = 3
PATIENCE_STOP= 7
VIZ_EVERY    = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ds = COCOColorization(COCO_IMG_DIR, COCO_ANN, img_size=IMG_SIZE, n_samples=30000)
val_ds   = COCOColorization(COCO_IMG_DIR, COCO_ANN, img_size=IMG_SIZE, n_samples=5000)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

model     = ColorNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.L1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=PATIENCE_LR)
scaler    = GradScaler("cuda")

def compute_psnr(pred, target):
    return 10 * log10(1.0 / torch.mean((pred - target) ** 2).item())

def save_viz(epoch, n=4):
    model.eval()
    L, ab = next(iter(val_loader))
    L, ab = L[:n].to(device), ab[:n].to(device)
    with torch.no_grad(), autocast("cuda"):
        ab_pred = model(L)
    lab_gt   = torch.cat([L, ab], dim=1)
    lab_pred = torch.cat([L, ab_pred], dim=1)
    grid = vutils.make_grid(torch.cat([lab_pred, lab_gt], dim=0), nrow=n)
    os.makedirs(os.path.join(BASE_DIR, 'visuals'), exist_ok=True)
    vutils.save_image(grid, os.path.join(BASE_DIR, f'visuals/gen_viz{epoch:02d}.png'))

def train_epoch():
    model.train()
    running = 0
    for L, ab in train_loader:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        with autocast("cuda"):
            ab_pred = model(L)
            loss    = criterion(ab_pred, ab)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
    return running / len(train_loader)

def validate():
    model.eval()
    running, psnr_total = 0, 0
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)
            with autocast("cuda"):
                ab_pred = model(L)
                loss    = criterion(ab_pred, ab)
            running     += loss.item()
            psnr_total  += compute_psnr(ab_pred, ab)
    return running / len(val_loader), psnr_total / len(val_loader)

if __name__ == '__main__':
    best, no_improve = float('inf'), 0
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_epoch()
        val_loss, val_psnr = validate()
        print(f"Epoch {epoch:02d} — Train: {tr_loss:.4f} | Val: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")
        scheduler.step(val_loss)

        if val_loss < best:
            best, no_improve = val_loss, 0
            torch.save(model.state_dict(), MODEL_OUT)
            print("Saved general.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE_STOP:
                print("Early stopping.")
                break

        if epoch % VIZ_EVERY == 0:
            save_viz(epoch)
    print("Done. Best val loss:", best)
