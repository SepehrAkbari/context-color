#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.utils as vutils

# make sure we can import from faces/
# __file__ = .../COLORME/src/faces/train.py
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_ROOT)

from faces.datasets import CelebAColorization
from faces.model    import ColorNet

# ─── 1) Repro & Performance ──────────────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# ─── 2) Hyperparams ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(SRC_ROOT, '..'))    # .../COLORME
DATA_ROOT    = os.path.join(BASE_DIR, 'data', 'celeba')
IMG_SIZE     = 128
BATCH_SIZE   = 32
NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE_LR  = 3     # epochs w/o val-improve → halve LR
PATIENCE_STOP= 7     # epochs w/o val-improve → early stop
VIZ_EVERY    = 5     # save visuals every N epochs

# ─── 3) Device & Dataloaders ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = CelebAColorization(DATA_ROOT, split='train', img_size=IMG_SIZE)
val_ds   = CelebAColorization(DATA_ROOT, split='val',   img_size=IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ─── 4) Model, Opt, Loss, Sched, AMP ──────────────────────────────────────────
model     = ColorNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.L1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=PATIENCE_LR)
scaler    = GradScaler()

# ─── 5) Metrics & Viz Utils ──────────────────────────────────────────────────
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 10 * log10(1 / mse.item())

def lab_to_rgb_tensor(lab_tensor):
    lab_np = lab_tensor.permute(0,2,3,1).cpu().numpy()
    out = []
    for img in lab_np:
        img[:,:,0] *= 100.0
        img[:,:,1:] *= 128.0
        rgb = __import__('skimage').color.lab2rgb(img)
        out.append(rgb.transpose(2,0,1))
    return torch.from_numpy(np.stack(out)).float()

def save_viz(epoch, n=4):
    model.eval()
    L, ab_gt = next(iter(val_loader))
    L, ab_gt = L[:n].to(device), ab_gt[:n].to(device)
    with torch.no_grad(), autocast():
        ab_pred = model(L)
    lab_gt   = torch.cat([L, ab_gt], dim=1)
    lab_pred = torch.cat([L, ab_pred], dim=1)
    rgb_gt   = lab_to_rgb_tensor(lab_gt)
    rgb_pred = lab_to_rgb_tensor(lab_pred)
    grid = vutils.make_grid(torch.cat([rgb_pred, rgb_gt], dim=0), nrow=n)
    os.makedirs(os.path.join(BASE_DIR, 'visuals'), exist_ok=True)
    vutils.save_image(grid, os.path.join(BASE_DIR, f'visuals/viz_epoch{epoch:02d}.png'))

# ─── 6) Train & Val ───────────────────────────────────────────────────────────
def train_epoch(epoch):
    model.train()
    total_loss = 0.0
    for L, ab in train_loader:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        with autocast():
            ab_pred = model(L)
            loss    = criterion(ab_pred, ab)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch:02d} — Train Loss: {avg:.4f}")
    return avg

def validate(epoch):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)
            with autocast():
                ab_pred = model(L)
                loss    = criterion(ab_pred, ab)
            total_loss += loss.item()
            total_psnr += compute_psnr(ab_pred, ab)
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    print(f"Epoch {epoch:02d} — Val Loss: {avg_loss:.4f}, PSNR_ab: {avg_psnr:.2f} dB")
    return avg_loss

# ─── 7) Main Loop ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting training on device:", device)
    best_loss, no_improve = float('inf'), 0

    for epoch in range(1, NUM_EPOCHS+1):
        _         = train_epoch(epoch)
        val_loss  = validate(epoch)

        # LR scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  ↳ LR reduced {old_lr:.2e} → {new_lr:.2e}")

        # checkpoint & early stop
        if val_loss < best_loss:
            best_loss, no_improve = val_loss, 0
            torch.save(model.state_dict(),
                       os.path.join(BASE_DIR, 'models', 'best_model.pt'))
            print("  ↳ Saved new best model.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE_STOP:
                print(f"No improvement for {PATIENCE_STOP} epochs → stopping.")
                break

        # visual checkpoint
        if epoch % VIZ_EVERY == 0:
            save_viz(epoch)

    print(f"Training complete. Best val loss: {best_loss:.4f}")
