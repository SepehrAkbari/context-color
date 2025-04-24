#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
from math import log10
from itertools import islice

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.utils as vutils

from skimage import color
from PIL import Image

# ─── Make sure Python can find your src/ folder ───────────────────────────────
# Assumes this file lives in project root alongside `src/`
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from faces.datasets import CelebAColorization
from faces.model import ColorNet

# ─── 1) Reproducibility & Performance ────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# ─── 2) Hyperparameters ───────────────────────────────────────────────────────
DATA_ROOT    = 'data/celeba'
IMG_SIZE     = 128
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE_LR  = 3    # epochs to wait before halving LR
PATIENCE_STOP= 7    # epochs to wait before early stopping
VIZ_EVERY    = 5    # epochs between saving visual checkpoints

# ─── 3) Device, Data ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = CelebAColorization(DATA_ROOT, split='train', img_size=IMG_SIZE)
val_ds   = CelebAColorization(DATA_ROOT, split='val',   img_size=IMG_SIZE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ─── 4) Model, Optimizer, Criterion, Scheduler, AMP ──────────────────────────
model     = ColorNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.L1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=PATIENCE_LR,
                                                 verbose=True)
scaler    = GradScaler()

# ─── 5) Metrics & Utilities ───────────────────────────────────────────────────
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 10 * log10(1 / mse.item())

def lab_to_rgb_tensor(lab_tensor):
    """
    lab_tensor: torch.Tensor [B,3,H,W], L in [0,1], ab in [-1,1]
    returns:    torch.Tensor [B,3,H,W], rgb in [0,1]
    """
    lab_np = lab_tensor.permute(0,2,3,1).cpu().numpy()
    out = []
    for img in lab_np:
        img_copy = img.copy()
        img_copy[:,:,0] = img_copy[:,:,0] * 100
        img_copy[:,:,1:] = img_copy[:,:,1:] * 128
        rgb = color.lab2rgb(img_copy.astype(np.float64))
        out.append(rgb.transpose(2,0,1))
    return torch.from_numpy(np.stack(out)).float()

def save_viz(epoch, n_images=4):
    model.eval()
    L, ab_gt = next(iter(val_loader))
    L, ab_gt = L[:n_images].to(device), ab_gt[:n_images].to(device)
    with torch.no_grad(), autocast():
        ab_pred = model(L)
    # build LAB stacks
    lab_gt   = torch.cat([L, ab_gt], dim=1)
    lab_pred = torch.cat([L, ab_pred], dim=1)
    # convert to RGB
    rgb_gt   = lab_to_rgb_tensor(lab_gt)
    rgb_pred = lab_to_rgb_tensor(lab_pred)
    # stack: first row = preds, second row = ground truth
    grid = vutils.make_grid(torch.cat([rgb_pred, rgb_gt], dim=0), nrow=n_images)
    os.makedirs('visuals', exist_ok=True)
    vutils.save_image(grid, f'visuals/viz_epoch{epoch:02d}.png')

# ─── 6) Training & Validation Loops ───────────────────────────────────────────
def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    for L, ab_gt in train_loader:
        L, ab_gt = L.to(device), ab_gt.to(device)
        optimizer.zero_grad()
        with autocast():
            ab_pred = model(L)
            loss    = criterion(ab_pred, ab_gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    avg = running_loss / len(train_loader)
    print(f"Epoch {epoch:02d} — Train Loss: {avg:.4f}")
    return avg

def validate(epoch):
    model.eval()
    val_loss  = 0.0
    psnr_total= 0.0
    with torch.no_grad():
        for L, ab_gt in val_loader:
            L, ab_gt = L.to(device), ab_gt.to(device)
            with autocast():
                ab_pred = model(L)
                loss    = criterion(ab_pred, ab_gt)
            val_loss   += loss.item()
            psnr_total += compute_psnr(ab_pred, ab_gt)
    avg_loss = val_loss / len(val_loader)
    avg_psnr = psnr_total / len(val_loader)
    print(f"Epoch {epoch:02d} — Val Loss: {avg_loss:.4f}, PSNR_ab: {avg_psnr:.2f} dB")
    return avg_loss

# ─── 7) Main Loop with Checkpointing & Early Stopping ─────────────────────────
if __name__ == '__main__':
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        _ = train_epoch(epoch)
        val_loss = validate(epoch)

        # LR scheduler step
        scheduler.step(val_loss)

        # checkpoint & early stop
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("  ↳ Saved new best model.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE_STOP:
                print(f"No improvement for {PATIENCE_STOP} epochs — stopping early.")
                break

        # visual checkpoint
        if epoch % VIZ_EVERY == 0:
            save_viz(epoch)

    print("Training complete. Best val loss: {:.4f}".format(best_loss))