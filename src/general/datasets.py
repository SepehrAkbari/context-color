import os
import random
import numpy as np
from skimage import color
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO

class COCOColorization(Dataset):
    def __init__(self, root: str, annFile: str, img_size: int = 128, n_samples: int = 30000):
        self.coco = COCO(annFile)
        self.ids  = list(self.coco.imgs.keys())
        if n_samples < len(self.ids):
            self.ids = random.sample(self.ids, n_samples)
        self.root      = root
        self.img_size  = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.coco.loadImgs(img_id)[0]
        path   = os.path.join(self.root, info['file_name'])
        img    = Image.open(path).convert('RGB')
        img    = self.transform(img)

        rgb_np  = np.array(img) / 255.0
        lab_np  = color.rgb2lab(rgb_np).astype(np.float32)
        L_np    = lab_np[:, :, 0:1] / 100.0
        ab_np   = lab_np[:, :, 1:]  / 128.0

        L  = torch.from_numpy(L_np).permute(2,0,1)
        ab = torch.from_numpy(ab_np).permute(2,0,1)
        return L, ab
