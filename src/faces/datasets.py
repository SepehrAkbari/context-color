import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from skimage import color

class CelebAColorization(Dataset):
    def __init__(self, root_dir, split='train', img_size=128):
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.filenames = sorted(os.listdir(self.img_dir))
        if split=='train':
            self.filenames = self.filenames[:180_000]
        else:
            self.filenames = self.filenames[180_000:200_000]
        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((img_size, img_size)),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        rgb_np = np.array(img) / 255.0
        lab_np = color.rgb2lab(rgb_np).astype(np.float32)
        L_np  = lab_np[:, :, 0:1]   / 100.0 
        ab_np = lab_np[:, :, 1:]    / 128.0 

        L   = torch.from_numpy(L_np).permute(2,0,1)
        ab  = torch.from_numpy(ab_np).permute(2,0,1)

        return L, ab
