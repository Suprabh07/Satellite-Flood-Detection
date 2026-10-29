import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class FloodDataset(Dataset):
    def __init__(self, raw_dir, mask_dir, transform=None):
        self.raw_dir = raw_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.samples = [f for f in os.listdir(raw_dir) if f.endswith("_S1Hand.tif")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sar_filename = self.samples[idx]
        mask_filename = sar_filename.replace("_S1Hand.tif", "_LabelHand.tif")

        sar_path = os.path.join(self.raw_dir, sar_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # 1. Read Dual-Band SAR (VV and VH)
        with rasterio.open(sar_path) as src:
            vv = src.read(1)
            vh = src.read(2)
            sar_image = np.stack([vv, vh], axis=0) # Shape: (2, H, W)

        # Clean NaN/Inf values and normalize dB values
        sar_image = np.nan_to_num(sar_image, nan=-30.0, posinf=-5.0, neginf=-35.0)
        sar_image = (sar_image - sar_image.mean()) / (sar_image.std() + 1e-6)

        # 2. Read Label Mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            
        # Binary mask (1 = Water, 0 = Non-Water)
        mask = (mask == 1).astype(np.float32)
        mask = np.expand_dims(mask, axis=0) # Shape: (1, H, W)

        return torch.tensor(sar_image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)