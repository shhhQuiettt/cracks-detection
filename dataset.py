from torch.utils.data import TensorDataset
import torch
from pathlib import Path
from PIL import Image
import numpy as np


def load_train_dataset():
    train_dir_images = Path("./data/train/images")
    train_dir_masks = Path("./data/train/masks")
    img_tensors = []
    img_masks = []

    i = 0
    for img in train_dir_images.glob("*.jpg"):
        img_tensors.append(torch.tensor(
            np.array(Image.open(img), dtype=np.float32)/255.0))
        i += 1
        if i == 100:
            break

    i = 0
    for mask in train_dir_masks.glob("*.jpg"):
        img_masks.append(torch.tensor(np.array(Image.open(mask))/255.0))
        i += 1
        if i == 100:
            break

    assert len(img_tensors) == len(img_masks)

    return TensorDataset(
        torch.stack(img_tensors).permute((0, 3, 1, 2)), torch.stack(
            img_masks).permute((0, 3, 1, 2))
    )


import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        """
        Custom dataset for loading images and masks.

        Args:
            image_dir (str): Directory containing images.
            mask_dir (str): Directory containing masks.
            transform (callable, optional): Transformations to apply to the images.
            mask_transform (callable, optional): Transformations to apply to the masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))  # Sort to align images with masks
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")  # Convert image to RGB
        mask = Image.open(mask_path).convert("L")      # Convert mask to grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

