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
        img_tensors.append(torch.tensor(np.array(Image.open(img))))
        i += 1
        if i == 100:
            break

    i = 0
    for mask in train_dir_masks.glob("*.jpg"):
        img_masks.append(torch.tensor(np.array(Image.open(mask))))
        i += 1
        if i == 100:
            break

    assert len(img_tensors) == len(img_masks)

    return TensorDataset(
        torch.stack(img_tensors).permute((0, 3, 1, 2)), torch.stack(img_masks)
    )


print(load_train_dataset()[0][1].shape)
