from dataset import load_train_dataset
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np


def set_seeds():
    torch.manual_seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)


dataset = load_train_dataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(train_dataset[1][0].shape)
