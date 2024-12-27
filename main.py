
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
from torchvision import transforms
from model import UNet, Trainer
from dataset import load_train_dataset, ImageMaskDataset


def set_seeds():
    torch.manual_seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)

def check_dataloader_scaling(dataloader):
    """
    Check if the data in the DataLoader is scaled correctly (values in [0, 1]).

    Args:
        dataloader (DataLoader): The DataLoader to check.

    Returns:
        None
    """
    for i, (images, masks) in enumerate(dataloader):
        # Check min and max for images
        img_min, img_max = images.min().item(), images.max().item()
        # Check min and max for masks
        mask_min, mask_max = masks.min().item(), masks.max().item()

        print(f"Batch {i + 1}:")
        print(f"  Images - Min: {img_min}, Max: {img_max}")
        print(f"  Masks - Min: {mask_min}, Max: {mask_max}")

        # Ensure values are within the expected range
        if not (0.0 <= img_min <= 1.0 and 0.0 <= img_max <= 1.0):
            print(f"  Error: Image values are out of range [0, 1]!")
        if not (0.0 <= mask_min <= 1.0 and 0.0 <= mask_max <= 1.0):
            print(f"  Error: Mask values are out of range [0, 1]!")

        # Break after checking the first few batches for efficiency
        if i == 2:  # Change this number to check more batches
            break



# Define paths
image_dir = "data/tra/images"
mask_dir = "data/tra/masks"

# Define transformations for images and masks
image_transform = transforms.Compose([
    transforms.Resize((448, 448)),  # Ensure consistent size
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

mask_transform = transforms.Compose([
    transforms.Resize((448, 448)),  # Ensure consistent size
    transforms.ToTensor()           # Convert to tensor (values will be in [0, 1])
])

# Create dataset
dataset = ImageMaskDataset(image_dir, mask_dir, transform=image_transform, mask_transform=mask_transform)

# Split dataset into train and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Check DataLoader scaling
# check_dataloader_scaling(val_loader)


# dataset = load_train_dataset()

# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


endoding_channels = (3, 64, 128, 256, 512, 1024)
decoding_channels = (1024, 512, 256, 128, 64)

model = UNet(endoding_channels, decoding_channels, retain_dim=True)

trainer = Trainer(model=model, loss_function=torch.nn.CrossEntropyLoss(), train_loader=train_loader,
                  val_loader=val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001))
print("Training...")
trainer.train(epochs=10)
print("Training complete!")


# print(train_dataset[1][0].shape)
