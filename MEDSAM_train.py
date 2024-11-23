import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Assuming MedSAM is available and properly installed
# from medsam import MedSAM  # Uncomment this line and replace with the actual import path

# Placeholder for MedSAM model (replace with actual import)
class MedSAM(nn.Module):
    def __init__(self, pretrained=True):
        super(MedSAM, self).__init__()
        # Initialize the actual MedSAM model here
        pass

    def forward(self, x):
        # Define the forward pass
        pass

# Dice Loss implementation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        intersection = (outputs * targets).sum()
        dice_coefficient = (2. * intersection + self.smooth) / (
            outputs.sum() + targets.sum() + self.smooth
        )
        loss = 1 - dice_coefficient
        return loss

# Custom Dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Image transformations
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.round(x)),  # Binarize mask
])

# Create dataset and dataloader
dataset = MedicalImageDataset(
    images_dir="/images",
    masks_dir="/masks",
    transform=image_transform,
    mask_transform=mask_transform
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the MedSAM model with pretrained weights for finetuning
model = MedSAM(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = DiceLoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid if outputs are logits

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
