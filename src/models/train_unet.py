import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src.utils.dataset import FloodDataset

def train_unet():
    # 1. Paths & Directories
    raw_dir = os.path.join("data", "raw")
    mask_dir = os.path.join("data", "masks")
    model_save_path = os.path.join("src", "models", "unet_flood_model.pth")

    # 2. Setup Device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training U-Net Model on Device: {device} ---")

    # 3. Load Dataset & DataLoader
    dataset = FloodDataset(raw_dir=raw_dir, mask_dir=mask_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 4. Initialize U-Net Model with ResNet-34 Encoder Backbone
    # in_channels=2 for dual-band SAR (VV, VH), classes=1 for binary flood output
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None
    ).to(device)

    # 5. Loss Function & Optimizer
    criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 6. Training Loop (15 Epochs)
    epochs = 15
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for sar_imgs, masks in dataloader:
            sar_imgs, masks = sar_imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(sar_imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Dice Loss: {avg_loss:.4f}")

    # Save Trained Model Weights
    torch.save(model.state_dict(), model_save_path)
    print(f"\n--- Model Training Complete! Saved weights to: {model_save_path} ---")

if __name__ == "__main__":
    train_unet()