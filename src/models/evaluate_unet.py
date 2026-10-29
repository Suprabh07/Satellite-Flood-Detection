import os
import sys

# Ensure root folder is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

def evaluate_unet():
    raw_dir = os.path.join("data", "raw")
    mask_dir = os.path.join("data", "masks")
    processed_dir = os.path.join("data", "processed")
    model_path = os.path.join("src", "models", "unet_flood_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running U-Net Model Inference on Device: {device} ---")

    # 1. Load Trained U-Net Model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sar_files = [f for f in os.listdir(raw_dir) if f.endswith("_S1Hand.tif")]

    for sar_file in sar_files:
        chip_id = sar_file.replace("_S1Hand.tif", "")
        mask_file = f"{chip_id}_LabelHand.tif"

        sar_path = os.path.join(raw_dir, sar_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # 2. Read Dual-Band SAR Imagery
        with rasterio.open(sar_path) as src:
            vv = src.read(1)
            vh = src.read(2)
            meta = src.meta.copy()
            transform = src.transform

        sar_image = np.stack([vv, vh], axis=0)
        sar_image = np.nan_to_num(sar_image, nan=-30.0, posinf=-5.0, neginf=-35.0)
        sar_image = (sar_image - sar_image.mean()) / (sar_image.std() + 1e-6)

        # Prepare Tensor for Model
        input_tensor = torch.tensor(sar_image, dtype=torch.float32).unsqueeze(0).to(device)

        # 3. Model Prediction
        with torch.no_grad():
            output = model(input_tensor)
            prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (prob_mask > 0.5).astype(np.int16)

        # 4. Load Ground Truth Mask & Calculate Metrics
        with rasterio.open(mask_path) as mask_src:
            raw_gt = mask_src.read(1)
            ground_truth = (raw_gt == 1).astype(np.int16)

        gt_flat = ground_truth.flatten()
        pred_flat = pred_mask.flatten()

        iou = jaccard_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
        precision = precision_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
        recall = recall_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
        f1 = f1_score(gt_flat, pred_flat, pos_label=1, zero_division=0)

        # 5. Calculate Inundated Flood Area (in sq km)
        # Pixel area calculation based on geo-transform resolution
        pixel_size_deg = transform[0]
        # Approximate 1 degree ~ 111.32 km
        pixel_area_sq_km = (pixel_size_deg * 111.32) ** 2
        water_pixel_count = np.sum(pred_mask == 1)
        total_flooded_area_sq_km = water_pixel_count * pixel_area_sq_km

        print(f"\n--- U-Net Prediction: {chip_id} ---")
        print(f"IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"Detected Flood Water Pixels: {water_pixel_count} / {pred_mask.size}")
        print(f"Calculated Inundated Flood Area: {total_flooded_area_sq_km:.2f} sq. km")

        # Save U-Net Prediction Raster
        meta.update(count=1, dtype=np.int16, nodata=-1)
        out_path = os.path.join(processed_dir, f"{chip_id}_UNetPred.tif")
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(pred_mask, 1)

if __name__ == "__main__":
    evaluate_unet()