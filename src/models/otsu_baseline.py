import os
import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

def run_otsu_flood_detection(image_path, mask_path):
    # 1. Load Dual-Band SAR Image (Band 1: VV, Band 2: VH)
    with rasterio.open(image_path) as src:
        vv_band = src.read(1)
        vh_band = src.read(2)
        meta = src.meta.copy()

    # 2. Load Ground Truth Label Mask
    with rasterio.open(mask_path) as mask_src:
        raw_gt = mask_src.read(1)

    # Convert ground truth strictly to binary (1 = Water, 0 = Non-Water / No Data)
    ground_truth = (raw_gt == 1).astype(np.int16)

    # 3. Clean NaN / Inf values from floating point SAR backscatter matrix
    vh_clean = np.nan_to_num(vh_band, nan=-30.0, posinf=-5.0, neginf=-35.0)

    # 4. Apply Otsu Thresholding on cleaned VH band
    threshold_value = threshold_otsu(vh_clean)
    
    # Water surfaces reflect radar away (specular reflection), resulting in low backscatter
    predicted_mask = (vh_clean < threshold_value).astype(np.int16)

    # 5. Calculate Computer Vision Evaluation Metrics
    gt_flat = ground_truth.flatten()
    pred_flat = predicted_mask.flatten()

    iou = jaccard_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, pos_label=1, zero_division=0)

    print(f"\n--- Otsu Baseline Analysis: {os.path.basename(image_path)} ---")
    print(f"Calculated Backscatter Threshold (VH): {threshold_value:.2f} dB")
    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

    return predicted_mask, meta

def evaluate_baseline_on_dataset():
    raw_dir = os.path.join("data", "raw")
    mask_dir = os.path.join("data", "masks")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    sar_files = [f for f in os.listdir(raw_dir) if f.endswith("_S1Hand.tif")]

    print("--- Running Otsu Baseline Engine on Local Study Areas ---")
    for sar_file in sar_files:
        chip_id = sar_file.replace("_S1Hand.tif", "")
        mask_file = f"{chip_id}_LabelHand.tif"

        sar_path = os.path.join(raw_dir, sar_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path):
            pred_mask, meta = run_otsu_flood_detection(sar_path, mask_path)

            # Fix nodata type compatibility for int16 masks
            meta.update(count=1, dtype=np.int16, nodata=-1)
            out_path = os.path.join(processed_dir, f"{chip_id}_OtsuPred.tif")
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(pred_mask, 1)

if __name__ == "__main__":
    evaluate_baseline_on_dataset()