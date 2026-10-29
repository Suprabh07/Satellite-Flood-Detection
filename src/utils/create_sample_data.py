import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_local_flood_sample(chip_id, lat, lon, output_raw_dir, output_mask_dir):
    width, height = 512, 512
    # Geo-transform centered on local coordinates (30m pixel resolution ~ 0.00027 degrees)
    transform = from_origin(lon, lat, 0.00027, 0.00027)
    
    # 1. Generate Synthetic Dual-Band SAR Imagery (VV and VH polarizations in dB)
    vv_band = np.random.normal(loc=-10.0, scale=3.0, size=(height, width)).astype(np.float32)
    vh_band = np.random.normal(loc=-15.0, scale=3.0, size=(height, width)).astype(np.float32)
    
    # Create simulated flooded inundation zone
    mask = np.zeros((height, width), dtype=np.int16)
    
    y_indices, x_indices = np.ogrid[:height, :width]
    water_center = 256 + 60 * np.sin(x_indices / 35.0)
    flood_condition = np.abs(y_indices - water_center) < 45
    
    # Set water pixels in mask (1 = Flood Water, 0 = Non-Water)
    mask[flood_condition] = 1
    
    # Lower SAR backscatter over flooded water surfaces
    vv_band[flood_condition] = np.random.normal(loc=-22.0, scale=1.5, size=np.sum(flood_condition))
    vh_band[flood_condition] = np.random.normal(loc=-26.0, scale=1.5, size=np.sum(flood_condition))
    
    # 2. Save Dual-Band SAR GeoTIFF
    sar_filepath = os.path.join(output_raw_dir, f"{chip_id}_S1Hand.tif")
    with rasterio.open(
        sar_filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=2,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(vv_band, 1)
        dst.write(vh_band, 2)
        dst.set_band_description(1, 'VV')
        dst.set_band_description(2, 'VH')
        
    # 3. Save Label Mask
    mask_filepath = os.path.join(output_mask_dir, f"{chip_id}_LabelHand.tif")
    with rasterio.open(
        mask_filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.int16,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, 'Label')
        
    print(f"Generated sample chip for area: {chip_id} at Lat: {lat}, Lon: {lon}")

def build_dataset():
    raw_dir = os.path.join("data", "raw")
    mask_dir = os.path.join("data", "masks")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Focal study locations
    study_locations = [
        ("Bengaluru_Varthur_01", 12.9406, 77.7460),
        ("Bengaluru_Bellandur_02", 12.9279, 77.6727),
        ("Gandak_RiverBasin_01", 26.5000, 84.5000),
        ("Gandak_RiverBasin_02", 26.8000, 84.8000),
        ("Sen1Floods11_India_01", 25.3000, 82.9000)
    ]
    
    print("--- Generating Study Area Sample Chips & Label Masks ---")
    for name, lat, lon in study_locations:
        create_local_flood_sample(name, lat, lon, raw_dir, mask_dir)
    print("--- Sample Generation Completed! ---")

if __name__ == "__main__":
    build_dataset()