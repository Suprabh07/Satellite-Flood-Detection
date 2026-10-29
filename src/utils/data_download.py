import os
import urllib.request

# Verified direct Google Cloud Storage endpoint for Sen1Floods11
BASE_URL = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"

# Verified chip filenames in Sen1Floods11 HandLabeled subset
SAMPLE_CHIPS = [
    "Bolivia_103757_S1Hand.tif",
    "Ghana_103272_S1Hand.tif",
    "India_1010324_S1Hand.tif",
    "Mekong_1102143_S1Hand.tif",
    "Spain_1018318_S1Hand.tif"
]

def download_sample_data():
    raw_dir = os.path.join("data", "raw")
    mask_dir = os.path.join("data", "masks")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print("--- Starting Sen1Floods11 Sample Data Download ---")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for chip in SAMPLE_CHIPS:
        # 1. Download Sentinel-1 SAR Image Chip
        sar_url = f"{BASE_URL}/S1Hand/{chip}"
        sar_path = os.path.join(raw_dir, chip)
        
        if not os.path.exists(sar_path):
            print(f"Downloading SAR Image: {chip}...")
            try:
                req = urllib.request.Request(sar_url, headers=headers)
                with urllib.request.urlopen(req) as response, open(sar_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {chip}: {e}")
        else:
            print(f"SAR Image already exists: {chip}")
            
        # 2. Download Matching Ground Truth Flood Label Mask
        mask_name = chip.replace("_S1Hand.tif", "_LabelHand.tif")
        mask_url = f"{BASE_URL}/LabelHand/{mask_name}"
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Downloading Ground Truth Mask: {mask_name}...")
            try:
                req = urllib.request.Request(mask_url, headers=headers)
                with urllib.request.urlopen(req) as response, open(mask_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {mask_name}: {e}")
        else:
            print(f"Mask already exists: {mask_name}")

    print("\n--- Download Task Finished ---")

if __name__ == "__main__":
    download_sample_data()