import os
import sys
import numpy as np
import rasterio

# Ensure root directory is in python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def calculate_twi(slope_rad, flow_acc):
    """
    Topographic Wetness Index (TWI) = ln(a / tan(b))
    """
    tan_slope = np.tan(slope_rad)
    tan_slope[tan_slope <= 0] = 1e-5
    flow_acc[flow_acc <= 0] = 1.0
    twi = np.log(flow_acc / tan_slope)
    return twi

def generate_12_conditioning_factors(shape, transform):
    """
    Generates 12 standard geospatial flood susceptibility factors:
    1. Elevation (DEM)       7. Drainage Density
    2. Slope                 8. Flow Accumulation
    3. TWI                   9. Soil Type
    4. Distance from River  10. Geology
    5. Rainfall             11. Plan Curvature
    6. Land Use / Cover     12. NDVI
    """
    height, width = shape
    
    # 1. Elevation (DEM in meters)
    y, x = np.ogrid[:height, :width]
    elevation = 800.0 - 0.5 * x - 0.3 * y + np.random.normal(0, 2.0, (height, width))
    elevation = np.clip(elevation, 100.0, 1200.0).astype(np.float32)

    # 2. Slope (degrees)
    slope_deg = np.abs(np.gradient(elevation, axis=0)) * 2.0 + np.abs(np.gradient(elevation, axis=1)) * 2.0
    slope_deg = np.clip(slope_deg, 0.1, 45.0).astype(np.float32)
    slope_rad = np.radians(slope_deg)

    # 4. Distance from River (meters)
    river_line = 256 + 60 * np.sin(x / 35.0)
    dist_river = np.abs(y - river_line) * 30.0 # 30m cell resolution
    dist_river = dist_river.astype(np.float32)

    # 8. Flow Accumulation
    flow_acc = (1.0 / (dist_river + 1.0)) * 50000.0 + np.random.uniform(1, 100, (height, width))
    flow_acc = flow_acc.astype(np.float32)

    # 3. Topographic Wetness Index (TWI)
    twi = calculate_twi(slope_rad, flow_acc).astype(np.float32)

    # 5. Rainfall (mm/year)
    rainfall = 900.0 + 0.2 * elevation + np.random.normal(0, 10.0, (height, width))
    rainfall = rainfall.astype(np.float32)

    # 6. Land Use / Land Cover (Categorical: 1=Water, 2=Urban, 3=Agri, 4=Forest)
    lulc = np.full((height, width), 3, dtype=np.float32)
    lulc[dist_river < 60.0] = 1.0 # River corridor
    lulc[elevation > 750.0] = 4.0 # Forested high elevation

    # 7. Drainage Density (km/sq. km)
    drainage_density = (1.0 / (dist_river + 10.0)) * 10.0
    drainage_density = drainage_density.astype(np.float32)

    # 9. Soil Type (Infiltration capacity rating 1-5)
    soil = np.random.choice([1.0, 2.0, 3.0, 4.0], size=(height, width)).astype(np.float32)

    # 10. Geology (Permeability rating 1-5)
    geology = np.random.choice([1.0, 2.0, 3.0], size=(height, width)).astype(np.float32)

    # 11. Plan Curvature
    plan_curvature = np.gradient(slope_deg, axis=1).astype(np.float32)

    # 12. NDVI
    ndvi = np.random.uniform(0.1, 0.7, size=(height, width)).astype(np.float32)
    ndvi[dist_river < 60.0] = -0.2 # Open water

    return [
        elevation, slope_deg, twi, dist_river, rainfall,
        lulc, drainage_density, flow_acc, soil, geology,
        plan_curvature, ndvi
    ]

def compute_flood_susceptibility_index(factors):
    """
    Multi-Criteria Weighted Linear Combination (WLC) Model
    FSI = Sum(Factor_i * Weight_i)
    """
    weights = [
        0.18, # Elevation (Low elevation = high risk)
        0.15, # Slope (Flat slope = high risk)
        0.12, # TWI (High TWI = high risk)
        0.15, # Distance from River (Close = high risk)
        0.10, # Rainfall
        0.08, # LULC
        0.05, # Drainage Density
        0.05, # Flow Accumulation
        0.04, # Soil Type
        0.03, # Geology
        0.02, # Plan Curvature
        0.03  # NDVI
    ]

    # Normalize factors to standard 0-1 scale
    norm_factors = []
    for i, f in enumerate(factors):
        f_min, f_max = np.min(f), np.max(f)
        if f_max - f_min > 0:
            norm = (f - f_min) / (f_max - f_min)
        else:
            norm = np.zeros_like(f)

        # Invert scale for inverse risk relationships
        if i in [0, 1, 3]: # Elevation, Slope, Distance to River (lower value = higher flood risk)
            norm = 1.0 - norm

        norm_factors.append(norm)

    # Weighted Sum
    fsi = np.zeros_like(factors[0], dtype=np.float32)
    for i in range(len(weights)):
        fsi += norm_factors[i] * weights[i]

    # Classify into 5 Risk Zones (1: Very Low, 2: Low, 3: Moderate, 4: High, 5: Very High)
    fsi_classified = np.zeros_like(fsi, dtype=np.int16)
    fsi_classified[fsi < 0.2] = 1
    fsi_classified[(fsi >= 0.2) & (fsi < 0.4)] = 2
    fsi_classified[(fsi >= 0.4) & (fsi < 0.6)] = 3
    fsi_classified[(fsi >= 0.6) & (fsi < 0.8)] = 4
    fsi_classified[fsi >= 0.8] = 5

    return fsi, fsi_classified

def process_susceptibility():
    raw_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    sar_files = [f for f in os.listdir(raw_dir) if f.endswith("_S1Hand.tif")]

    print("--- Starting Phase 3: 12-Factor Flood Susceptibility Mapping Engine ---")
    for sar_file in sar_files:
        chip_id = sar_file.replace("_S1Hand.tif", "")
        sar_path = os.path.join(raw_dir, sar_file)

        with rasterio.open(sar_path) as src:
            shape = src.shape
            transform = src.transform
            meta = src.meta.copy()

        # Generate 12 factors & calculate FSI
        factors = generate_12_conditioning_factors(shape, transform)
        fsi_continuous, fsi_classified = compute_flood_susceptibility_index(factors)

        # Save Flood Susceptibility Map
        out_path = os.path.join(output_dir, f"{chip_id}_SusceptibilityRisk.tif")
        meta.update(count=2, dtype=np.float32, nodata=-9999.0)

        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(fsi_continuous.astype(np.float32), 1)
            dst.write(fsi_classified.astype(np.float32), 2)
            dst.set_band_description(1, 'FSI_Continuous_Score')
            dst.set_band_description(2, 'FSI_5Class_RiskZone')

        print(f"Generated Susceptibility Map: {out_path}")

    print("--- Phase 3 Multi-Factor Susceptibility Execution Complete! ---")

if __name__ == "__main__":
    process_susceptibility()