# 🌊 Satellite-Based Flood Detection & Mapping System Using Deep Learning

An end-to-end geospatial machine learning pipeline and interactive web application for real-time flood inundation mapping and 12-factor flood susceptibility zonation.

---

## 📌 Focal Study Catchments
1. **Local Catchments:** Bengaluru (Varthur & Bellandur Lake Catchments)
2. **Regional River Basin:** Gandak River Basin (Northern Bihar)
3. **Global Benchmark:** Sen1Floods11 Dataset

---

## 🏗️ System Architecture

1. **Data Ingestion & Preprocessing:** Dual-band Sentinel-1 SAR (`VV`/`VH`), Sentinel-2 Optical (`NDVI`/`MNDWI`), and 30m Cartosat/Copernicus DEM.
2. **Baseline Engine:** Automated Otsu Backscatter Intensity Thresholding.
3. **Deep Learning Engine:** PyTorch U-Net with ResNet-34 Encoder backbone trained on multi-spectral input chips.
4. **Susceptibility Zonation:** Multi-Criteria Evaluation (MCE) incorporating 12 topographic, hydrological, and environmental conditioning factors.
5. **Interactive Dashboard:** Streamlit + Folium web interface displaying raster maps, metrics, and vector export features.

---

## 📊 Model Performance Comparison

| Model Architecture | Input Features | Target Region | IoU Score | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Otsu Baseline** | Sentinel-1 SAR (`VH`) | Bengaluru Catchments | 0.8035 | 0.8035 | 1.0000 | 0.8910 |
| **Otsu Baseline** | Sentinel-1 SAR (`VH`) | Gandak River Basin | 0.8055 | 0.8055 | 1.0000 | 0.8923 |
| **U-Net (ResNet-34)** | Sentinel-1 SAR (`VV` + `VH`) | Bengaluru Catchments | **0.9545** | **0.9550** | **0.9995** | **0.9767** |
| **U-Net (ResNet-34)** | Sentinel-1 SAR (`VV` + `VH`) | Gandak River Basin | **0.9539** | **0.9544** | **0.9995** | **0.9764** |
| **U-Net (Fused)** | S1 SAR + S2 Optical (`NDVI` + `MNDWI`) | Multi-Regional | **0.9620** | **0.9640** | **0.9980** | **0.9806** |

---

## 🚀 Quickstart Guide

### 1. Clone & Activate Environment
```bash
git clone [https://github.com/your-username/satellite-flood-detection.git](https://github.com/your-username/satellite-flood-detection.git)
cd satellite-flood-detection

python -m venv venv
# PowerShell (Windows)
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt