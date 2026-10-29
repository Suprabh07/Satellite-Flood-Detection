import os
import sys

# Ensure root directory is in python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import rasterio
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Satellite Flood Detection System",
    page_icon="🌊",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size:28px; font-weight:bold; color:#1E88E5; }
    .metric-card { background-color:#F0F2F6; padding:15px; border-radius:10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Satellite-Based Flood Detection & Mapping Engine")
st.subheader("Multi-Sensor Deep Learning (U-Net) & 12-Factor Susceptibility Zonation")

# File directories
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
EXPORT_DIR = os.path.join("data", "exports")

# Get list of available study chips
chip_files = [f.replace("_S1Hand.tif", "") for f in os.listdir(RAW_DIR) if f.endswith("_S1Hand.tif")]

# Sidebar Region Selector
st.sidebar.header("📍 Module & Region Selection")
selected_chip = st.sidebar.selectbox("Select Study Area / Chip:", chip_files)

selected_module = st.sidebar.radio(
    "Select Active Module:",
    [
        "Phase 1: U-Net Deep Learning Engine",
        "Phase 1: Otsu Baseline Engine",
        "Phase 2: SAR + Optical (MNDWI/NDVI) Fusion",
        "Phase 3: 12-Factor Susceptibility Zonation",
        "Phase 4: Export Engine & Reports"
    ]
)

if selected_chip:
    sar_path = os.path.join(RAW_DIR, f"{selected_chip}_S1Hand.tif")
    
    with rasterio.open(sar_path) as src:
        vh_band = src.read(2)
        transform = src.transform
        bounds = src.bounds
    vh_clean = np.nan_to_num(vh_band, nan=-30.0)

    # MODULE 1 & 2: FLOOD DETECTION (U-Net / Otsu)
    if "Phase 1" in selected_module:
        if "U-Net" in selected_module:
            pred_path = os.path.join(PROCESSED_DIR, f"{selected_chip}_UNetPred.tif")
        else:
            pred_path = os.path.join(PROCESSED_DIR, f"{selected_chip}_OtsuPred.tif")

        with rasterio.open(pred_path) as pred_src:
            pred_mask = pred_src.read(1)

        water_pixels = np.sum(pred_mask == 1)
        pixel_size_deg = transform[0]
        pixel_area_sq_km = (pixel_size_deg * 111.32) ** 2
        total_flooded_area = water_pixels * pixel_area_sq_km
        flood_percentage = (water_pixels / pred_mask.size) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Selected Region", selected_chip.split("_")[0])
        with col2:
            st.metric("Inundated Area (sq km)", f"{total_flooded_area:.2f} km²")
        with col3:
            st.metric("Surface Coverage", f"{flood_percentage:.1f}%")
        with col4:
            st.metric("Water Pixel Count", f"{water_pixels:,}")

        st.markdown("---")
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("📡 Sentinel-1 SAR Backscatter (VH Band)")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            im1 = ax1.imshow(vh_clean, cmap='gray')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Backscatter (dB)")
            st.pyplot(fig1)

        with right_col:
            st.subheader("🗺️ Classified Flood Water Mask")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            masked_display = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
            masked_display[pred_mask == 1] = [0.11, 0.53, 0.89]
            ax2.imshow(vh_clean, cmap='gray')
            ax2.imshow(masked_display, alpha=0.6)
            ax2.axis('off')
            st.pyplot(fig2)

    # MODULE 3: PHASE 2 OPTICAL FUSION
    elif "Phase 2" in selected_module:
        fused_path = os.path.join(PROCESSED_DIR, f"{selected_chip}_Fused4Band.tif")
        with rasterio.open(fused_path) as fused_src:
            mndwi = fused_src.read(3)
            ndvi = fused_src.read(4)

        st.subheader("✨ Sentinel-1 SAR & Sentinel-2 Optical Fusion Stack")
        c1, c2 = st.columns(2)

        with c1:
            st.write("💧 **MNDWI (Modified Normalized Difference Water Index)**")
            fig_mndwi, ax_m = plt.subplots(figsize=(6, 6))
            im_m = ax_m.imshow(mndwi, cmap='Blues')
            ax_m.axis('off')
            plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
            st.pyplot(fig_mndwi)

        with c2:
            st.write("🌿 **NDVI (Normalized Difference Vegetation Index)**")
            fig_ndvi, ax_n = plt.subplots(figsize=(6, 6))
            im_n = ax_n.imshow(ndvi, cmap='YlGn')
            ax_n.axis('off')
            plt.colorbar(im_n, ax=ax_n, fraction=0.046, pad=0.04)
            st.pyplot(fig_ndvi)

    # MODULE 4: PHASE 3 SUSCEPTIBILITY ZONATION
    elif "Phase 3" in selected_module:
        risk_path = os.path.join(PROCESSED_DIR, f"{selected_chip}_SusceptibilityRisk.tif")
        with rasterio.open(risk_path) as risk_src:
            fsi_score = risk_src.read(1)
            fsi_classes = risk_src.read(2)

        high_risk_pixels = np.sum(fsi_classes >= 4)
        total_pixels = fsi_classes.size
        high_risk_pct = (high_risk_pixels / total_pixels) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Study Catchment", selected_chip.split("_")[0])
        with col2:
            st.metric("Conditioning Factors Used", "12 Geo-Factors")
        with col3:
            st.metric("High/Very High Risk Ratio", f"{high_risk_pct:.1f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Continuous Flood Susceptibility Index (FSI)")
            fig_fsi, ax_f = plt.subplots(figsize=(6, 6))
            im_f = ax_f.imshow(fsi_score, cmap='magma')
            ax_f.axis('off')
            plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04, label="FSI Score (0 to 1)")
            st.pyplot(fig_fsi)

        with c2:
            st.subheader("🚨 5-Class Risk Zonation Map")
            fig_cls, ax_c = plt.subplots(figsize=(6, 6))
            im_c = ax_c.imshow(fsi_classes, cmap='RdYlGn_r')
            ax_c.axis('off')
            plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04, label="1:V.Low to 5:V.High")
            st.pyplot(fig_cls)

    # MODULE 5: PHASE 4 EXPORTS & REPORTING
    elif "Phase 4" in selected_module:
        st.subheader("📁 Export Engine & Analytical Reports")
        st.write("Download processed GIS vector layers, GeoTIFF rasters, and summary analysis text files.")

        report_path = os.path.join(EXPORT_DIR, f"{selected_chip}_SummaryReport.txt")
        geojson_path = os.path.join(EXPORT_DIR, f"{selected_chip}_FloodVector.geojson")
        tif_path = os.path.join(PROCESSED_DIR, f"{selected_chip}_UNetPred.tif")

        col1, col2, col3 = st.columns(3)

        with col1:
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report_text = f.read()
                st.download_button(
                    label="📄 Download Analytical Summary Report",
                    data=report_text,
                    file_name=f"{selected_chip}_SummaryReport.txt",
                    mime="text/plain"
                )

        with col2:
            if os.path.exists(geojson_path):
                with open(geojson_path, "rb") as f:
                    geojson_bytes = f.read()
                st.download_button(
                    label="🗺️ Download GeoJSON Vector Layer",
                    data=geojson_bytes,
                    file_name=f"{selected_chip}_FloodVector.geojson",
                    mime="application/geo+json"
                )

        with col3:
            if os.path.exists(tif_path):
                with open(tif_path, "rb") as f:
                    tif_bytes = f.read()
                st.download_button(
                    label="🌐 Download Classified GeoTIFF Mask",
                    data=tif_bytes,
                    file_name=f"{selected_chip}_UNetPred.tif",
                    mime="image/tiff"
                )

        st.markdown("---")
        if os.path.exists(report_path):
            st.text_area("📄 Preview Analytical Summary Report:", report_text, height=350)

    # Interactive Folium Map Section
    st.markdown("---")
    st.subheader("🌍 Interactive Geospatial Bounds Map")
    
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
    
    folium.Rectangle(
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        color="#1E88E5",
        fill=True,
        fill_color="#1E88E5",
        fill_opacity=0.2,
        popup=f"Study Area: {selected_chip}"
    ).add_to(m)

    st_folium(m, width=1100, height=450)