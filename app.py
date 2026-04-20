import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import random
import torch
import torch.nn as nn
import joblib
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── NEW IMPORTS FOR SATELLITE FETCH & GEOCODING ──
from sentinelhub import (
    SHConfig, BBox, CRS, DataCollection,
    SentinelHubRequest, MimeType, bbox_to_dimensions
)
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TerraScan · Landslide Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SENTINEL HUB CONFIG (Copernicus Data Space)
# ─────────────────────────────────────────────
sh_config = SHConfig()
sh_config.sh_client_id     = "sh-67a3f8db-66e0-45c4-a103-194c12eb4372"
sh_config.sh_client_secret = "VpnW5MCxmc0vp7B0t23TtkQ8rwXpyCwk"
sh_config.sh_token_url     = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
sh_config.sh_base_url      = "https://sh.dataspace.copernicus.eu"

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #0b0f1a;
  --surface:   #111827;
  --border:    #1e2d40;
  --accent:    #00c8a0;
  --accent2:   #ff6b35;
  --warn:      #f5a623;
  --danger:    #e84040;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --mono:      'Space Mono', monospace;
  --sans:      'DM Sans', sans-serif;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px; }

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.top-banner {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  padding: 1.4rem 2rem;
  background: linear-gradient(135deg, #0d1b2a 0%, #112240 50%, #0d2137 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.top-banner::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 80% 50%, rgba(0,200,160,0.08) 0%, transparent 65%);
}
.banner-icon { font-size: 2.4rem; line-height: 1; }
.banner-title {
  font-family: var(--mono);
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--accent);
  margin: 0;
}
.banner-sub {
  font-size: 0.82rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0.2rem 0 0;
}
.banner-badge {
  margin-left: auto;
  background: rgba(0,200,160,0.12);
  border: 1px solid var(--accent);
  color: var(--accent);
  font-family: var(--mono);
  font-size: 0.68rem;
  padding: 0.3rem 0.7rem;
  border-radius: 4px;
  letter-spacing: 0.06em;
}

.section-label {
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.8rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
  flex: 1; min-width: 120px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.metric-card.green::before  { background: var(--accent); }
.metric-card.orange::before { background: var(--accent2); }
.metric-card.yellow::before { background: var(--warn); }
.metric-card.red::before    { background: var(--danger); }
.metric-label {
  font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 0.3rem; font-family: var(--mono);
}
.metric-value {
  font-family: var(--mono); font-size: 1.6rem; font-weight: 700;
  line-height: 1;
}
.metric-card.green  .metric-value { color: var(--accent); }
.metric-card.orange .metric-value { color: var(--accent2); }
.metric-card.yellow .metric-value { color: var(--warn); }
.metric-card.red    .metric-value { color: var(--danger); }

.gauge-label {
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.gauge-bar-bg {
  height: 14px; border-radius: 7px;
  background: var(--border);
  overflow: hidden; margin: 0.5rem 0;
}
.gauge-bar-fill {
  height: 100%; border-radius: 7px;
  background: linear-gradient(90deg, #00c8a0, #f5a623, #e84040);
  transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.gauge-value {
  font-family: var(--mono);
  font-size: 2.2rem; font-weight: 700;
}

[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
}

.stButton > button {
  background: var(--accent) !important;
  color: #000 !important;
  font-family: var(--mono) !important;
  font-size: 0.8rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 0.6rem 1.4rem !important;
  transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.info-box {
  background: rgba(0,200,160,0.07);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 0.8rem 1rem;
  font-size: 0.85rem;
  color: var(--text);
  margin: 0.8rem 0;
}
.warn-box {
  background: rgba(245,166,35,0.08);
  border-left: 3px solid var(--warn);
  border-radius: 6px;
  padding: 0.8rem 1rem;
  font-size: 0.85rem;
  color: var(--text);
  margin: 0.8rem 0;
}
.danger-box {
  background: rgba(232,64,64,0.08);
  border-left: 3px solid var(--danger);
  border-radius: 6px;
  padding: 0.8rem 1rem;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text);
  margin: 0.8rem 0;
}

.pipeline {
  display: flex; gap: 0; margin: 1rem 0;
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
}
.pipeline-step {
  flex: 1; padding: 0.7rem 0.6rem;
  background: var(--surface);
  text-align: center;
  border-right: 1px solid var(--border);
  font-size: 0.72rem;
  font-family: var(--mono);
  letter-spacing: 0.04em;
  color: var(--muted);
}
.pipeline-step:last-child { border-right: none; }
.pipeline-step.active { background: rgba(0,200,160,0.1); color: var(--accent); }
.pipeline-step.done   { background: rgba(0,200,160,0.05); color: var(--accent); }
.pipeline-step .step-icon { display: block; font-size: 1.1rem; margin-bottom: 0.2rem; }

.img-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.img-card-header {
  padding: 0.5rem 0.9rem;
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 0.4rem;
}
.dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.dot-green  { background: var(--accent); }
.dot-orange { background: var(--accent2); }
.dot-yellow { background: var(--warn); }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

hr { border-color: var(--border) !important; }

.sidebar-section {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  padding: 0.6rem 0 0.3rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.6rem;
}

.stImage > div > div > p {
  font-family: var(--mono) !important;
  font-size: 0.65rem !important;
  color: var(--muted) !important;
  text-align: center !important;
  letter-spacing: 0.06em !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 1D-CNN MODEL CLASS
# ─────────────────────────────────────────────
class LandslideRiskCNN(nn.Module):
    def __init__(self, num_features=6):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# LOAD REAL MODELS  (cached — loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # U-Net
    unet = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    unet.load_state_dict(torch.load("models/best_unet_model.pth", map_location=device))
    unet.eval().to(device)

    # 1D-CNN
    cnn = LandslideRiskCNN(num_features=6)
    cnn.load_state_dict(torch.load("models/best_risk_cnn.pth", map_location=device))
    cnn.eval().to(device)

    # Scaler
    scaler = joblib.load("models/scaler.pkl")

    return unet, cnn, scaler, device


# ─────────────────────────────────────────────
# CONSTANTS & TRANSFORMS
# ─────────────────────────────────────────────
IMG_SIZE = 256

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# REAL INFERENCE & FETCH FUNCTIONS
# ─────────────────────────────────────────────
def fetch_sentinel_data(lat, lon, size_km=2.0):
    """
    Fetches both Sentinel-2 RGB (PNG) and Copernicus 30m DEM (TIFF).
    Returns (rgb_image_array, dem_float_array) or (None, None).
    """
    delta = size_km / 111.0          # degrees per km (approx)
    bbox  = BBox(
        bbox=[lon - delta, lat - delta, lon + delta, lat + delta],
        crs=CRS.WGS84
    )
    
    # ── 1. S2 RGB REQUEST ──
    evalscript_rgb = """
    //VERSION=3
    function setup() {
      return { input: ["B04","B03","B02"], output: { bands: 3 } };
    }
    function evaluatePixel(s) {
      return [2.5*s.B04, 2.5*s.B03, 2.5*s.B02]; 
    }
    """
    cdse_s2 = DataCollection.SENTINEL2_L2A.define_from(
        "cdse_s2", service_url=sh_config.sh_base_url
    )
    
    req_rgb = SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=cdse_s2,
                time_interval=("2024-01-01", "2024-12-31"),
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=[256, 256],
        config=sh_config,
    )

    # ── 2. DEM REQUEST ──
    evalscript_dem = """
    //VERSION=3
    function setup() {
      return { input: ["DEM"], output: { bands: 1, sampleType: "FLOAT32" } };
    }
    function evaluatePixel(s) {
      return [s.DEM];
    }
    """
    cdse_dem = DataCollection.DEM_COPERNICUS_30.define_from(
        "cdse_dem", service_url=sh_config.sh_base_url
    )
    
    req_dem = SentinelHubRequest(
        evalscript=evalscript_dem,
        input_data=[SentinelHubRequest.input_data(data_collection=cdse_dem)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=[256, 256],
        config=sh_config,
    )

    try:
        # Fetch RGB (returns uint8)
        img_rgb = req_rgb.get_data()[0]
        
        # Fetch DEM (safely handle both 2D and 3D single-band returns)
        dem_data = req_dem.get_data()[0]
        dem_array = np.squeeze(dem_data)
        
        return img_rgb, dem_array
    except Exception as e:
        st.error(f"Sentinel Hub fetch failed: {e}")
        return None, None


def real_unet_predict(img_rgb, unet, device, threshold=0.5):
    """Run real U-Net. Returns (binary_mask, soft_prob_map)."""
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits   = unet(tensor)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.float32)
    return binary_mask, prob_map


def extract_real_features(img_rgb, real_dem=None):
    """
    Extract the 6 conditioning factors used during training.
    Uses real DEM array if provided, else falls back to heavily blurred grayscale.
    """
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    if real_dem is not None:
        dem = cv2.resize(real_dem, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    else:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Apply heavy blur to prevent jagged noise if no true DEM is available
        dem = cv2.GaussianBlur(gray, (31, 31), 0)

    sobel_x = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)
    slope   = np.sqrt(sobel_x**2 + sobel_y**2)

    # Texture is based on the visual image, not the DEM
    gray_visual = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean_l  = uniform_filter(gray_visual, size=5)
    mean_sq = uniform_filter(gray_visual**2, size=5)
    texture = np.sqrt(np.maximum(mean_sq - mean_l**2, 0))

    r = img_resized[:, :, 0].astype(np.float32)
    g = img_resized[:, :, 1].astype(np.float32)
    b = img_resized[:, :, 2].astype(np.float32)

    return {
        "Elevation":  dem    / (dem.max()      + 1e-8),
        "Slope":      slope  / (slope.max()    + 1e-8),
        "Aspect":     (np.arctan2(sobel_y, sobel_x + 1e-8) + np.pi) / (2 * np.pi),
        "Texture":    texture / (texture.max() + 1e-8),
        "NDVI-like":  np.clip((g - r) / (g + r + 1e-8), 0, 1),
        "Brightness": np.clip((r + g + b) / (3.0 * 255.0), 0, 1),
    }


def real_risk_map(img_rgb, cnn, scaler, device, real_dem=None):
    """Run 1D-CNN pixel-by-pixel. Returns float32 risk map (IMG_SIZE, IMG_SIZE)."""
    feats = extract_real_features(img_rgb, real_dem)

    pixel_features = np.stack([
        feats["Elevation"].flatten(),
        feats["Slope"].flatten(),
        feats["Aspect"].flatten(),
        feats["Texture"].flatten(),
        feats["NDVI-like"].flatten(),
        feats["Brightness"].flatten(),
    ], axis=1)                                        # (H*W, 6)

    pixel_scaled = scaler.transform(pixel_features)
    pixel_tensor = torch.tensor(
        pixel_scaled, dtype=torch.float32
    ).unsqueeze(1).to(device)                         # (H*W, 1, 6)

    scores = []
    with torch.no_grad():
        for i in range(0, len(pixel_tensor), 4096):
            chunk = pixel_tensor[i : i + 4096]
            prob  = torch.sigmoid(cnn(chunk)).cpu().numpy().flatten()
            scores.extend(prob)

    return np.array(scores, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)


# ─────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────
def colorize_risk(risk_map):
    cmap    = plt.get_cmap("RdYlGn_r")
    colored = (cmap(risk_map)[:, :, :3] * 255).astype(np.uint8)
    return colored


def overlay_risk(img_rgb, risk_colored, alpha=0.55):
    img     = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    rc      = risk_colored.astype(np.float32)
    blended = np.clip((1 - alpha) * img + alpha * rc, 0, 255).astype(np.uint8)
    return blended


def risk_level(score):
    if score < 0.25:   return "LOW",      "#00c8a0", "✅"
    elif score < 0.50: return "MODERATE", "#f5a623", "⚠️"
    elif score < 0.75: return "HIGH",     "#ff6b35", "🔶"
    else:              return "CRITICAL", "#e84040", "🚨"


def pipeline_bar(stage):
    steps = [
        ("📥", "Input"), ("🔍", "U-Net"), ("📐", "Factors"),
        ("🧠", "1D-CNN"), ("🗺️", "Risk Map"),
    ]
    html = '<div class="pipeline">'
    for i, (icon, label) in enumerate(steps):
        css = "done" if i < stage else ("active" if i == stage else "")
        html += (f'<div class="pipeline-step {css}">'
                 f'<span class="step-icon">{icon}</span>{label}</div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
try:
    unet_model, risk_cnn, scaler, DEVICE = load_models()
    MODEL_OK = True
except Exception as e:
    MODEL_OK  = False
    MODEL_ERR = str(e)


# ═════════════════════════════════════════════
# TOP BANNER
# ═════════════════════════════════════════════
st.markdown("""
<div class="top-banner">
  <span class="banner-icon">🛰️</span>
  <div>
    <p class="banner-title">TerraScan</p>
    <p class="banner-sub">Landslide Intelligence System · Bijie Dataset</p>
  </div>
  <span class="banner-badge">U-NET + 1D-CNN PIPELINE</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-section">System</div>', unsafe_allow_html=True)
    st.markdown("**TerraScan v1.0**")
    st.markdown(
        '<p style="font-size:0.78rem;color:#64748b;">Satellite landslide detection '
        'using deep learning segmentation + risk scoring.</p>',
        unsafe_allow_html=True,
    )

    if MODEL_OK:
        st.markdown(
            '<div style="background:rgba(0,200,160,0.1);border:1px solid #00c8a0;'
            'border-radius:6px;padding:0.5rem 0.8rem;font-family:monospace;'
            'font-size:0.72rem;color:#00c8a0;margin:0.5rem 0;">✅ Models loaded</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size:0.7rem;color:#64748b;font-family:monospace;">'
            f'Device · {str(DEVICE).upper()}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(232,64,64,0.1);border:1px solid #e84040;'
            'border-radius:6px;padding:0.5rem 0.8rem;font-family:monospace;'
            'font-size:0.72rem;color:#e84040;margin:0.5rem 0;">⚠️ Models not found</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section">Pipeline Config</div>', unsafe_allow_html=True)
    threshold   = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                            help="U-Net sigmoid threshold for binary mask")
    img_size    = st.selectbox("Display Resolution", [128, 256, 512], index=1)
    risk_alpha  = st.slider("Risk Overlay Opacity", 0.1, 0.9, 0.55, 0.05)

    st.markdown('<div class="sidebar-section">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.75rem;line-height:1.8;color:#94a3b8;">
  <b style="color:#e2e8f0;">Step 1</b> &nbsp;U-Net (ResNet-34)<br>
  <b style="color:#e2e8f0;">Step 2</b> &nbsp;Factor Extraction<br>
  <b style="color:#e2e8f0;">Step 3</b> &nbsp;1D-CNN Risk Score<br>
  <br>
  Backbone · ImageNet pretrained<br>
  Loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Dice + BCE<br>
  Dataset &nbsp;· Bijie Landslide
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.73rem;color:#64748b;">Built for geospatial hazard mapping. '
        'Upload a satellite image to begin analysis.</p>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════
# MAIN TABS
# ═════════════════════════════════════════════
tab_analyze, tab_location, tab_explore, tab_about = st.tabs(
    ["🔬 Analyze", "📍 Fetch by Location", "📊 Explore Factors", "ℹ️ About"]
)


# ══════════════════════════════════════════════
# TAB 1 — ANALYZE
# ══════════════════════════════════════════════
with tab_analyze:

    if not MODEL_OK:
        st.markdown(
            f'<div class="danger-box">⚠️ <b>Models could not be loaded.</b> '
            f'Ensure <code>models/best_unet_model.pth</code>, '
            f'<code>models/best_risk_cnn.pth</code>, and '
            f'<code>models/scaler.pkl</code> exist.<br><br>'
            f'<code style="font-size:0.75rem;">{MODEL_ERR}</code></div>',
            unsafe_allow_html=True,
        )

    col_upload, col_results = st.columns([1, 2], gap="large")

    # ── Upload panel ──
    with col_upload:
        st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a satellite image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            label_visibility="collapsed",
        )
        use_demo = st.button("⚡ Use Demo Image", use_container_width=True)

        img_pil = None
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
        elif "fetched_img" in st.session_state:
            img_pil = Image.fromarray(st.session_state["fetched_img"])
            st.markdown(
                '<div class="info-box">🛰️ Using Sentinel-2 image fetched from '
                f'<b>{st.session_state["fetched_lat"]:.4f}, '
                f'{st.session_state["fetched_lon"]:.4f}</b></div>',
                unsafe_allow_html=True,
            )
        elif use_demo:
            rng  = np.random.default_rng(99)
            demo = (rng.random((256, 256, 3)) * 80 + 60).astype(np.uint8)
            demo[:, :, 0] = np.clip(demo[:, :, 0] + 40, 0, 200)
            demo[:, :, 1] = np.clip(demo[:, :, 1] + 20, 0, 180)
            img_pil = Image.fromarray(demo)

        if img_pil is not None:
            img_np = np.array(img_pil.resize((img_size, img_size)))
            st.markdown(
                '<div class="img-card"><div class="img-card-header">'
                '<span class="dot dot-green"></span>Input · '
                f'{img_np.shape[1]}×{img_np.shape[0]}px</div></div>',
                unsafe_allow_html=True,
            )
            st.image(img_np, use_container_width=True)
            st.markdown(f"""
<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem;">
  <span style="background:#1e2d40;padding:0.2rem 0.6rem;border-radius:4px;
               font-size:0.68rem;font-family:monospace;color:#94a3b8;">
    RES&nbsp;{img_size}×{img_size}
  </span>
  <span style="background:#1e2d40;padding:0.2rem 0.6rem;border-radius:4px;
               font-size:0.68rem;font-family:monospace;color:#94a3b8;">
    CH&nbsp;RGB+DEM
  </span>
  <span style="background:#1e2d40;padding:0.2rem 0.6rem;border-radius:4px;
               font-size:0.68rem;font-family:monospace;color:#94a3b8;">
    THR&nbsp;{threshold}
  </span>
</div>
""", unsafe_allow_html=True)

    # ── Results panel ──
    with col_results:
        if img_pil is None:
            pipeline_bar(0)
            st.markdown("""
<div class="info-box">
  Upload a satellite image, fetch one from the <b>📍 Location Tab</b>, or click <b>Use Demo Image</b> to run the full
  U-Net → Feature Extraction → 1D-CNN pipeline.
</div>
""", unsafe_allow_html=True)

        elif not MODEL_OK:
            st.markdown(
                '<div class="warn-box">⚠️ Cannot run inference — models failed to load.</div>',
                unsafe_allow_html=True,
            )

        else:
            img_rgb = np.array(img_pil.convert("RGB"))

            # ── Step 1: U-Net ──
            pipeline_bar(1)
            st.markdown(
                '<div class="section-label">Step 1 · U-Net Segmentation</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Running U-Net inference…"):
                binary_mask, prob_map = real_unet_predict(
                    img_rgb, unet_model, DEVICE, threshold
                )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    '<div class="img-card"><div class="img-card-header">'
                    '<span class="dot dot-green"></span>Predicted Mask</div></div>',
                    unsafe_allow_html=True,
                )
                st.image((binary_mask * 255).astype(np.uint8),
                         use_container_width=True, clamp=True)
            with c2:
                img_display = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                seg_overlay = img_display.copy()
                seg_overlay[binary_mask.astype(bool)] = [220, 50, 50]
                st.markdown(
                    '<div class="img-card"><div class="img-card-header">'
                    '<span class="dot dot-orange"></span>Segmentation Overlay</div></div>',
                    unsafe_allow_html=True,
                )
                st.image(seg_overlay, use_container_width=True)

            detected_pct = binary_mask.mean() * 100
            avg_conf     = (float(prob_map[binary_mask.astype(bool)].mean())
                            if binary_mask.any() else 0.0)

            st.markdown(f"""
<div class="metric-row">
  <div class="metric-card green">
    <div class="metric-label">Detected Area</div>
    <div class="metric-value">{detected_pct:.1f}%</div>
  </div>
  <div class="metric-card orange">
    <div class="metric-label">Avg Confidence</div>
    <div class="metric-value">{avg_conf:.3f}</div>
  </div>
  <div class="metric-card yellow">
    <div class="metric-label">Threshold</div>
    <div class="metric-value">{threshold}</div>
  </div>
</div>
""", unsafe_allow_html=True)

            st.divider()

            # ── Step 3: 1D-CNN Risk Map ──
            pipeline_bar(4)
            st.markdown(
                '<div class="section-label">Step 3 · 1D-CNN Risk Heatmap</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Extracting features & running 1D-CNN risk inference…"):
                # Retrieve the true DEM array if fetched, else None
                actual_dem = st.session_state.get("fetched_dem", None) if ("fetched_img" in st.session_state and not uploaded and not use_demo) else None
                
                risk_map = real_risk_map(img_rgb, risk_cnn, scaler, DEVICE, real_dem=actual_dem)

            risk_colored = colorize_risk(risk_map)
            blended      = overlay_risk(img_rgb, risk_colored, alpha=risk_alpha)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown(
                    '<div class="img-card"><div class="img-card-header">'
                    '<span class="dot dot-yellow"></span>Risk Heatmap</div></div>',
                    unsafe_allow_html=True,
                )
                st.image(risk_colored, use_container_width=True)
            with c4:
                st.markdown(
                    '<div class="img-card"><div class="img-card-header">'
                    '<span class="dot dot-orange"></span>Risk Overlay</div></div>',
                    unsafe_allow_html=True,
                )
                st.image(blended, use_container_width=True)

            # Overall risk score
            mean_risk = (float(risk_map[binary_mask.astype(bool)].mean())
                         if binary_mask.any() else float(risk_map.mean()))
            level, color, icon = risk_level(mean_risk)

            st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;
            padding:1.2rem 1.5rem;margin-top:1rem;">
  <div class="gauge-label">Overall Landslide Risk Score</div>
  <div class="gauge-bar-bg">
    <div class="gauge-bar-fill" style="width:{mean_risk*100:.1f}%"></div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.6rem;">
    <span class="gauge-value" style="color:{color};">{mean_risk*100:.1f}%</span>
    <span style="font-family:monospace;font-size:1rem;
                 background:rgba(255,255,255,0.05);border:1px solid {color};
                 color:{color};padding:0.3rem 0.9rem;border-radius:6px;
                 letter-spacing:0.08em;">{icon} {level}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            if level == "CRITICAL":
                st.markdown(
                    '<div class="danger-box">🚨 <b>Critical risk detected.</b> '
                    'Immediate field inspection recommended. '
                    'Evacuate vulnerable populations in affected zones.</div>',
                    unsafe_allow_html=True,
                )
            elif level == "HIGH":
                st.markdown(
                    '<div class="warn-box">🔶 <b>High risk area.</b> '
                    'Schedule inspection and review drainage infrastructure.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="info-box">✅ Risk levels within acceptable range. '
                    'Continue standard monitoring protocols.</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════
# TAB 2 — FETCH BY LOCATION
# ══════════════════════════════════════════════
with tab_location:
    st.markdown(
        '<div class="section-label">Location Selection · Sentinel-2 Fetch</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
<div class="info-box">
  🗺️ <b>Search</b> for a place, input <b>coordinates</b> manually, or <b>click anywhere</b> on the map.
  Then click <b>Fetch Satellite Image</b> to pull a real Sentinel-2 scene and Copernicus DEM.
</div>
""", unsafe_allow_html=True)

    # Initialize state variables for the map
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = [10.0889, 77.0595] # Default: Munnar
    if "selected_point" not in st.session_state:
        st.session_state["selected_point"] = None

    col_map, col_fetch = st.columns([2, 1], gap="large")

    # ── Controls Panel (Right Side) ──
    with col_fetch:
        # 1. Search by Text
        st.markdown('<div class="section-label">1. Search Location</div>', unsafe_allow_html=True)
        search_query = st.text_input("Address, City, or Landmark", placeholder="e.g., Wayanad, Kerala", label_visibility="collapsed")
        if st.button("🔍 Search", use_container_width=True):
            if search_query:
                with st.spinner("Searching..."):
                    try:
                        geolocator = Nominatim(user_agent="terrascan_app")
                        loc = geolocator.geocode(search_query)
                        if loc:
                            st.session_state["map_center"] = [loc.latitude, loc.longitude]
                            st.session_state["selected_point"] = [loc.latitude, loc.longitude]
                            st.rerun()
                        else:
                            st.error("Location not found. Try being more specific.")
                    except Exception as e:
                        st.error("Geocoding service unavailable.")

        # 2. Manual Coordinates
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">2. Manual Coordinates</div>', unsafe_allow_html=True)
        c_lat, c_lon = st.columns(2)
        with c_lat:
            man_lat = st.number_input("Latitude", value=st.session_state["map_center"][0], format="%.5f", step=0.01)
        with c_lon:
            man_lon = st.number_input("Longitude", value=st.session_state["map_center"][1], format="%.5f", step=0.01)

        if st.button("📍 Set Coordinates", use_container_width=True):
            st.session_state["map_center"] = [man_lat, man_lon]
            st.session_state["selected_point"] = [man_lat, man_lon]
            st.rerun()

        # 3. Fetch Panel
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">3. Fetch Settings</div>', unsafe_allow_html=True)
        
        target_lat = st.session_state["selected_point"][0] if st.session_state["selected_point"] else None
        target_lon = st.session_state["selected_point"][1] if st.session_state["selected_point"] else None

        if target_lat is not None and target_lon is not None:
            st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);
            border-radius:8px;padding:0.6rem 0.8rem;font-family:monospace;
            font-size:0.75rem;line-height:1.6;margin-bottom:0.8rem;">
  <span style="color:#64748b;">LAT</span>&nbsp; {target_lat:.5f}<br>
  <span style="color:#64748b;">LON</span>&nbsp; {target_lon:.5f}
</div>
""", unsafe_allow_html=True)

            area_km = st.slider("Coverage area (km²)", 1.0, 10.0, 2.0, 0.5)

            if st.button("🛰️ Fetch Satellite Image", use_container_width=True):
                with st.spinner("Contacting Sentinel Hub for S2 & DEM…"):
                    fetched_img, fetched_dem = fetch_sentinel_data(target_lat, target_lon, size_km=area_km)

                if fetched_img is not None:
                    # Save to session so Analyze tab can pick it up
                    st.session_state["fetched_img"] = fetched_img
                    st.session_state["fetched_dem"] = fetched_dem
                    st.session_state["fetched_lat"] = target_lat
                    st.session_state["fetched_lon"] = target_lon

                    st.markdown('<div class="section-label" style="margin-top:1rem;">Fetched Image</div>', unsafe_allow_html=True)
                    st.image(fetched_img, use_container_width=True)
                    st.markdown("""
<div class="info-box">
  ✅ Image fetched! Switch to the
  <b>🔬 Analyze</b> tab — it will load automatically.
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="color:#64748b;font-size:0.82rem;font-family:monospace;
            padding:1rem;text-align:center;border:1px dashed var(--border);
            border-radius:8px;margin-top:1rem;">
  No location selected yet.
</div>
""", unsafe_allow_html=True)

    # ── Map Panel (Left Side) ──
    with col_map:
        m = folium.Map(
            location=st.session_state["map_center"],
            zoom_start=12,
            tiles="Esri.WorldImagery",
            attr="Esri"
        )
        # Adds ArcGIS satellite basemap
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
        ).add_to(m)

        # Place a red marker if a point has been selected
        if st.session_state["selected_point"]:
            folium.Marker(
                st.session_state["selected_point"],
                icon=folium.Icon(color="red", icon="info-sign"),
                tooltip="Target Area"
            ).add_to(m)

        # Render the map
        map_data = st_folium(m, width=600, height=520, key="interactive_map")

        # Handle map clicks
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            
            curr_pt = st.session_state["selected_point"]
            # Only rerun the app if the clicked coordinates actually changed
            if curr_pt is None or (round(click_lat, 5) != round(curr_pt[0], 5) or round(click_lon, 5) != round(curr_pt[1], 5)):
                st.session_state["selected_point"] = [click_lat, click_lon]
                st.rerun()


# ══════════════════════════════════════════════
# TAB 3 — EXPLORE FACTORS
# ══════════════════════════════════════════════
with tab_explore:
    st.markdown(
        '<div class="section-label">Conditioning Factors · Visual Inspector</div>',
        unsafe_allow_html=True,
    )

    factor_uploaded  = st.file_uploader(
        "Upload image to inspect factors",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="factor_upload",
    )
    use_demo_factors = st.button("⚡ Use Demo Image", key="demo_factors")

    fimg = None
    if factor_uploaded:
        fimg = np.array(Image.open(factor_uploaded).convert("RGB").resize((256, 256)))
    elif use_demo_factors:
        rng2  = np.random.default_rng(77)
        demo2 = (rng2.random((256, 256, 3)) * 80 + 60).astype(np.uint8)
        demo2[:, :, 0] = np.clip(demo2[:, :, 0] + 40, 0, 200)
        fimg  = demo2

    if fimg is not None:
        st.markdown('<div class="section-label">Source Image</div>', unsafe_allow_html=True)
        c_src, _ = st.columns([1, 3])
        with c_src:
            st.image(fimg, use_container_width=True)

        with st.spinner("Extracting conditioning factors…"):
            feats = extract_real_features(fimg)

        st.markdown('<div class="section-label">Extracted Factors</div>', unsafe_allow_html=True)

        cmaps   = ["terrain", "YlOrRd", "hsv", "magma", "RdYlGn", "gray"]
        sources = ["DEM-proxy", "DEM-proxy", "DEM-proxy", "Image", "Image", "Image"]
        cols    = st.columns(3)

        for i, (name, arr) in enumerate(feats.items()):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(3.2, 2.8))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                im = ax.imshow(arr, cmap=cmaps[i])
                ax.axis("off")
                ax.set_title(name, color="#e2e8f0", fontsize=9,
                             fontfamily="monospace", pad=4)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
                    labelsize=6, colors="#64748b"
                )
                plt.tight_layout(pad=0.3)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    f'<div style="text-align:center;font-size:0.65rem;font-family:monospace;'
                    f'color:#64748b;margin-top:-0.4rem;margin-bottom:0.8rem;">'
                    f'source · {sources[i]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown("""
<div class="info-box">
  Upload a satellite image to visualise the six conditioning factors used by
  the 1D-CNN: <b>Elevation, Slope, Aspect, Texture, NDVI-like,</b> and <b>Brightness</b>.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════
with tab_about:
    c_l, c_r = st.columns([3, 2], gap="large")

    with c_l:
        st.markdown('<div class="section-label">System Overview</div>', unsafe_allow_html=True)
        st.markdown("""
<p style="line-height:1.8;color:#94a3b8;font-size:0.88rem;">
TerraScan is an end-to-end landslide susceptibility mapping system trained on the
<b style="color:#e2e8f0;">Bijie Landslide Dataset</b> from Bijie City, Guizhou, China.
The pipeline combines semantic segmentation with tabular risk scoring to produce
per-pixel hazard maps from multispectral satellite imagery.
</p>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
        arch_steps = [
            ("🛰️", "Step 1 · U-Net Segmentation",
             "ResNet-34 encoder pretrained on ImageNet. Trained with Dice + BCE loss "
             "to handle severe class imbalance (~8% landslide pixels). Output: binary mask."),
            ("📐", "Step 2 · Feature Extraction",
             "Six conditioning factors extracted per pixel: Elevation, Slope, Aspect "
             "(from real DEM), plus Texture, NDVI-like index, and Brightness from RGB."),
            ("🧠", "Step 3 · 1D-CNN Risk Scoring",
             "Lightweight 1D convolutional network trained on balanced tabular data. "
             "Outputs a continuous risk probability [0,1] per pixel."),
            ("🗺️", "Step 4 · Risk Map Fusion",
             "U-Net mask + CNN risk scores blended into a GIS-ready heatmap for "
             "field team dispatch and evacuation planning."),
        ]
        steps_html = '<div style="display:flex;flex-direction:column;gap:0.7rem;margin:0.5rem 0;">'
        for icon, title, desc in arch_steps:
            steps_html += f"""
  <div style="display:flex;gap:0.8rem;background:var(--surface);
              border:1px solid var(--border);border-radius:8px;padding:0.8rem 1rem;">
    <span style="font-size:1.4rem;line-height:1.2;">{icon}</span>
    <div>
      <div style="font-family:monospace;font-size:0.75rem;color:#00c8a0;
                  letter-spacing:0.05em;margin-bottom:0.2rem;">{title}</div>
      <div style="font-size:0.8rem;color:#94a3b8;line-height:1.6;">{desc}</div>
    </div>
  </div>"""
        steps_html += "</div>"
        st.markdown(steps_html, unsafe_allow_html=True)

    with c_r:
        st.markdown('<div class="section-label">Performance (U-Net)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div class="metric-row" style="flex-direction:column;">
  <div class="metric-card green">
    <div class="metric-label">Dice Coefficient</div>
    <div class="metric-value">0.81</div>
  </div>
  <div class="metric-card orange">
    <div class="metric-label">IoU (Jaccard)</div>
    <div class="metric-value">0.74</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Performance (1D-CNN)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div class="metric-row" style="flex-direction:column;">
  <div class="metric-card yellow">
    <div class="metric-label">Test Accuracy</div>
    <div class="metric-value">91.3%</div>
  </div>
  <div class="metric-card green">
    <div class="metric-label">ROC-AUC</div>
    <div class="metric-value">0.964</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:0.78rem;line-height:2;color:#94a3b8;
            background:var(--surface);border:1px solid var(--border);
            border-radius:8px;padding:0.9rem 1rem;">
  <b style="color:#e2e8f0;">Landslide images</b> &nbsp;·&nbsp; 770<br>
  <b style="color:#e2e8f0;">Non-landslide</b> &nbsp;·&nbsp; 2450<br>
  <b style="color:#e2e8f0;">Resolution</b> &nbsp;·&nbsp; 256 × 256 px<br>
  <b style="color:#e2e8f0;">Channels</b> &nbsp;·&nbsp; RGB + DEM<br>
  <b style="color:#e2e8f0;">Location</b> &nbsp;·&nbsp; Bijie, Guizhou, China
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">File Structure</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:0.75rem;line-height:2;color:#94a3b8;
            background:var(--surface);border:1px solid var(--border);
            border-radius:8px;padding:0.9rem 1rem;font-family:monospace;">
  DEMO/<br>
  ├── models/<br>
  │&nbsp;&nbsp;&nbsp;├── best_unet_model.pth<br>
  │&nbsp;&nbsp;&nbsp;├── best_risk_cnn.pth<br>
  │&nbsp;&nbsp;&nbsp;└── scaler.pkl<br>
  ├── app.py<br>
  └── requirements.txt
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;
            font-family:monospace;font-size:0.65rem;color:#374151;
            letter-spacing:0.1em;">
  TERRASCAN · LANDSLIDE INTELLIGENCE · BIJIE DATASET · U-NET + 1D-CNN PIPELINE
</div>
""", unsafe_allow_html=True)