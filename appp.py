# app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np

BASE = Path(__file__).parent

INPUT_DIR = BASE / "stream_input"
BASELINE_DIR = BASE / "stream_baseline"
ENHANCED_DIR = BASE / "stream_enhanced"

def list_sample_ids():
    # Collect common filenames (without extension)
    input_ids = {p.stem for p in INPUT_DIR.glob("*")}
    base_ids  = {p.stem for p in BASELINE_DIR.glob("*")}
    enh_ids   = {p.stem for p in ENHANCED_DIR.glob("*")}
    
    ids = sorted(list(input_ids & base_ids & enh_ids))
    return ids

def load_image(path: Path):
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")

st.set_page_config(layout="wide", page_title="Visual Comparison")
st.title("Visual Results Comparison — Baseline vs Enhanced")

sample_ids = list_sample_ids()

if not sample_ids:
    st.warning("No matching images found across stream_input/, stream_baseline/, and stream_enhanced/.")
    st.stop()

sample = st.selectbox("Choose sample", sample_ids, index=0)

col1, col2, col3 = st.columns(3)

input_img = load_image(INPUT_DIR / f"{sample}.png") or \
            load_image(INPUT_DIR / f"{sample}.jpg")

baseline_img = load_image(BASELINE_DIR / f"{sample}.png") or \
               load_image(BASELINE_DIR / f"{sample}.jpg")

enhanced_img = load_image(ENHANCED_DIR / f"{sample}.png") or \
               load_image(ENHANCED_DIR / f"{sample}.jpg")

opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.5, step=0.05)
show_diff = st.sidebar.checkbox("Show difference image", False)

with col1:
    st.subheader("Input")
    if input_img:
        st.image(input_img, width="stretch")
    else:
        st.write("No input image found.")

with col2:
    st.subheader("Baseline")
    if baseline_img:
        st.image(baseline_img, width="stretch")
    else:
        st.write("No baseline image found.")

with col3:
    st.subheader("Enhanced")
    if enhanced_img:
        st.image(enhanced_img, width="stretch")
    else:
        st.write("No enhanced image found.")

# Optional Difference View
if show_diff and baseline_img and enhanced_img:
    b = np.array(baseline_img.resize((256,256))).astype(int)
    e = np.array(enhanced_img.resize((256,256))).astype(int)
    diff = np.clip(np.abs(b - e), 0, 255).astype(np.uint8)
    st.subheader("Difference (Resized Visualization)")
    st.image(diff, width="stretch")