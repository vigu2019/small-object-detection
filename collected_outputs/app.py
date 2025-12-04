# app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import re

# --- Configure paths (update only if your folders are somewhere else) ---
BASE = Path(__file__).parent
DATASET_DIR = BASE / "member1_dataset_samples"
YOLO_DIR    = BASE / "member2_yolo_outputs"
GRADCAM_DIR = BASE / "member3_gradcam"

# --- Helpers ---
COMMON_SUFFIXES = ["_yolo", "_pred", "_preds", "_improved", "_baseline", "_gradcam", "_overlay", "_cam"]

def normalize_stem(stem: str) -> str:
    """Remove common suffixes to produce a sample id."""
    s = stem
    for suf in COMMON_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    # also remove trailing dash or underscore numbers if present like img-001_a -> img-001
    s = re.sub(r'[_\-]?(copy|final|v\d+)$', '', s, flags=re.IGNORECASE)
    return s

def gather_sample_ids():
    stems = []
    for folder in (DATASET_DIR, YOLO_DIR, GRADCAM_DIR):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.is_file():
                stems.append(normalize_stem(p.stem))
    return sorted(set(stems))

def find_file_with_prefix(folder: Path, sample_id: str):
    """Find a file in folder that matches sample_id after normalization."""
    if not folder.exists():
        return None
    # direct matches first
    for ext in ("png","jpg","jpeg","bmp","tiff","webp"):
        candidate = folder / f"{sample_id}.{ext}"
        if candidate.exists():
            return candidate
    # look for files whose normalized stem == sample_id
    for p in folder.iterdir():
        if not p.is_file(): 
            continue
        if normalize_stem(p.stem) == sample_id:
            return p
    # last resort: file whose stem startswith sample_id
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.stem.startswith(sample_id):
            return p
    return None

def load_img(path: Path, size=None):
    if path is None or not Path(path).exists():
        return None
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size)
    return im

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Visual Results Comparison (YOLO + GradCAM)")
st.title("Visual Results Comparison — YOLO outputs & Grad-CAM")

sample_ids = gather_sample_ids()
if not sample_ids:
    st.warning("No images found in your folders. Make sure member1_dataset_samples/, member2_yolo_outputs/, member3_gradcam/ exist and have images.")
    st.stop()

# sample selection
sample = st.selectbox("Choose sample ID", sample_ids, index=0)

# sidebar options
opacity = st.sidebar.slider("Grad-CAM overlay opacity", 0.0, 1.0, 0.5, step=0.05)
show_diff = st.sidebar.checkbox("Show visual difference (baseline vs YOLO)", False)
resize_preview = st.sidebar.checkbox("Force-preview size 512x512", True)

# find paths
orig_path   = find_file_with_prefix(DATASET_DIR, sample)
yolo_path   = find_file_with_prefix(YOLO_DIR, sample)
gradcam_path= find_file_with_prefix(GRADCAM_DIR, sample)

# load images
preview_size = (512,512) if resize_preview else None
orig_img  = load_img(orig_path, preview_size)
yolo_img  = load_img(yolo_path, preview_size)
grad_img  = load_img(gradcam_path, preview_size)

# layout columns
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.subheader("Original")
    if orig_img:
        st.image(orig_img, use_column_width=True)
        st.caption(orig_path.name if orig_path else "")
    else:
        st.write("No original image found for this sample.")

with col2:
    st.subheader("YOLO output / Predictions")
    if yolo_img:
        st.image(yolo_img, use_column_width=True)
        st.caption(yolo_path.name if yolo_path else "")
    else:
        st.write("No YOLO output found for this sample.")

with col3:
    st.subheader("Grad-CAM (overlay)")
    if grad_img:
        # if gradcam is already an overlay, show it; else blend with orig or yolo
        base = yolo_img or orig_img
        if base:
            grad_resized = grad_img.resize(base.size)
            blended = Image.blend(base.convert("RGBA"), grad_resized.convert("RGBA"), opacity)
            st.image(blended, use_column_width=True)
            st.caption(gradcam_path.name)
        else:
            st.image(grad_img, use_column_width=True)
            st.caption(gradcam_path.name)
    else:
        st.write("No Grad-CAM image found for this sample.")

# optional diff view
if show_diff and orig_img and yolo_img:
    st.subheader("Absolute difference (visual, resized)")
    # convert to arrays, compute abs diff
    b = np.array(orig_img).astype(int)
    p = np.array(yolo_img).astype(int)
    # ensure same shape
    if b.shape != p.shape:
        # resize to the smaller size
        min_h = min(b.shape[0], p.shape[0])
        min_w = min(b.shape[1], p.shape[1])
        b = b[:min_h, :min_w]
        p = p[:min_h, :min_w]
    diff = np.clip(np.abs(b - p).astype(np.uint8), 0, 255)
    st.image(diff, use_column_width=True)

# thumbnails / quick gallery for fast scan
st.markdown("---")
st.subheader("Thumbnail gallery (first 20 samples)")
thumb_cols = st.columns(5)
for idx, sid in enumerate(sample_ids[:20]):
    col = thumb_cols[idx % 5]
    thumb_yolo = find_file_with_prefix(YOLO_DIR, sid)
    if thumb_yolo:
        img = load_img(thumb_yolo, size=(160,160))
        col.image(img, caption=sid)
    else:
        col.write(sid)
