import os
import cv2
import numpy as np

# ==========================
# CONFIG (EDIT ONLY THIS)
# ==========================

DATASET = "nwpu"  # "nwpu" or "visdrone"

BASELINE_PANEL_DIR = f"gradcam_output/{DATASET}/baseline/panels"
ENHANCED_PANEL_DIR = f"gradcam_output/{DATASET}/enhanced/panels"

OUT_DIR = f"gradcam_output/{DATASET}/comparison_panels"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# PROCESS
# ==========================

images = sorted(os.listdir(BASELINE_PANEL_DIR))

for name in images:
    base_path = os.path.join(BASELINE_PANEL_DIR, name)
    enh_path  = os.path.join(ENHANCED_PANEL_DIR, name)

    if not os.path.exists(enh_path):
        continue

    base_img = cv2.imread(base_path)
    enh_img  = cv2.imread(enh_path)

    if base_img is None or enh_img is None:
        continue

    # Resize to same height
    h = min(base_img.shape[0], enh_img.shape[0])
    base_img = cv2.resize(base_img, (base_img.shape[1], h))
    enh_img  = cv2.resize(enh_img, (enh_img.shape[1], h))

    comparison = np.hstack((base_img, enh_img))
    cv2.imwrite(os.path.join(OUT_DIR, name), comparison)

print("Comparison panels saved in:", OUT_DIR)
