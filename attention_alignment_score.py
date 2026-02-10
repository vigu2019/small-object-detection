import os
import cv2
import csv
import numpy as np
from ultralytics_backup import YOLO

# ==========================
# CONFIG
# ==========================

DATASET = "visdrone"

BASELINE_MODEL = "baseline.pt"
ENHANCED_MODEL = "best.pt"

BASELINE_IMG_DIR = "runs/detect/predict3"
ENHANCED_IMG_DIR = "runs/detect/predict7"

BASELINE_HEAT_DIR = "gradcam_output/visdrone/baseline/heatmaps"
ENHANCED_HEAT_DIR = "gradcam_output/visdrone/enhanced/heatmaps"

OUT_CSV = f"gradcam_output/{DATASET}_attention_alignment_score_VISDRONE.csv"

# ==========================
# LOAD MODELS
# ==========================

baseline_model = YOLO(BASELINE_MODEL)
enhanced_model = YOLO(ENHANCED_MODEL)

rows = [("image", "baseline_avg_attention", "enhanced_avg_attention")]

images = sorted(os.listdir(BASELINE_IMG_DIR))

# ==========================
# HELPER FUNCTION
# ==========================

def average_attention(res, heat):
    """
    Computes average heatmap intensity inside detected boxes.
    """
    if res.boxes is None or len(res.boxes) == 0:
        return 0.0

    scores = []

    for box in res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)

        # clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(heat.shape[1] - 1, x2)
        y2 = min(heat.shape[0] - 1, y2)

        region = heat[y1:y2, x1:x2]
        if region.size > 0:
            scores.append(region.mean())

    return float(np.mean(scores)) if scores else 0.0

# ==========================
# PROCESS
# ==========================

for img in images:
    base_img_path = os.path.join(BASELINE_IMG_DIR, img)
    enh_img_path  = os.path.join(ENHANCED_IMG_DIR, img)

    base_heat_path = os.path.join(BASELINE_HEAT_DIR, img)
    enh_heat_path  = os.path.join(ENHANCED_HEAT_DIR, img)

    if not os.path.exists(enh_img_path):
        continue

    base_heat = cv2.imread(base_heat_path, cv2.IMREAD_GRAYSCALE)
    enh_heat  = cv2.imread(enh_heat_path, cv2.IMREAD_GRAYSCALE)

    if base_heat is None or enh_heat is None:
        continue

    # normalize heatmaps
    base_heat = base_heat.astype(np.float32) / 255.0
    enh_heat  = enh_heat.astype(np.float32) / 255.0

    base_res = baseline_model.predict(base_img_path, conf=0.05, verbose=False)[0]
    enh_res  = enhanced_model.predict(enh_img_path, conf=0.05, verbose=False)[0]

    base_score = average_attention(base_res, base_heat)
    enh_score  = average_attention(enh_res, enh_heat)

    rows.append((img, round(base_score, 4), round(enh_score, 4)))

# ==========================
# SAVE CSV
# ==========================

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("Saved Attention Alignment Scores to:", OUT_CSV)
