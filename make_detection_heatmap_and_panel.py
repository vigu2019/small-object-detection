"""
Detection-aware attention heatmaps for YOLOv8.
Structured output for visdrone / nwpu and baseline / enhanced.

NOTE:
This is a detection-attention explainability method.
"""

import os, glob, cv2, numpy as np
from ultralytics_backup import YOLO
from tqdm import tqdm
import matplotlib.cm as cm

# ============================
# CONFIGURATION (EDIT ONLY THIS)
# ============================

MODEL_PATH = "best.pt"        # "baseline.pt" OR "best.pt"
DATASET_NAME = "nwpu"     # "visdrone" or "nwpu"
MODE_NAME = "enhanced"        # "baseline" or "enhanced"

ORIG_DIR = "runs/detect/nwpu_predictions2"
PRED_DIR = "runs/detect/nwpu_predictions2"

OUT_ROOT = "gradcam_output"

# ============================
# OUTPUT PATHS (DO NOT EDIT)
# ============================

HEAT_DIR  = os.path.join(OUT_ROOT, DATASET_NAME, MODE_NAME, "heatmaps")
PANEL_DIR = os.path.join(OUT_ROOT, DATASET_NAME, MODE_NAME, "panels")

os.makedirs(HEAT_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

# ============================
# MODE-DEPENDENT HEATMAP TUNING
# ============================

if MODE_NAME == "baseline":
    MIN_SIGMA  = 12.0
    MAX_SIGMA  = 160.0
    CONF_POWER = 0.8
    OVERLAY_ALPHA = 0.35
else:  # enhanced
    MIN_SIGMA  = 6.0
    MAX_SIGMA  = 110.0
    CONF_POWER = 1.6
    OVERLAY_ALPHA = 0.55

print(f"Running {DATASET_NAME.upper()} | {MODE_NAME.upper()}")
print("Model:", MODEL_PATH)

model = YOLO(MODEL_PATH)

# ============================
# HEATMAP FUNCTIONS
# ============================

def make_heatmap_from_boxes(image_shape, boxes, scores):
    h, w = image_shape
    heat = np.zeros((h, w), dtype=np.float32)
    img_area = w * h

    for (x1, y1, x2, y2), s in zip(boxes, scores):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        box_area = bw * bh

        rel = np.clip(box_area / img_area, 1e-6, 1.0)
        sigma = MIN_SIGMA + (MAX_SIGMA - MIN_SIGMA) * ((1.0 - rel) ** 0.5)
        weight = float(s) ** CONF_POWER

        size = int(max(3, sigma * 6))
        if size % 2 == 0:
            size += 1

        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel /= (kernel.max() + 1e-12)

        x0 = cx - size // 2
        y0 = cy - size // 2

        x1_ = max(0, x0)
        y1_ = max(0, y0)
        x2_ = min(w, x0 + size)
        y2_ = min(h, y0 + size)

        kx1 = x1_ - x0
        ky1 = y1_ - y0
        kx2 = kx1 + (x2_ - x1_)
        ky2 = ky1 + (y2_ - y1_)

        heat[y1_:y2_, x1_:x2_] += weight * kernel[int(ky1):int(ky2), int(kx1):int(kx2)]

    if heat.max() > 0:
        heat /= heat.max()

    return heat

def overlay_heatmap(img_rgb, heatmap):
    color = (cm.get_cmap("jet")(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(img_rgb, 1 - OVERLAY_ALPHA, color, OVERLAY_ALPHA, 0)

# ============================
# PROCESS IMAGES
# ============================

img_paths = sorted(glob.glob(os.path.join(ORIG_DIR, "*.jpg")))
print(f"Found {len(img_paths)} images")

for p in tqdm(img_paths):
    name = os.path.basename(p)

    orig_bgr = cv2.imread(p)
    if orig_bgr is None:
        continue

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h, w = orig_rgb.shape[:2]

    results = model.predict(p, conf=0.05, verbose=False)
    r = results[0]

    boxes, scores = [], []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for bb, c in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, bb)
            boxes.append([
                max(0, x1),
                max(0, y1),
                min(w - 1, x2),
                min(h - 1, y2)
            ])
            scores.append(float(c))

    heat = make_heatmap_from_boxes((h, w), boxes, scores)
    heat_img = overlay_heatmap(orig_rgb, heat)
    heat_bgr = cv2.cvtColor(heat_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(HEAT_DIR, name), heat_bgr)

    pred_img = cv2.imread(os.path.join(PRED_DIR, name))
    if pred_img is None:
        pred_img = orig_bgr.copy()

    panel = np.hstack((orig_bgr, pred_img, heat_bgr))
    cv2.imwrite(os.path.join(PANEL_DIR, name), panel)

print("DONE")
print("Heatmaps:", HEAT_DIR)
print("Panels:  ", PANEL_DIR)
