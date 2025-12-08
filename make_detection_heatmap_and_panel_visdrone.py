# make_detection_heatmap_and_panel_visdrone.py
"""
VisDrone baseline explainability.

Uses baseline.pt (YOLOv8 baseline trained for VisDrone-style data)
and generates, for each image in gradcam_output_visdrone/originals:

    gradcam_output_visdrone/predictions/<img>  - detection result (boxes + labels)
    gradcam_output_visdrone/heatmaps/<img>     - heatmap overlay
    gradcam_output_visdrone/panels/<img>       - ORIGINAL | PREDICTION | HEATMAP

Heatmap is an attention approximation built from boxes (not true pixel-level Grad-CAM),
but it clearly shows where the detector focuses.
"""

import os, glob, cv2, numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.cm as cm

# -------------------------
# PATHS
# -------------------------
MODEL_PATH = "baseline.pt"  # <- VisDrone baseline model
ORIG_DIR   = "gradcam_output_visdrone/originals"
PRED_DIR   = "gradcam_output_visdrone/predictions"
HEAT_DIR   = "gradcam_output_visdrone/heatmaps"
PANEL_DIR  = "gradcam_output_visdrone/panels"

# ---------------------------------------------------------
# Only process selected VisDrone images
# ---------------------------------------------------------
USE_SELECTION = True

SELECTED_BASENAMES = [
    "0000010_05149_d_0000057.jpg",
    "0000030_00754_d_0000036.jpg",
    "0000042_02421_d_0000076.jpg",
    "0000045_01032_d_0000085.jpg",
    "0000046_00720_d_0000088.jpg",
    "0000056_00727_d_0000111.jpg",
    "0000068_04169_d_0000014.jpg",
    "0000071_03281_d_0000004.jpg",
    "0000071_05298_d_0000008.jpg",
    "0000071_06447_d_0000009.jpg",
]


os.makedirs(PRED_DIR,  exist_ok=True)
os.makedirs(HEAT_DIR,  exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

print("\nLoading YOLO model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# -------------------------
# HEATMAP HELPERS
# -------------------------
MAX_SIGMA   = 150.0
MIN_SIGMA   = 12.0
CONF_POWER  = 1.0

def make_heatmap_from_boxes(image_shape, boxes, scores):
    """Builds a smooth heatmap from detection boxes and confidences."""
    h, w = image_shape
    heat = np.zeros((h, w), dtype=np.float32)
    img_area = w * h

    for (x1, y1, x2, y2), s in zip(boxes, scores):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        box_area = bw * bh
        rel = np.clip(box_area / img_area, 1e-6, 1.0)

        inv_rel = 1.0 - rel
        sigma = MIN_SIGMA + (MAX_SIGMA - MIN_SIGMA) * (inv_rel ** 0.5)

        weight = float(s) ** CONF_POWER

        size = int(max(3, sigma * 6))
        if size % 2 == 0:
            size += 1

        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / (kernel.max() + 1e-12)

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
        heat = heat / heat.max()
    return heat

def overlay_heatmap(img_rgb, heatmap, alpha=0.5):
    color = (cm.get_cmap("jet")(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(img_rgb, 1 - alpha, color, alpha, 0)

# -------------------------
# PROCESS IMAGES
# -------------------------
if USE_SELECTION:
    img_paths = [os.path.join(ORIG_DIR, name) for name in SELECTED_BASENAMES]
else:
    img_paths = sorted(glob.glob(os.path.join(ORIG_DIR, "*.*")))

print(f"Found {len(img_paths)} VisDrone images.")

for path in tqdm(img_paths):
    fname = os.path.basename(path)

    orig_bgr = cv2.imread(path)
    if orig_bgr is None:
        print("Skipping unreadable:", fname)
        continue

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h, w = orig_rgb.shape[:2]

    # run baseline model with low conf so small objects are kept
    results = model.predict(path, conf=0.05, iou=0.5, verbose=False)
    r = results[0]

    boxes_xyxy, scores, cls_ids = [], [], []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy  = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy()

        for bb, c, cls in zip(xyxy, confs, clses):
            x1, y1, x2, y2 = map(int, bb)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(float(c))
            cls_ids.append(int(cls))

    # prediction image: draw boxes + labels from baseline.pt
    pred_img = orig_bgr.copy()
    names = model.names  # dict: class_id -> class_name

    for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, cls_ids):
        label_name = names.get(cid, "obj")
        label = f"{label_name} {sc:.2f}"
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(pred_img, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(PRED_DIR, fname), pred_img)

    # heatmap overlay
    heat = make_heatmap_from_boxes((h, w), boxes_xyxy, scores)
    heat_bgr = cv2.cvtColor(overlay_heatmap(orig_rgb, heat), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(HEAT_DIR, fname), heat_bgr)

    # panel ORIGINAL | PREDICTION | HEATMAP
    pred_resized = cv2.resize(pred_img, (w, h))
    panel = np.hstack((orig_bgr, pred_resized, heat_bgr))
    cv2.imwrite(os.path.join(PANEL_DIR, fname), panel)

print("\nDONE!")
print("Predictions saved in:", PRED_DIR)
print("Heatmaps saved in:   ", HEAT_DIR)
print("Panels saved in:     ", PANEL_DIR)
