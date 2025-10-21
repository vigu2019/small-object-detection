"""
gradcam_yolo_fallback.py

Purpose:
 - Run YOLOv8 inference on an image
 - Compute per-object Grad-CAM (ResNet50 fallback) for each detected box
 - Save:
     outputs/detections.png  -> detection boxes overlay
     outputs/gradcams/       -> individual Grad-CAM overlays per detection
"""

import os
import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp
from torchvision import models

# ---------- CONFIG ----------
IMAGE_PATH = r"C:\Users\hp\small-object-detection\image1.png"
WEIGHTS = r"C:\Users\hp\small-object-detection\runs\detect\train\weights\best.pt"
IMG_SIZE = 640
CONF_THR = 0.25
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "gradcams"), exist_ok=True)

def draw_detections(image_bgr, boxes, classes, scores, class_names=None):
    img = image_bgr.copy()
    for i, (box, cid, sc) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)
        label = f"{(class_names[cid] if class_names else str(cid))}:{sc:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img

print("Loading YOLOv8 model:", WEIGHTS)
model = YOLO(WEIGHTS)
model.to(DEVICE)

print("Running YOLO inference...")
results = model.predict(source=IMAGE_PATH, imgsz=IMG_SIZE, conf=CONF_THR, device=DEVICE, verbose=False)
if len(results) == 0:
    raise SystemExit("No results from YOLO predict.")

res = results[0]
boxes = res.boxes.xyxy.cpu().numpy()
scores = res.boxes.conf.cpu().numpy()
classes = res.boxes.cls.cpu().numpy().astype(int)
img_bgr = cv2.imread(IMAGE_PATH)
class_names = model.names if hasattr(model, "names") else None

# Save detection overlay
vis_det = draw_detections(img_bgr, boxes, classes, scores, class_names)
cv2.imwrite(os.path.join(OUTPUT_DIR, "detections.png"), vis_det)
print("✅ Saved detection overlay.")

# ------------------- Per-object Grad-CAM -------------------
print("Running per-object Grad-CAM (ResNet fallback)...")

resnet = models.resnet50(pretrained=True).to(DEVICE).eval()
cam_extractor = SmoothGradCAMpp(resnet)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        continue  # skip invalid boxes

    input_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)
    out = resnet(input_tensor)
    class_idx = out.squeeze(0).argmax().item()

    activation_map = cam_extractor(class_idx, out)

    # -------- FIXED heatmap code --------
    heatmap = activation_map[0].cpu().numpy()     # to numpy
    heatmap = np.squeeze(heatmap)                 # remove extra dimensions
    heatmap = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))  # resize to box
    heatmap = heatmap - np.min(heatmap)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)      # normalize
    heatmap = np.uint8(255 * heatmap)            # convert to uint8
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # -----------------------------------

    overlay = cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)
    out_path = os.path.join(OUTPUT_DIR, "gradcams", f"gradcam_{i}.png")
    cv2.imwrite(out_path, overlay)
    print(f"✅ Saved Grad-CAM for detection {i} -> {out_path}")

print("✅ Done! Check outputs/gradcams for per-object Grad-CAMs.")
