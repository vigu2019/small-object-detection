# make_detection_heatmap_and_panel.py
"""
Approximate attention heatmaps based on detection boxes.
Produces: gradcam_output/heatmaps/<img> and gradcam_output/panels/<img>
Panel layout: Original | Prediction | Heatmap overlay
This is NOT gradient-based Grad-CAM. It's an attention approximation
derived from predicted boxes (centered Gaussians weighted by confidence).
"""
import os, glob, cv2, numpy as np
from ultralytics import YOLO
from tqdm import tqdm

MODEL_PATH = "best.pt"   # your trained model
ORIG_DIR = "gradcam_output/originals"
PRED_DIR = "gradcam_output/predictions"
HEAT_DIR = "gradcam_output/heatmaps"
PANEL_DIR = "gradcam_output/panels"

os.makedirs(HEAT_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

# parameters for gaussian blob
MAX_SIGMA = 150.0   # largest blur radius for small confidence & big box
MIN_SIGMA = 12.0    # smallest blur for very small boxes
CONF_POWER = 1.0    # exponent applied to confidences (tweak if needed)

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

def make_heatmap_from_boxes(image_shape, boxes, scores):
    """
    image_shape: (h,w)
    boxes: Nx4 numpy array in xyxy format
    scores: N array of confidences (0..1)
    returns: heatmap float32 0..1
    """
    h, w = image_shape
    heat = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2), s in zip(boxes, scores):
        # center
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        # object size metric (smaller objects -> smaller sigma)
        box_area = bw * bh
        # sigma inversely proportional to object size (so small object -> small sigma)
        # normalize by image area
        img_area = w * h
        rel = np.clip(box_area / img_area, 1e-6, 1.0)
        # sigma: interpolate between MIN_SIGMA and MAX_SIGMA, smaller rel -> smaller sigma
        # we invert rel
        inv = 1.0 - rel
        sigma = MIN_SIGMA + (MAX_SIGMA - MIN_SIGMA) * (inv ** 0.5)
        # weight by confidence (optionally raise to power)
        weight = float(s) ** CONF_POWER
        # create gaussian patch
        size = int(max(3, sigma * 6))
        # ensure odd
        if size % 2 == 0:
            size += 1
        # create gaussian kernel
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * (sigma**2)))
        kernel = kernel / (kernel.max() + 1e-12)
        # position kernel center at (cx, cy)
        x0 = cx - size//2
        y0 = cy - size//2
        x1_ = max(0, x0)
        y1_ = max(0, y0)
        x2_ = min(w, x0 + size)
        y2_ = min(h, y0 + size)
        kx1 = x1_ - x0
        ky1 = y1_ - y0
        kx2 = kx1 + (x2_ - x1_)
        ky2 = ky1 + (y2_ - y1_)
        try:
            heat[y1_:y2_, x1_:x2_] += weight * kernel[int(ky1):int(ky2), int(kx1):int(kx2)]
        except Exception as e:
            # fallback: skip if region invalid
            continue

    # normalize
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat

def overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.5):
    # img_rgb: H,W,3 uint8 RGB; heatmap: H,W float32 0..1
    import matplotlib.cm as cm
    cmap = cm.get_cmap('jet')
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, colored, alpha, 0)
    return blended

# iterate
img_paths = sorted(glob.glob(os.path.join(ORIG_DIR, "*.*")))
print("Found", len(img_paths), "original images to process.")

for p in tqdm(img_paths):
    base = os.path.basename(p)
    orig_bgr = cv2.imread(p)
    if orig_bgr is None:
        print("Failed to read", p); continue
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h, w = orig_rgb.shape[:2]

    # run model to get boxes/scores (we also have predictions folder; either is fine)
    results = model.predict(source=p, conf=0.05, verbose=False)  # low conf to capture many boxes
    r = results[0]
    boxes_xyxy = []
    scores = []
    if len(r.boxes) > 0:
        # r.boxes.xyxy is tensor
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
        except:
            # fallback: if attributes differ, skip
            xyxy = []
            confs = []
        for bb, c in zip(xyxy, confs):
            x1, y1, x2, y2 = bb
            # clip
            x1, y1, x2, y2 = max(0,int(x1)), max(0,int(y1)), min(w-1,int(x2)), min(h-1,int(y2))
            boxes_xyxy.append([x1,y1,x2,y2]); scores.append(float(c))
    else:
        # no detections: make a tiny uniform heatmap (very low)
        boxes_xyxy = []
        scores = []

    heat = make_heatmap_from_boxes((h,w), boxes_xyxy, scores)
    heat_vis = overlay_heatmap_on_image(orig_rgb, heat, alpha=0.5)
    heat_bgr = cv2.cvtColor(heat_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(HEAT_DIR, base), heat_bgr)

    # Prediction image fallback: use existing prediction image if available else draw boxes on original
    pred_path = os.path.join(PRED_DIR, base)
    if os.path.exists(pred_path):
        pred_img = cv2.imread(pred_path)
    else:
        # draw boxes on a copy
        pred_img = orig_bgr.copy()
        for (x1,y1,x2,y2), sc in zip(boxes_xyxy, scores):
            cv2.rectangle(pred_img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(pred_img, f"{sc:.2f}", (x1, max(12,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

    # create panel Original | Prediction | Heatmap
    pred_resized = cv2.resize(pred_img, (w,h))
    panel = np.hstack((orig_bgr, pred_resized, heat_bgr))
    cv2.imwrite(os.path.join(PANEL_DIR, base), panel)

print("Done. Heatmaps:", HEAT_DIR, "Panels:", PANEL_DIR)
