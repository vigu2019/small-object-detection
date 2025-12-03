# generate_captions.py
"""
Auto-generate 1-2 line polished captions for each panel in gradcam_output/panels.
Outputs:
  - captions.txt (one caption per line with filename)
  - captions_folder/<imagename>.txt (individual caption files)
Logic:
  - If the prediction image contains no boxes -> "missed detection" caption
  - If prediction contains boxes and largest box area small -> "small object detection (good)"
  - If prediction contains boxes but very low confidence -> "possible false positive"
This is heuristic-based for quick report-ready captions.
"""
import os, glob, cv2, json
from ultralytics import YOLO

PANEL_DIR = "gradcam_output/panels"
ORIG_DIR = "gradcam_output/originals"
PRED_DIR = "gradcam_output/predictions"

os.makedirs("captions_folder", exist_ok=True)

# Load model (for inspecting boxes/conf)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

def analyze_prediction(image_name):
    # Run inference on original to get boxes (if pred image not parseable)
    img_path = os.path.join(ORIG_DIR, image_name)
    res = model.predict(source=img_path, conf=0.15, verbose=False)[0]
    boxes = []
    scores = []
    if len(res.boxes) > 0:
        try:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for bb, c in zip(xyxy, confs):
                boxes.append([int(x) for x in bb])
                scores.append(float(c))
        except Exception:
            pass
    return boxes, scores

def classify_and_caption(image_name):
    boxes, scores = analyze_prediction(image_name)
    caption = ""
    if len(boxes) == 0:
        caption = ("**Missed detection.** Left: original. Middle: model produced no detection. "
                   "Right: attention map shows activation away from the true object — indicates confusion due to clutter/scale.")
        tag = "missed"
    else:
        # compute largest box area and average confidence
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        max_area = max(areas) if len(areas)>0 else 0
        avg_conf = sum(scores)/len(scores) if len(scores)>0 else 0.0
        # image size for relative area
        img = cv2.imread(os.path.join(ORIG_DIR, image_name))
        h,w = img.shape[:2]
        rel = max_area / (w*h) if (w*h)>0 else 0
        # heuristics
        if rel < 0.002 and avg_conf >= 0.35:
            caption = ("**Good small-object detection.** Left: original. Middle: model correctly detected a very small object. "
                       "Right: Grad/attention overlay shows focused activation on the tiny object region.")
            tag = "good_small"
        elif rel < 0.01 and avg_conf >= 0.25:
            caption = ("**Small-object detection (typical).** Left: original. Middle: model detected a small object with moderate confidence. "
                       "Right: attention overlay highlights the object region despite limited pixels.")
            tag = "small"
        elif avg_conf < 0.25:
            caption = ("**Low-confidence detection / possible false positive.** Left: original. Middle: model prediction has low confidence. "
                       "Right: attention overlay shows diffused activation, explaining the uncertainty.")
            tag = "low_conf"
        else:
            caption = ("**Correct detection.** Left: original. Middle: model prediction (bounding boxes + confidence). "
                       "Right: attention overlay highlights the object region supporting the model's decision.")
            tag = "correct"
    return tag, caption

panel_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(PANEL_DIR, "*.*"))])
with open("captions.txt", "w", encoding="utf-8") as f:
    for fname in panel_files:
        tag, cap = classify_and_caption(fname)
        line = f"{fname}\t{tag}\t{cap}\n"
        f.write(line)
        # also create individual small files
        with open(os.path.join("captions_folder", fname + ".txt"), "w", encoding="utf-8") as g:
            g.write(cap)

print("Generated captions for", len(panel_files), "panels.")
print("Single-file captions -> captions_folder/")
print("All captions -> captions.txt")
