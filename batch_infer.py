# batch_infer.py
from ultralytics import YOLO
import glob, os, cv2

# ----------------------
# CHANGE MODEL HERE
MODEL_PATH = "best.pt"   # your trained model
# ----------------------

INPUT_DIR = "gradcam_output/originals"
OUT_DIR = "gradcam_output/predictions"
CONF_THRESH = 0.25

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

img_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.*")))
print(f"Found {len(img_paths)} images to run inference on.")

for p in img_paths:
    print("Processing:", p)
    results = model.predict(source=p, conf=CONF_THRESH, save=False, verbose=False)
    r = results[0]

    vis = r.plot()  # RGB image

    base = os.path.basename(p)
    outp = os.path.join(OUT_DIR, base)

    cv2.imwrite(outp, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("Saved prediction:", outp)

print("Inference done.")
