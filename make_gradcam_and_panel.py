# make_gradcam_and_panel.py (fixed for different grad-cam versions)
import os, glob, cv2, numpy as np, torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn

# ----------------------
MODEL_PATH = "best.pt"
# ----------------------

ORIG_DIR = "gradcam_output/originals"
PRED_DIR = "gradcam_output/predictions"
GRADCAM_DIR = "gradcam_output/gradcams"
PANEL_DIR = "gradcam_output/panels"

os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

print("Loading model:", MODEL_PATH)
ul_model = YOLO(MODEL_PATH)
pt_model = ul_model.model
pt_model.eval()

# Find a Conv2d layer for Grad-CAM
def find_target_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            print("Using Grad-CAM target layer:", name)
            return module
    print("No Conv2d found; using last module as target layer.")
    return list(model.modules())[-1]

target_layer = find_target_layer(pt_model)

transform = transforms.Compose([transforms.ToTensor()])

image_paths = sorted(glob.glob(os.path.join(ORIG_DIR, "*.*")))
print("Found", len(image_paths), "images for Grad-CAM")

# device handling
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device for Grad-CAM:", device)

# Try a few ways to construct the GradCAM object to be robust across versions
cam = None
init_errors = []
for attempt in [
    {"kw": {"use_cuda": use_cuda}},
    {"kw": {"device": device}},
    {"kw": {}},
]:
    try:
        cam = GradCAM(model=pt_model, target_layers=[target_layer], **attempt["kw"])
        print("GradCAM initialized with kwargs:", attempt["kw"])
        break
    except TypeError as e:
        init_errors.append((attempt["kw"], str(e)))
        cam = None

if cam is None:
    print("Failed to initialize GradCAM with attempts:", init_errors)
    raise RuntimeError("Could not initialize GradCAM. See attempts above.")

# Main loop
for img_path in tqdm(image_paths):
    base = os.path.basename(img_path)

    results = ul_model.predict(source=img_path, conf=0.25, verbose=False)
    r = results[0]

    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        print("Failed to read:", img_path)
        continue
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h, w = orig_rgb.shape[:2]
    img_float = orig_rgb.astype(np.float32) / 255.0

    if len(r.boxes) == 0:
        print("No detections → skipping Grad-CAM:", base)
        blank = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(GRADCAM_DIR, base), blank)
        pred_img = cv2.imread(os.path.join(PRED_DIR, base)) if os.path.exists(os.path.join(PRED_DIR, base)) else blank
        panel = np.hstack((orig_bgr, pred_img, blank))
        cv2.imwrite(os.path.join(PANEL_DIR, base), panel)
        continue

    # pick top scoring detection
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    top_idx = int(np.argmax(scores))
    top_class = int(classes[top_idx])

    # prepare input tensor and move to device if required
    input_tensor = transform(orig_rgb).unsqueeze(0)  # 1,C,H,W
    if hasattr(cam, "device") and cam.device is not None:
        # some versions store device in cam; ensure tensor on same device
        input_tensor = input_tensor.to(cam.device)
    else:
        input_tensor = input_tensor.to(device)

    targets = [ClassifierOutputTarget(int(top_class))]

    # call cam; different versions accept numpy or torch tensor, but our input is torch
    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    except TypeError:
        # some versions expect tensor only, try without keywords
        grayscale_cam = cam(input_tensor, targets)[0]

    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    grad_out = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(GRADCAM_DIR, base), grad_out)

    # Prediction image (assumed created by batch_infer.py)
    pred_path = os.path.join(PRED_DIR, base)
    pred_img = cv2.imread(pred_path) if os.path.exists(pred_path) else np.zeros_like(orig_bgr)
    pred_img = cv2.resize(pred_img, (w, h))
    grad_out = cv2.resize(grad_out, (w, h))

    panel = np.hstack((orig_bgr, pred_img, grad_out))
    cv2.imwrite(os.path.join(PANEL_DIR, base), panel)

print("Grad-CAM panels saved in:", PANEL_DIR)
