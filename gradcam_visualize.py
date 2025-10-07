# gradcam_visualize_fixed.py
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from grad_cam import GradCAM  # updated import for latest package
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- CONFIG ---
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
IMAGE_PATH = "data/images/test/test_image.jpg"  # <-- update this to an existing image
OUT_PATH = "gradcam_output.jpg"
RESIZE = 640
# --------------

# YOLO wrapper for Grad-CAM
class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super(YOLOWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # YOLOv8 returns a list of Results
        preds = self.model(x)  # preds is list of Results
        # Grad-CAM expects a scalar; take max of confidence scores
        scores = []
        for p in preds:
            if hasattr(p, 'boxes') and p.boxes.shape[0] > 0:
                scores.append(p.boxes.cls.float().max())
        if scores:
            return torch.stack(scores).sum()  # scalar to enable backward
        else:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

def find_last_conv(module):
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1] if convs else None

def main():
    print("Loading model:", MODEL_PATH)
    yolo_model = YOLO(MODEL_PATH)
    pt_model = YOLOWrapper(yolo_model.model)
    pt_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_model.to(device)

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Load & preprocess image
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    img_resized = cv2.resize(img_rgb, (RESIZE, RESIZE))
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    input_tensor.requires_grad_(True)

    # Target layer
    target_layer = find_last_conv(pt_model)
    print("Using target layer:", target_layer)

    # Grad-CAM (latest grad-cam package no longer needs use_cuda)
    cam = GradCAM(model=pt_model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True, aug_smooth=True)

    # Overlay CAM on image
    cam_image = show_cam_on_image(img_resized, grayscale_cam[0], use_rgb=True)
    out_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUT_PATH, out_bgr)
    print("✅ Grad-CAM image saved to", OUT_PATH)

if __name__ == "__main__":
    main()
