from ultralytics_backup import YOLO
import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import os

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO(args.model)
yolo_model = model.model

# ----------------------------
# Read image
# ----------------------------
img = cv2.imread(args.img)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = np.float32(img_rgb) / 255.0

# 🔴 THIS WAS MISSING — VERY IMPORTANT
input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

# ----------------------------
# YOLO wrapper (CRITICAL FIX)
# ----------------------------
class YOLOWrapper(torch.nn.Module):
    def __init__(self, yolo):
        super().__init__()
        self.yolo = yolo

    def forward(self, x):
        outputs = self.yolo(x)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        return outputs.sum()

wrapped_model = YOLOWrapper(yolo_model)

# ----------------------------
# Target layer
# ----------------------------
target_layer = yolo_model.model[-2]

# ----------------------------
# Grad-CAM
# ----------------------------
cam = GradCAM(
    model=wrapped_model,
    target_layers=[target_layer]
)

grayscale_cam = cam(input_tensor=input_tensor)[0]

heatmap = show_cam_on_image(
    img_float,
    grayscale_cam,
    use_rgb=True
)

# ----------------------------
# Save output
# ----------------------------
os.makedirs(os.path.dirname(args.out), exist_ok=True)
cv2.imwrite(args.out, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

print(f"Grad-CAM saved at {args.out}")
