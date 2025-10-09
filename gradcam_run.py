from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Load image and resize
image_path = "test_image.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))  # resize to multiple of 32
img_input = np.float32(img_resized) / 255.0
img_tensor = torch.from_numpy(img_input).permute(2,0,1).unsqueeze(0)

# Select last conv layer
target_layer = model.model.model[-2]

# Grad-CAM
cam = GradCAM(model=model.model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
grayscale_cam = cam(input_tensor=img_tensor)[0,:,:]
cam_image = show_cam_on_image(img_input, grayscale_cam, use_rgb=True)

# Show and save
plt.imshow(cam_image)
plt.axis("off")
plt.show()
cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
print("Grad-CAM saved as gradcam_result.jpg")
