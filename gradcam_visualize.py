from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

# Load your trained model
model = YOLO("best.pt")  # replace with your actual model file

# Load test image
image_path = "test_image.jpg"  # replace with your test image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = np.float32(img_rgb) / 255.0
img_tensor = torch.from_numpy(img_input).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]

# Select last convolutional layer
target_layer = model.model.model[-2]

# Grad-CAM
cam = GradCAM(model=model.model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
grayscale_cam = cam(input_tensor=img_tensor)[0,:,:]
cam_image = show_cam_on_image(img_input, grayscale_cam, use_rgb=True)

# Display
plt.imshow(cam_image)
plt.axis("off")
plt.show()

# Save the result
cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
print("Grad-CAM visualization saved as gradcam_result.jpg")
