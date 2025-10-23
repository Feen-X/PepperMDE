from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Base directory = the folder containing this script
BASE_DIR = Path(__file__).resolve().parent

# Load pipeline from local model directory
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0,               # set to -1 for CPU, 0 for first CUDA GPU
    use_fast=True
)

image_dir = str(BASE_DIR / "Images")
image = Image.open(image_dir + "/" + "Room.jpg").convert("RGB")

result = pipe(image)  # Model inference

depth_tensor = result["predicted_depth"]  # torch.Tensor type
depth_array = depth_tensor.squeeze().cpu().numpy()  # HxW numpy array

# Normalize for visualization
depth_norm = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
depth_uint8 = (depth_norm * 255).astype("uint8")

# Make a color image
depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
depth_color = cv2.resize(depth_color, (image.width // 2, image.height // 2))
cv2.imshow("My Depth Map", depth_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
