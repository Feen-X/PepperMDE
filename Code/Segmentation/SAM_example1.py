import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Import SAM2 build functions (match actual filenames)
from SAM.sam2.build_sam import build_sam2
from SAM.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Path settings
# make base directory be the sam2 folder which is in the directory of this file
BASE_DIR = Path(__file__).resolve().parent
img_path = str(BASE_DIR / ".." / "Images" / "cabinet2.jpg")
checkpoint = str(BASE_DIR / "Models" / "sam2.1_hiera_small.pt")
config = str(BASE_DIR / "Models" / "sam2.1_hiera_s.yaml")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model
sam2 = build_sam2(config, checkpoint, device=device, apply_postprocessing=True)

# Create mask generator
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# Load image
image = Image.open(img_path).convert("RGB")
image_np = np.array(image)

# Generate masks (segment everything)
masks = mask_generator.generate(image_np)

# A helper to overlay masks
def show_anns(anns, img_np, border=True):
    if len(anns) == 0:
        return img_np
    img = img_np.copy()
    for ann in anns:
        m = ann["segmentation"]
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        img[m] = img[m] * 0.5 + np.array(color) * 0.5
        if border:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, color, 2)
    return img.astype(np.uint8)

# Visualize result
overlay = show_anns(masks, image_np)
plt.figure(figsize=(10,10))
plt.imshow(overlay)
plt.axis("off")
plt.show()

# Optionally save mask overlay
# save_path = r"C:\images\cabinet_segmented.png"
# Image.fromarray(overlay).save(save_path)
# print("Saved overlay:", save_path)
