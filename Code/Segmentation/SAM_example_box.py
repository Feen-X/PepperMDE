import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Import SAM2 stuff
from SAM.sam2.build_sam import build_sam2
from SAM.sam2.sam2_image_predictor import SAM2ImagePredictor
from SAM.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).resolve().parent
img_path = str(BASE_DIR / ".." / "Images" / "cabinet.jpg")
checkpoint = str(BASE_DIR / "Models" / "sam2.1_hiera_small.pt")
config = str(BASE_DIR / "Models" / "sam2.1_hiera_s.yaml")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load image
image = np.array(Image.open(img_path).convert("RGB"))

# -------------------------------
# Helper functions 
# -------------------------------
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        
# -------------------------------
# Segmentation
# -------------------------------
# Bounding box input
input_box = np.array([400, 60, 1100, 1400])
# input_box = np.array([700, 500, 900, 700])

# Build model
sam2 = build_sam2(config, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2)
predictor.set_image(image)

masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

print(f"Number of masks from box prompt: {len(masks)}")

# Visualize results
show_masks(image, masks, scores, box_coords=input_box)