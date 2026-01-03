# DE_functions.py
'''
This module contains functions for depth estimation using Hugging Face transformers pipeline.
It includes functions to estimate depth from an image and to extract depth information from
a region of interest (ROI) defined by bounding boxes.
'''

from PIL import Image
import cv2
import numpy as np

def estimate_depth(frame, pipe) -> tuple[np.ndarray, np.ndarray]:
    """
    Depth Estimation, using the transformers pipeline from Hugging Face.
    Parameters:
    - frame: input image as a numpy array (HxWx3)
    - pipe: transformers pipeline for depth estimation
    Returns:
    - depth: the estimated metric depth as a 2D numpy array (HxW)
    - depth_uint8_inv: normalized inverted depth map (0-255) for visualization
    """
    frame = Image.fromarray(frame) # convert to PIL Image so that the pipeline works

    result = pipe(frame)
    
    # Get predicted metric depth map as numpy array
    depth = result["predicted_depth"].squeeze().cpu().numpy()  # HxW numpy array

    # Normalize for visualization
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype('uint8')
    depth_uint8_inv = 255 - depth_uint8
    depth_uint8_inv = cv2.applyColorMap(depth_uint8_inv, cv2.COLORMAP_JET)

    return depth, depth_uint8_inv

def ROI_depth_info(depth_map, bbox=None):
    """
    Calculate the median depth within the bounding box region of interest (ROI)
    Returns:
    - roi_median_depth: median depth within the bbox ROI (or None if no bbox)
    - max_depth: maximum depth in the entire depth map
    - min_depth: minimum depth in the entire depth map
    - median_depth: median depth in the entire depth map
    - mean_depth: mean depth in the entire depth map
    """
    median_depth = np.median(depth_map)
    mean_depth = np.mean(depth_map)
    max_depth = np.max(depth_map)
    min_depth = np.min(depth_map)

    if bbox is None:
        return None, max_depth, min_depth, median_depth, mean_depth

    x1, y1, x2, y2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
    roi = depth_map[y1:y2, x1:x2]
    roi_median_depth = np.median(roi)

    return roi_median_depth, max_depth, min_depth, median_depth, mean_depth
