from PIL import Image
import numpy as np

def estimate_depth(frame, pipe):
    """
    Depth Estimation, using the transformers pipeline from Hugging Face
    """
    frame = Image.fromarray(frame) # convert to PIL Image so that the pipeline works

    result = pipe(frame)
    depth = np.array(result["depth"], dtype=np.float32)

    # Normalize depth map to 0–255 for visualization
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype('uint8')

    # Apply colormap for visualization
    # depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    return depth_uint8

def ROI_depth_info(depth_map, bbox=None):
    """
    Calculate the median depth within the bounding box region of interest (ROI)
    """
    median_depth = np.median(depth_map)
    mean_depth = np.mean(depth_map)

    if bbox is None:
        return None, None, None, median_depth, mean_depth

    x1, y1, x2, y2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
    roi = depth_map[y1:y2, x1:x2]
    roi_mean_depth = np.mean(roi)
    roi_max_depth = np.max(roi)
    roi_min_depth = np.min(roi)

    return roi_mean_depth, roi_max_depth, roi_min_depth, median_depth, mean_depth


def distance_to_object(depth_map, bbox, real_object_height_m,
                       focal_length_px=None,
                       focal_length_mm=None, sensor_height_mm=None, image_height_px=None):
    """
    Estimate distance (meters) to an object when depth_map is normalized (0-255)
    using the pinhole camera model: Z = (H_real * f_pixels) / h_pixels

    Parameters:
    - depth_map: 2D array (normalized 0-255). Only used for bounds checking / image size.
    - bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
    - real_object_height_m: known real object height in meters (required).
    - focal_length_px: focal length in pixels (preferred).
    - OR provide focal_length_mm, sensor_height_mm, image_height_px to compute focal_length_px.

    Returns:
    - distance_m (float)
    """
    if real_object_height_m is None:
        raise ValueError("Provide real_object_height_m (real object height in meters)")

    x1, y1 = int(np.floor(bbox[0])), int(np.floor(bbox[1]))
    x2, y2 = int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))

    # clamp to image bounds
    h, w = depth_map.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid bbox or bbox outside image bounds")

    object_height_px = float(y2 - y1)
    if object_height_px <= 0:
        raise ValueError("Computed object height in pixels is zero")

    if focal_length_px is None:
        if focal_length_mm is not None and sensor_height_mm is not None and image_height_px is not None:
            focal_length_px = (focal_length_mm * image_height_px) / sensor_height_mm
        else:
            raise ValueError("Provide focal_length_px or (focal_length_mm, sensor_height_mm, image_height_px)")

    distance_m = (real_object_height_m * float(focal_length_px)) / object_height_px
    return float(distance_m)