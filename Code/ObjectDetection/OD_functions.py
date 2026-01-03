# OD_functions.py
'''
This module is focused around YOLO helper functions. It contains functions for drawing 
bounding boxes and labels onto images, as well as resizing images for YOLO input.
'''

import cv2

def draw_boxes(target_frame, boxes, classes=None, confs=None, names=None,
               box_color=(255, 0, 255), text_color=(255, 0, 255), thickness=2, text_scale=0.6):
    """
    Draw bounding boxes and labels onto target_frame.

    - boxes: iterable of [x1,y1,x2,y2] in source coordinates (source size = src_size)
    - classes: list of class indices for each box (optional)
    - confs: list of confidences for each box (optional)
    - names: mapping (dict or list-like) from class index to class name (optional)
    - src_size: (width, height) of the source coordinate system for boxes. If None,
      boxes are assumed to already be in target_frame coordinates.
    - box_color: color for the bounding box (BGR tuple)
    - text_color: color for the text (BGR tuple)
    - thickness: thickness of box lines
    - text_scale: scale factor for text size
    """
    if boxes is None or len(boxes) == 0:
        return

    # Assume boxes are already in the same coordinate system as target_frame
    for i, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = [int(coord) for coord in box]
        except Exception:
            continue

        cv2.rectangle(target_frame, (x1, y1), (x2, y2), box_color, thickness+2)

        label = None
        cls_idx = None
        if classes is not None and i < len(classes):
            cls_idx = classes[i]

        if names is not None:
            if isinstance(names, dict):
                label_name = names.get(cls_idx, '')
            else:
                try:
                    label_name = names[cls_idx]
                except Exception:
                    label_name = ''
        else:
            label_name = ''

        conf_text = ''
        if confs is not None and i < len(confs):
            try:
                conf_text = f" {confs[i]:.2f}"
            except Exception:
                conf_text = ''

        if label_name or conf_text:
            label = f"{label_name}{conf_text}".strip()

        if label:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, max(1, thickness))[0]
            text_width, text_height = text_size
            if y1 - text_height < 0:
                text_y = y1 + text_height
            else:
                text_y = max(y1 - 3, 0)
            cv2.putText(target_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, max(1, thickness))


def draw_line_to_most_confident(target_frame, best_box, color=(255,0,0), thickness=2, dot_radius=5):
    """
    Draws a line from the center of target frame to the center of the specified bounding box onto target_frame. Returns the center coordinates of the box.
    """
    if best_box is None:
        return None, None
    x1, y1, x2, y2 = [int(coord) for coord in best_box]
    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    try:
        cv2.line(target_frame, 
                 (target_frame.shape[1]//2, target_frame.shape[0]//2), 
                 box_center, 
                 color, 
                 thickness)
        cv2.circle(target_frame, box_center, dot_radius, color, -1)
    except Exception:
        pass

    return box_center

def resize_for_yolo(frame, w=640):
    '''Resize frame to target width while maintaining aspect ratio for YOLO input.'''
    h, ww = frame.shape[:2]
    if ww == w:
        return frame
    nh = int(h * (w/ww))
    return cv2.resize(frame, (w, nh), interpolation=cv2.INTER_AREA)
