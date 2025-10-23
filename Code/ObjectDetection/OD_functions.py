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

        cv2.rectangle(target_frame, (x1, y1), (x2, y2), box_color, thickness)

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
            cv2.putText(target_frame, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, max(1, thickness))


def draw_line_to_most_confident(target_frame, image_center, boxes, classes=None, confs=None, names=None,
                                class_filter=None, color=(255,0,0), thickness=2, dot_radius=5):
    """
    Select the most confident detection (optionally filtered by class name) and draw a line
    from image_center to the center of that box, plus a filled dot at the box center.

    - boxes: list of [x1,y1,x2,y2]
    - classes: list of class indices
    - confs: list of confidences (floats)
    - names: mapping index->class name (dict or list)
    - class_filter: optional class name to restrict selection (e.g., 'chair')
    Returns the selected index and center or (None, None) if nothing selected.
    """
    if not boxes:
        return None, None

    # Build candidate indices
    indices = list(range(len(boxes)))

    # If class_filter provided, keep only indices matching that class name
    if class_filter is not None and names is not None and classes is not None:
        def cls_name_for(i):
            try:
                cls_idx = classes[i]
                if isinstance(names, dict):
                    return names.get(cls_idx, '')
                else:
                    return names[cls_idx] if cls_idx is not None and cls_idx < len(names) else ''
            except Exception:
                return ''

        indices = [i for i in indices if cls_name_for(i) == class_filter]

    if not indices:
        return None, None

    # Choose by highest confidence if confs available, otherwise pick first
    best_idx = None
    if confs is not None and len(confs) >= 1:
        best_idx = max(indices, key=lambda i: confs[i] if i < len(confs) and confs[i] is not None else -1.0)
    else:
        best_idx = indices[0]

    try:
        x1, y1, x2, y2 = [int(c) for c in boxes[best_idx]]
    except Exception:
        return None, None

    box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    try:
        cv2.line(target_frame, image_center, box_center, color, thickness)
        cv2.circle(target_frame, box_center, dot_radius, color, -1)
    except Exception:
        pass

    return best_idx, box_center

