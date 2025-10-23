from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import threading
import queue
from pathlib import Path


#------------------------------------#
# INITIALIZATION OF MODELS AND PATHS #
#------------------------------------#

# Base directory = the folder containing this script
BASE_DIR = Path(__file__).resolve().parent

# Paths relative to the script
model_path = str(BASE_DIR / "models" / "yolo-world-chair.pt") # Pre-trained model path
imgDir = str(BASE_DIR / "Images" / "Room.jpg") # Image path if camera is not available

# Load YOLO model
OD_model = YOLO(model_path, verbose=False)
OD_model.fuse()  # Optional: fuse Conv + BN layers for slightly faster inference
device = 0  # 0 for CUDA GPU, -1 for CPU


# Load Depth Estimation Model
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",  # You can pick Small/Medium/Large
    device=device,  # set to -1 for CPU, 0 for first CUDA GPU
    use_fast=True,
    verbose=False
)

# HELPER FUNCTIONS #

def estimate_depth(frame, pipe=pipe):
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


# ------------------------------------#
# THREADING AND QUEUES FOR PROCESSING #
# ------------------------------------#

# Shared variables
img = True # If true, forces use of image instead of camera
running = True

# Queues for thread communication
raw_frame_queue = queue.Queue(maxsize=2)      # Capture → Inference & Depth
# Threads now send structured payloads (dicts) to fusion
OD_frame_queue = queue.Queue(maxsize=2) # Inference (dict) → Fusion
DE_frame_queue = queue.Queue(maxsize=2)     # Depth (dict) → Fusion  
final_frame_queue = queue.Queue(maxsize=2)     # Fusion → Display

# Video capture thread #
def capture_thread():
    global running, img
    capture = cv2.VideoCapture(0)

    # IMAGE CAPTURE #
    if not capture.isOpened() or img is True:
        print("Error: Could not open camera. Showing image instead.")
        img_frame = cv2.imread(imgDir)
        img_frame = cv2.resize(img_frame, (img_frame.shape[1] // 3, img_frame.shape[0] // 3))
        if img_frame is None:
            print(f"Error: Could not load fallback image from {imgDir}.")
            return
        while running:
            try:
                raw_frame_queue.put_nowait(img_frame.copy())
            except queue.Full:
                pass  # Skip if queue is full, keep latest frame
            cv2.waitKey(100) # Sleep a bit
        return

    # CAMERA REAL-TIME CAPTURE #
    while running:
        ret, f = capture.read()
        if not ret:
            break
        
        try:
            raw_frame_queue.put_nowait(f)
        except queue.Full:
            try:
                # Remove old frame and add new one to keep latest
                raw_frame_queue.get_nowait()
                raw_frame_queue.put_nowait(f)
            except queue.Empty:
                pass
    capture.release()

# Depth Estimation thread #
def depth_thread():
    global running
    
    while running:
        try:
            # Get frame from capture thread (same raw frame as YOLO)
            frame = raw_frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # frame_copy = frame.copy()
        # frame_copy = cv2.resize(frame_copy, (640, 480))
        
        # Estimate depth map
        depth_array = estimate_depth(frame)

        # Prepare structured payload for depth (include original frame for size info)
        de_payload = {
            "frame": frame,
            "depth": depth_array,
        }

        # Send depth payload to fusion thread
        try:
            DE_frame_queue.put_nowait(de_payload)
        except queue.Full:
            # Remove old payload and add new one to keep latest
            try:
                DE_frame_queue.get_nowait()
                DE_frame_queue.put_nowait(de_payload)
            except queue.Empty:
                pass

# YOLO Object Detection thread #
def YOLO_thread():
    global running
    
    while running:
        try:
            # Get frame from capture thread (blocks with timeout)
            frame = raw_frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        image_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        # Run YOLO (no need to resize manually, YOLO handles it)
        results = OD_model(frame, device=device, conf=0.7, verbose=False)  

        # Start with the original frame for annotation
        annotated = frame.copy()

        # List to hold center points of detected objects
        object_centers = []

        # Flatten detection boxes, classes, and confidences in the original order
        boxes = [box.xyxy[0].tolist() for box in results[0].boxes]
        classes = [int(box.cls[0]) for box in results[0].boxes]
        confs = [float(box.conf[0]) for box in results[0].boxes]

        # Draw boxes on the annotated frame using helper
        draw_boxes(annotated, boxes, classes, confs, names=results[0].names, 
                   box_color=(255,0,255), text_color=(255,0,255), thickness=2, text_scale=0.6)

        # Draw a line to the most confident detection (prefer 'chair' if available)
        best_idx, box_center = draw_line_to_most_confident(annotated, image_center, boxes, classes=classes, confs=confs, names=results[0].names, class_filter='chair', color=(255,0,0), thickness=2, dot_radius=5)
        if best_idx is not None and box_center is not None:
            object_centers.append(box_center)

        # Prepare structured data with detections and annotated frame
        od_payload = {
            "frame": frame,               # original frame (BGR)
            "annotated": annotated,       # annotated frame (BGR)
            "object_centers": object_centers,
            "boxes": [box.xyxy[0].tolist() for box in results[0].boxes],
            "classes": [int(box.cls[0]) for box in results[0].boxes],
            "confs": [float(box.conf[0]) for box in results[0].boxes],
            "names": results[0].names,
        }

        # Send structured detection data to fusion thread
        try:
            OD_frame_queue.put_nowait(od_payload)
        except queue.Full:
            # Remove old payload and add new one to keep latest
            try:
                OD_frame_queue.get_nowait()
                OD_frame_queue.put_nowait(od_payload)
            except queue.Empty:
                pass


# Fusion thread - combines YOLO and depth results
def fusion_thread():
    global running

    od_payload = None
    de_payload = None

    while running:
        # Try to get latest structured payloads
        try:
            od_payload = OD_frame_queue.get(timeout=0.05)
        except queue.Empty:
            pass

        try:
            de_payload = DE_frame_queue.get(timeout=0.05)
        except queue.Empty:
            pass

        # Need both payloads to proceed
        if od_payload is not None and de_payload is not None:
            orig_frame = od_payload.get("frame")
            yolo_annotated = od_payload.get("annotated")
            depth_map = de_payload.get("depth")

            # Ensure depth_map is single channel uint8; convert to 3-channel for visualization
            if depth_map is None:
                continue

            if len(depth_map.shape) == 2:
                depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            else:
                depth_color = depth_map

            # Resize depth_color to match yolo_annotated height
            h_y, w_y = yolo_annotated.shape[:2]
            depth_frame = cv2.resize(depth_color, (w_y, h_y))

            # Top: side-by-side YOLO annotated and depth
            top_row = cv2.hconcat([yolo_annotated, depth_frame])

            # Bottom: start with the colorized depth frame and draw YOLO boxes on it
            depth_with_boxes = depth_frame.copy()

            # Use helper to draw boxes onto depth visualization. Provide src_size so helper scales coords.
            boxes = od_payload.get("boxes", [])
            classes = od_payload.get("classes", [])
            confs = od_payload.get("confs", [])
            names = od_payload.get("names", {})
            orig_frame = od_payload.get("frame")
            src_size = None
            if orig_frame is not None:
                src_size = (orig_frame.shape[1], orig_frame.shape[0])

            draw_boxes(depth_with_boxes, boxes, classes=classes, confs=confs, names=names, box_color=(255,0,255), text_color=(255,0,255), thickness=2, text_scale=0.6)

            # Also draw the center-line on the depth visualization to match annotated frame
            draw_line_to_most_confident(depth_with_boxes, (depth_with_boxes.shape[1]//2, depth_with_boxes.shape[0]//2), boxes, classes=classes, confs=confs, names=names, class_filter='chair', color=(255,0,0), thickness=2, dot_radius=5)
            # Pad depth_with_boxes to match top_row width (which is 2*w_y)
            overlay_padded = cv2.copyMakeBorder(depth_with_boxes, 0, 0, 0, top_row.shape[1] - depth_with_boxes.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])

            # Stack top and bottom
            final_combined = cv2.vconcat([top_row, overlay_padded])

            # Add labels
            cv2.putText(final_combined, "YOLO Detection", (10, h_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(final_combined, "Depth Estimation", (w_y + 10, h_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(final_combined, "Combined", (10, h_y * 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Send combined result to display
            try:
                final_frame_queue.put_nowait(final_combined)
            except queue.Full:
                try:
                    final_frame_queue.get_nowait()
                    final_frame_queue.put_nowait(final_combined)
                except queue.Empty:
                    pass


# ----------------- #
# MAIN DISPLAY LOOP #
# ----------------- #

if __name__ == "__main__":
    # Start threads
    t1 = threading.Thread(target=capture_thread)
    t2 = threading.Thread(target=YOLO_thread)
    t3 = threading.Thread(target=depth_thread)
    t4 = threading.Thread(target=fusion_thread)
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    # Display loop with FPS counter
    display_frame_count = 0
    display_start_time = cv2.getTickCount()
    tick_frequency = cv2.getTickFrequency()
    display_fps_display = 0.0
    
    while running:
        try:
            # Get combined frame from fusion thread (blocks with timeout)
            combined_frame = final_frame_queue.get(timeout=0.1)
            
            # Update FPS counter (every successful get means a new processed frame)
            display_frame_count += 1
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - display_start_time) / tick_frequency
            
            # Update FPS display every 0.5 seconds
            if elapsed_time >= 0.5:
                display_fps_display = display_frame_count / elapsed_time
                display_frame_count = 0
                display_start_time = current_time
            
            # Display FPS on the frame (this shows true processing FPS)
            cv2.putText(combined_frame, f"FPS: {display_fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            # Show the combined frame
            cv2.imshow("YOLO + Depth Estimation", combined_frame)
            
        except queue.Empty:
            pass # No new frame available, continue loop

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    # Wait for threads to finish
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    cv2.destroyAllWindows()
