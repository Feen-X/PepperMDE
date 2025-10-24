from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import threading
import queue
from pathlib import Path
from DepthEstimation.DE_functions import ROI_depth_info, estimate_depth
from ObjectDetection.OD_functions import draw_boxes, draw_line_to_most_confident


#------------------------------------#
# INITIALIZATION OF MODELS AND PATHS #
#------------------------------------#

# Base directory = the folder containing this script
BASE_DIR = Path(__file__).resolve().parent

# Paths relative to the script
YOLO_model_path = str(BASE_DIR / "ObjectDetection" / "models" / "yolo-world-chair.pt") # Pre-trained model path
imgDir = str(BASE_DIR / "Images" / "chair3.jpg") # Image path if camera is not available

# Load YOLO model
OD_model = YOLO(YOLO_model_path, verbose=True)
OD_model.fuse()  # Optional: fuse Conv + BN layers for slightly faster inference
device = 0  # 0 for CUDA GPU, -1 for CPU


# Load Depth Estimation Model
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",  # You can pick Small/Medium/Large
    device=device,  # set to -1 for CPU, 0 for first CUDA GPU
    use_fast=True,
    verbose=True
)

# ------------------------------------#
# THREADING AND QUEUES FOR PROCESSING #
# ------------------------------------#

# Shared variables
img = True # If true, forces use of image instead of camera
running = True

# Queues for thread communication
raw_frame_queue = queue.Queue(maxsize=2)      # Capture → Inference & Depth
OD_frame_queue = queue.Queue(maxsize=2) # Inference (dict) → Fusion
DE_frame_queue = queue.Queue(maxsize=2)     # Depth (dict) → Fusion  
final_frame_queue = queue.Queue(maxsize=2)     # Fusion → Display

# VIDEO CAPTURE THREAD #
def capture_thread():
    global running, img
    capture = cv2.VideoCapture(0)

    # IMAGE CAPTURE #
    if not capture.isOpened() or img is True:
        print("Error: Could not open camera. Showing image instead.")
        img_frame = cv2.imread(imgDir)
        divisor = img_frame.shape[0] // 320
        img_frame = cv2.resize(img_frame, (img_frame.shape[1] // (divisor if divisor!=0 else 1), img_frame.shape[0] // (divisor if divisor!=0 else 1)))
        if img_frame is None:
            print(f"Error: Could not load fallback image from {imgDir}.")
            return
        while running:
            try:
                raw_frame_queue.put_nowait(img_frame.copy())
            except queue.Full:
                pass  # Skip if queue is full, keep latest frame
            cv2.waitKey(500) # Sleep a bit
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

# DEPTH ESTIMATION THREAD #
def depth_thread():
    global running
    
    while running:
        try:
            # Get frame from capture thread (same raw frame as YOLO)
            frame = raw_frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Resize to lower resolution to reduce inference cost (maintain aspect)
        target_w = 640
        h, w = frame.shape[:2]
        if w > target_w:
            new_h = int(h * (target_w / w))
            frame_small = cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame

        # Estimate depth map (use smaller frame)
        metric_depth_estimation, depth_array = estimate_depth(frame=frame_small, pipe=pipe)

        # Prepare structured payload for depth (include original frame for size info)
        de_payload = {
            "frame": frame,
            "relative_depth": depth_array,
            "metric_depth": metric_depth_estimation
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

# OBJECT DETECTION THREAD #
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

        # Flatten detection boxes, classes, and confidences in the original order
        boxes = [box.xyxy[0].tolist() for box in results[0].boxes]
        classes = [int(box.cls[0]) for box in results[0].boxes]
        confs = [float(box.conf[0]) for box in results[0].boxes]

        # Draw boxes on the annotated frame using helper
        draw_boxes(annotated, boxes, classes, confs, names=results[0].names, 
                   box_color=(255,0,255), text_color=(255,0,255), thickness=2, text_scale=0.6)

        # Draw a line to the most confident detection (prefer 'chair' if available)
        best_idx, box_center = draw_line_to_most_confident(annotated, image_center, boxes, classes=classes, 
                                                           confs=confs, names=results[0].names, class_filter='chair', 
                                                           color=(255,0,0), thickness=2, dot_radius=5)

        # Prepare structured data with detections and annotated frame
        od_payload = {
            "frame": frame,               # original frame (BGR)
            "annotated": annotated,       # annotated frame (BGR)
            "boxes": [box.xyxy[0].tolist() for box in results[0].boxes],
            "classes": [int(box.cls[0]) for box in results[0].boxes],
            "confs": [float(box.conf[0]) for box in results[0].boxes],
            "conf_box_center": box_center,
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


# Fusion thread - YOLO+Depth #
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
        if od_payload is None or de_payload is None:
            continue
        
        yolo_annotated = od_payload.get("annotated")
        depth_color = de_payload.get("relative_depth")
        metric_depth = de_payload.get("metric_depth")

        # Ensure that we get metric depth
        if metric_depth is None:
            continue

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
        conf_box_center = od_payload.get("conf_box_center", None)
        
        # Use helper functions
        roi_median_depth, max_depth, min_depth, median_depth, mean_depth = ROI_depth_info(metric_depth, boxes[0] if boxes else None)
        
        draw_boxes(depth_with_boxes, boxes, classes=classes, confs=confs, names=names, box_color=(255,0,255), 
                   text_color=(255,0,255), thickness=2, text_scale=0.6)
        
        draw_line_to_most_confident(depth_with_boxes, (depth_with_boxes.shape[1]//2, depth_with_boxes.shape[0]//2), 
                                    boxes, classes=classes, confs=confs, names=names, class_filter='chair', 
                                    color=(255,0,0), thickness=2, dot_radius=5)
        
        # Pad depth_with_boxes to match top_row width (which is 2*w_y)
        overlay_padded = cv2.copyMakeBorder(depth_with_boxes, 0, 0, 0, top_row.shape[1] - depth_with_boxes.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # Stack top and bottom
        final_combined = cv2.vconcat([top_row, overlay_padded])

        # Add labels
        cv2.putText(final_combined, "YOLO Detection", (10, h_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_combined, "Depth Estimation", (w_y + 10, h_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_combined, "Combined", (10, h_y * 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(final_combined, f"Median depth: {median_depth:.1f}", (w_y + 50, h_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_combined, f"Mean depth: {mean_depth:.1f}", (w_y + 50, h_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_combined, f"Max Depth: {max_depth:.1f}", (w_y + 50, h_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_combined, f"Min Depth: {min_depth:.1f}", (w_y + 50, h_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if roi_median_depth is not None:
            cv2.putText(final_combined, f"ROI Median Depth: {roi_median_depth:.1f}", (w_y + 50, h_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if roi_median_depth / median_depth > 0.8:
                cv2.putText(final_combined, "Move!", (w_y + 50, h_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

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