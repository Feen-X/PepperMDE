from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import time
import logging
import threading
import queue
from pathlib import Path
from DepthEstimation.DE_functions import ROI_depth_info, estimate_depth
from ObjectDetection.OD_functions import draw_boxes, draw_line_to_most_confident


# -------------------- CONFIG -------------------- #
BASE_DIR = Path(__file__).resolve().parent
YOLO_model_path = str(BASE_DIR / "ObjectDetection" / "models" / "yolo-world-s-cabinet.pt")
imgDir = str(BASE_DIR / "Images" / "cabinet1.jpg")
device = 0  # GPU = 0, CPU = -1

# Logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ----------------- GLOBAL STATE ----------------- #
img = True # If true, forces use of image instead of camera
running = True
system_mode = "search"  # "search", "approach", "interact"
movement_phase_active = False # True when movement phase has begun
movement_target = None        # last movement target payload (dict)

# Queues
raw_frame_queue = queue.Queue(maxsize=2)
OD_frame_queue = queue.Queue(maxsize=2)
DE_frame_queue = queue.Queue(maxsize=2)
final_frame_queue = queue.Queue(maxsize=2)
movement_target_queue = queue.Queue(maxsize=2) # movement controller receives targets here
movement_done_event = threading.Event()        # set to end current movement phase


# ------------------- MODEL LOAD ------------------- #
OD_model = YOLO(YOLO_model_path)
OD_model.fuse()

pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    device=device,
    use_fast=True,
    verbose=True
)


# ----------------- MODE HELPERS ----------------- #
def set_mode(new_mode: str):
    """Safely switch modes."""
    global system_mode
    if new_mode != system_mode:
        logging.info(f"Mode changed: {system_mode} → {new_mode}")
        system_mode = new_mode


# ------------------ THREADS ------------------ #
def capture_thread(): # Capture thread
    global running, img
    capture = cv2.VideoCapture(0)
    
    # IMAGE CAPTURE #
    if not capture.isOpened() or img is True:
        img_frame = cv2.imread(imgDir)
        divisor = img_frame.shape[0] // 540
        img_frame = cv2.resize(img_frame, (img_frame.shape[1] // (divisor if divisor!=0 else 1), img_frame.shape[0] // (divisor if divisor!=0 else 1)))
        if img_frame is None:
            logging.error(f"Could not load fallback image from {imgDir}.")
            return
        while running:
            try:
                raw_frame_queue.put_nowait(img_frame.copy())
            except queue.Full:
                pass  # Skip if queue is full, keep latest frame
            time.sleep(0.2)
        return
    # REAL-TIME CAPTURE #
    while running:
        ret, frame = capture.read()
        if not ret:
            break
        try:
            raw_frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                # Remove old frame and add new one to keep latest
                raw_frame_queue.get_nowait()
                raw_frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
    capture.release()

def depth_thread(): # Depth Estimation thread
    global running, system_mode
    while running:
        # Only run depth estimation in "approach" mode
        if system_mode != "approach":
            time.sleep(0.25)
            continue
        
        try:
            # Get frame from capture thread
            frame = raw_frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Resize to lower resolution to reduce inference cost (maintains aspect)
        target_w = 640
        h, w = frame.shape[:2]
        if w > target_w:
            new_h = int(h * (target_w / w))
            frame_small = cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame

        # Estimate depth map
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
            try:
                # Removes old payload and add new one to keep latest
                DE_frame_queue.get_nowait()
                DE_frame_queue.put_nowait(de_payload)
            except queue.Empty:
                pass

def YOLO_thread(): # Object Detection thread
    global running, system_mode
    while running:
        try:
            # Get frame from capture thread
            frame = raw_frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Run YOLO
        results = OD_model(frame, device=device, conf=0.7, verbose=False)  

        # Copy original frame for annotation
        annotated = frame.copy()

        # Get the different outputs from YOLO (flattened) and image center
        boxes = [box.xyxy[0].tolist() for box in results[0].boxes]
        classes = [int(box.cls[0]) for box in results[0].boxes]
        confs = [float(box.conf[0]) for box in results[0].boxes]
        names = results[0].names
        image_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        # Draw boxes on the annotated frame
        draw_boxes(annotated, boxes, classes, confs, names=names, 
                   box_color=(255,0,255), text_color=(255,0,255), thickness=2, text_scale=0.6)

        # # Draw a line to the most confident detection
        # best_idx, box_center = draw_line_to_most_confident(annotated, image_center, boxes, classes=classes, 
        #                                                    confs=confs, names=names, class_filter='chair', 
        #                                                    color=(255,0,0), thickness=2, dot_radius=5)

        # --------- FSM Mode switching  --------- #
        if confs is not None and confs[0] > 0.6:
            set_mode("approach")
            logging.info("Detected object with high confidence — switching to APPROACH mode.")
        else:
            set_mode("search")

        # Prepare structured data with detections and annotated frame
        od_payload = {
            "frame": frame,                 # original frame (BGR)
            "annotated": annotated,         # annotated frame (BGR)
            "boxes": boxes,                 # list of [x1, y1, x2, y2]
            "classes": classes,             # list of class IDs
            "confs": confs,                 # list of confidences
            # "conf_box_center": box_center,  # [x, y] coordinates of the box center
            "names": names,                 # class names mapping
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


def fusion_thread(): # Fusion thread - YOLO+Depth
    global running, system_mode
    while running:
        od_payload = None
        de_payload = None
        # Try to get latest structured payloads
        try:
            od_payload = OD_frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue # No YOLO data, skip iteration
        
        # ------ SEARCH MODE - Only show YOLO results ------ #
        if system_mode == "search":
            frame = od_payload.get("annotated")
            cv2.putText(frame, "SEARCH MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            try:
                final_frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Skip if queue is full
            continue

        # ----- APPROACH MODE - Combine YOLO and Depth Estimation -----
        if system_mode == "approach":
            try:
                de_payload = DE_frame_queue.get(timeout=0.2)
            except queue.Empty:
                de_payload = None
        
            # If we didn’t get depth yet, skip iteration
            if de_payload is None or de_payload.get("metric_depth") is None:
                logging.warning("No depth data available yet — waiting for next frame.")
                continue

            yolo_annotated = od_payload.get("annotated")
            depth_color = de_payload.get("relative_depth")
            metric_depth = de_payload.get("metric_depth")

            # Use helper to draw boxes onto depth visualization.
            boxes = od_payload.get("boxes", [])
            classes = od_payload.get("classes", [])
            confs = od_payload.get("confs", [])
            names = od_payload.get("names", {})
            # conf_box_center = od_payload.get("conf_box_center", None)

            # Get depth info for the most confident box
            roi_median_depth, max_depth, min_depth, median_depth, mean_depth = ROI_depth_info(metric_depth, boxes[0] if boxes else None)
            
            h_y, w_y = yolo_annotated.shape[:2]
            depth_frame = cv2.resize(depth_color, (w_y, h_y))
            display = depth_frame.copy()
            draw_boxes(display, boxes, classes=classes, confs=confs, names=names, box_color=(255,0,255), 
                    text_color=(255,0,255), thickness=2, text_scale=0.6)
            draw_line_to_most_confident(display, (display.shape[1]//2, display.shape[0]//2), 
                                        boxes, classes=classes, confs=confs, names=names, class_filter='cabinet', 
                                        color=(255,0,0), thickness=2, dot_radius=5)
            if roi_median_depth is not None:
                cv2.putText(display, f"Target Depth: {roi_median_depth:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if roi_median_depth < 0.8:
                    set_mode("interact")
                    logging.info("Reached target — switching to INTERACT mode.")
            
            cv2.putText(display, "APPROACH MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            try:
                final_frame_queue.put_nowait(display)
            except queue.Full:
                pass  # Skip if queue is full

        # ---------- INTERACT MODE ---------- #
        elif system_mode == "interact":
            display = od_payload.get("annotated").copy()
            cv2.putText(display, "INTERACT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            try:
                final_frame_queue.put_nowait(display)
            except queue.Full:
                pass  # Skip if queue is full


# Movement controller thread
def movement_controller_thread():
    global system_mode, movement_phase_active, movement_target, movement_target_queue, movement_done_event
    while running:
        if system_mode == "approach":
            try:
                # Wait for a movement target
                movement_target = movement_target_queue.get(timeout=1)
                movement_phase_active = True
                logging.info(f"Movement target acquired: {movement_target}")
            except queue.Empty:
                continue

        # Wait for movement to complete
        if movement_phase_active:
            logging.info("Waiting for movement to complete...")
            movement_done_event.wait()
            movement_done_event.clear()
            movement_phase_active = False
            logging.info("Movement completed.")


# ---------- MAIN DISPLAY LOOP ---------- #
if __name__ == "__main__":
    # Start threads
    threads = [
        threading.Thread(target=capture_thread),
        threading.Thread(target=YOLO_thread),
        threading.Thread(target=depth_thread),
        threading.Thread(target=fusion_thread),
        threading.Thread(target=movement_controller_thread)
    ]
    for thread in threads: thread.start()
    
    while running:
        try:
            frame = final_frame_queue.get(timeout=0.1)            
            cv2.imshow("Robot Vision System", frame)

        except queue.Empty:
            pass # No new frame available, continue loop

        # Key handling: 'q' to quit, 'c' to signal movement completion (structure control)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('r'):
            set_mode("search")
            logging.info("User requested search mode (pressed 'r')")

    for t in threads: t.join() # Wait for threads to finish
    cv2.destroyAllWindows() # Close display window
    