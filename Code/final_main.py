# final_main.py
'''
Main script for robot vision system integrating YOLO object detection and 
Depth Anything V2's Monocular Metric depth estimation.

Supports static image mode and live camera mode from a robot (e.g., Pepper) by changing the IMG_MODE variable.
'''

import time
import logging
import threading
import queue
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import pipeline

from DepthEstimation.DE_functions import estimate_depth, ROI_depth_info
from ObjectDetection.OD_functions import draw_boxes, draw_line_to_most_confident, resize_for_yolo
from robot.robot_client import RobotClient

# ------------------------------------------------
# MANUAL SETUP
# ------------------------------------------------
IMG_MODE = False  # Set to True to use static images instead of robot camera

# Robot configs #
ROBOT_IP = "192.168.1.116" # Change this to match the Pepper used. Example ips: "192.168.1.109", "192.168.1.113"
CAMERA_ID = 0 # 0=top camera, 1=bottom camera. NOTE: in this project, only top camera is supported
RESOLUTION = 2 # 0=160x120, 1=320x240, 2=640x480, 3=1280x960

BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = str(BASE_DIR / "ObjectDetection/models/yolo11l.pt") # YOLO model needs to be downloaded separately, and put in this path
DepthAnything_MODEL_PATH = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" # This model is loaded automatically, no need to download manually

# -------------------------------------------------
# CONFIG STATICS
# -------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# For image mode: load images from Images/ folder
BASE_DIR = Path(__file__).resolve().parent
images_dir = str(BASE_DIR / "Images")
images = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
current_image_index = 0
current_img_path = str(images[current_image_index]) if images else ""

# For Pepper robot mode: Camera and image settings
FOV_H = 60.9  # Horizontal field of view in degrees
TARGET_W = 640 # Target width for YOLO input and processing

# logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -------------------------------------------------
# OTHER ROBOT LOGIC GLOBALS
# -------------------------------------------------

# Specific target object to search for
TARGET_LIST = ["cabinet", "chair", "laptop", "person"]  # List of objects to cycle through as targets
current_target_index = 0  # Index of current target in the list
TARGET_OBJECT = TARGET_LIST[current_target_index]

# Desired distance to target in meters
DIST_TO_TARGET = 1.4

# Head movement globals (search state)
head_direction = 1  # 1 = right, -1 = left
last_head_switch_time = 0  # Initialize to 0 so it moves immediately on first call
last_turn_time = 0  # Time when last turn started
turning_start_time = 0 # Time when turning of robot started
turning_enabled = False  # Flag to control turning state
turn_flag = True  # Flag to avoid immediate repeated turns
last_seen_time = 0  # Timestamp of last seen target object

# Approach state logic globals
approaching = False  # Flag to indicate if robot is currently approaching target
lost_sight_time = 0  # Timestamp when target was lost during approach


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def switch_image(direction: str):
    '''Small helper function to switch to next or previous image in the list.'''
    global current_image_index, current_img_path
    if not images:
        return
    if direction == 'next':
        current_image_index = (current_image_index + 1) % len(images)
    elif direction == 'prev':
        current_image_index = (current_image_index - 1) % len(images)
    current_img_path = str(images[current_image_index])

def run_yolo(yolo, frame_small, conf=0.6):
    '''Run YOLO inference on the given frame and return results. Returns:
    
        - annotated (np.ndarray): Annotated frame with bounding boxes.
        - boxes (list): List of bounding box coordinates.
        - classes (list): List of class indices.
        - confs (list): List of confidence scores.
        - names (dict): Class names mapping.
        - best_target_box (list or None): Bounding box of the single most confident target object.
        - best_target_conf (float): Confidence score of the most confident target object.
    '''
    results = yolo(frame_small, conf=conf, verbose=False)
    if DEVICE != "cpu": torch.cuda.synchronize()

    boxes = [b.xyxy[0].tolist() for b in results[0].boxes]
    classes = [int(b.cls[0]) for b in results[0].boxes]
    confs = [float(b.conf[0]) for b in results[0].boxes]
    names = results[0].names

    annotated = frame_small.copy()
    draw_boxes(annotated, boxes, classes, confs, names)
    
    best_target_box = None
    best_target_conf = 0.0
    
    for box, cls, conf in zip(boxes, classes, confs):
        if names[cls] == TARGET_OBJECT and conf > best_target_conf:
            best_target_box = box
            best_target_conf = conf

    return annotated, boxes, classes, confs, names, best_target_box, best_target_conf


# -------------------------------------------------
# CAMERA THREAD
# -------------------------------------------------

def camera_thread_func(robot: RobotClient, 
                       frame_queue: queue.Queue, 
                       img_mode: bool):
    '''
    Thread function to capture frames from robot camera or static images. 
    Puts frames into frame_queue.
    '''
    global current_img_path
    if img_mode:
        last_path = current_img_path
        img = cv2.imread(current_img_path)
        if img is None:
            logging.error("Could not load static image")
            return
        while True:
            if current_img_path != last_path:
                img = cv2.imread(current_img_path)
                last_path = current_img_path
                if img is None:
                    logging.error("Could not load new static image")
                    continue
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_queue.put_nowait(img.copy())
            time.sleep(0.03)
    else:
        robot.start_camera(camera_id=CAMERA_ID, resolution=RESOLUTION)
        while True:
            frame = robot.video_frame
            if frame is not None:
                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except: pass
                frame_queue.put_nowait(frame.copy())
            time.sleep(0.01)


# -------------------------------------------------
# SEARCH STATE HANDLER
# -------------------------------------------------
def search_state_handler(robot: RobotClient, 
                         annotated_frame: np.ndarray, 
                         best_box: list | None,
                         best_conf: float) -> np.ndarray:
    '''Handle SEARCH state logic: head movement, state switch, and display.'''
    global head_direction, last_head_switch_time, last_turn_time, turning_enabled, turning_start_time, turn_flag, last_seen_time
    # logging.info("In SEARCH state handler") # for debugging

    # ---------------------------------
    # DISPLAY LOGIC
    # ---------------------------------
    frame = annotated_frame.copy()
    # Display image path if in image mode
    if IMG_MODE:
        cv2.putText(frame, current_img_path, (11, frame.shape[0]-9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(frame, current_img_path, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Display system state
    cv2.putText(frame, "SEARCH STATE", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(frame, "SEARCH STATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    
    # Show frame
    cv2.imshow("Robot Vision", frame)
        
    # ---------------------------------
    # DETECTION & TIMEOUT LOGIC
    # ---------------------------------
    detection_now = best_box is not None and best_conf > 0.65

    # Update last seen time
    if detection_now:
        last_seen_time = time.time()

    # Latch: object considered visible for 1s after last detection
    target_visible = (time.time() - last_seen_time) < 1.0
    
    # ---------------------------------
    # IMAGE MODE HANDLING
    # ---------------------------------
    if IMG_MODE:
        if target_visible:
            return "approach"
        else:
            return "search"
            
    
    # ---------------------------------------------------------
    # ROBOT CASE 1: TARGET VISIBLE → ALIGN ROBOT TOWARD OBJECT
    # ---------------------------------------------------------
    if target_visible:

        # Prevent head scanning
        robot.head_position(0, 0.2)

        # If not currently turning: begin turn
        if not turning_enabled:
            logging.info(f"Detected target object '{TARGET_OBJECT}'")

            # First, if we dont have a valid box, continue searching
            if best_box is None:
                return "search"

            # Angle estimation from bounding box
            box = best_box
            box_center_x = (box[0] + box[2]) / 2
            frame_center_x = TARGET_W / 2
            offset_x = box_center_x - frame_center_x
            angle_rad = (offset_x / (TARGET_W / 2)) * np.radians(FOV_H / 2)
            angle_rad = float(-angle_rad)

            # Turn robot to face target
            logging.info(f"Turning robot by {np.degrees(angle_rad):.1f} degrees")
            robot.turn_counter_clockwise(angle_rad, block=False)

            turning_enabled = True
            turning_start_time = time.time()
            return "search"

        # If turning has been ongoing for at least 2 seconds → evaluate
        else:
            if time.time() - turning_start_time > 2.0:
                # We need updated bbox for correction
                if best_box is not None:
                    box_center_x = (best_box[0] + best_box[2]) / 2
                    frame_center_x = TARGET_W / 2
                    offset_x = box_center_x - frame_center_x
                    angle_rad = (offset_x / (TARGET_W / 2)) * np.radians(FOV_H / 2)
                    angle_rad = float(-angle_rad)

                    # If still not centered → turn again
                    if abs(angle_rad) > np.radians(10):
                        logging.info(f"Still off-center, correcting {np.degrees(angle_rad):.1f}°")
                        robot.turn_counter_clockwise(angle_rad, block=False)
                        turning_start_time = time.time()
                        return "search"
                    else:
                        # Centered → finished
                        turning_enabled = False
                        logging.info("Target centered → switching to APPROACH.")
                        return "approach"

                else:
                    # Lost visual but still within timeout
                    turning_enabled = False
                    logging.info("Lost target briefly during turn → resuming search.")
                    return "search"

        # Continue SEARCH state after turn handling
        return "search"


    # ---------------------------------------------------------
    # ROBOT CASE 2: NO TARGET VISIBLE → HEAD SWEEP + OCCASIONAL TURNS
    # ---------------------------------------------------------
    if not target_visible and not turning_enabled:
        now = time.time()

        # HEAD SWEEP
        if now - last_head_switch_time > 5.0:
            head_direction *= -1
            robot.head_position(head_direction * np.pi/6, 0.2, 0.04)
            last_head_switch_time = now

        # FULL-ROBOT TURN AFTER LONG INACTIVITY
        if now - last_turn_time > 10.0:

            if turn_flag:
                last_turn_time = now
                turn_flag = False
                return "search"

            logging.info("No target for a while → rotating 90°")
            robot.turn_counter_clockwise(np.pi/2, block=False)
            last_turn_time = now

    return "search"

# -------------------------------------------------
# APPROACH STATE HANDLER
# -------------------------------------------------
def approach_state_handler(robot: RobotClient,
                           frame: np.ndarray,
                           boxes: list,
                           classes: list,
                           confs: list,
                           names: dict,
                           best_box: list | None,
                           best_conf: float,
                           pipe) -> str:
    global approaching, lost_sight_time
    # logging.info("In APPROACH state handler") # for debugging

    # ---------------------------------
    # COMPUTE DEPTH INFORMATION
    # ---------------------------------
    
    metric_depth, relative_depth = estimate_depth(frame, pipe) # Monocular depth estimation
    roi_median_depth = None # Estimated depth to target object
    
    # Check if the target object is still detected
    target_detected = best_box is not None and best_conf > 0.65
    
    # If target detected, get its estimated depth
    if target_detected:
        roi_median_depth, *_ = ROI_depth_info(metric_depth, best_box)
        roi_median_depth = float(roi_median_depth)  # Convert to simple float so it works with Pepper
    
    # ---------------------------------
    # DISPLAY LOGIC
    # ---------------------------------
    
    # Take a copy of relative depth for display
    display = relative_depth.copy()
    
    # draw boxes and line (from Object detection)
    draw_boxes(display, boxes, classes, confs, names)
    best_box_center = draw_line_to_most_confident(display, best_box)

    # Display image path if in image mode
    if IMG_MODE:
        cv2.putText(display, current_img_path, (11, display.shape[0]-9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(display, current_img_path, (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Show system state
    cv2.putText(display, "APPROACH STATE", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(display, "APPROACH STATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    # Show target depth if available
    if roi_median_depth is not None:
        cv2.putText(display, f"Target Depth: {roi_median_depth:.1f}", (11, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(display, f"Target Depth: {roi_median_depth:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
    
    cv2.imshow("Robot Vision", display)

    # -------------------------------------------------
    # ROBOT APPROACH LOGIC
    # -------------------------------------------------

    # If in image mode, just switch states without moving, preventing robot commands
    if IMG_MODE:
        if target_detected:
            return "approach"
        else: 
            return "search"
    
    # If target is detected, move towards it
    if target_detected:
        lost_sight_time = 0  # Reset lost sight timer when target is detected
        # logging.info(f"Approaching target object at estimated depth: {roi_median_depth:.2f} meters")

        # Move closer until DIST_TO_TARGET away
        if roi_median_depth > DIST_TO_TARGET:

            # Move forward
            robot.forward(roi_median_depth - DIST_TO_TARGET, block=False) 
            approaching = True

            # Adjust head pitch to keep object vertically centered
            if best_box is not None:
                box_center_y = (best_box[1] + best_box[3]) / 2
                frame_center_y = frame.shape[0] / 2
                offset_y = box_center_y - frame_center_y
                threshold = 50  # pixels
                if abs(offset_y) > threshold:
                    pitch_adjust = (offset_y / frame.shape[0]) * 0.5  # scale adjustment
                    new_pitch = 0.2 + pitch_adjust
                    new_pitch = np.clip(new_pitch, 0.0, 0.5)  # limit pitch range
                    new_pitch = float(new_pitch)
                    robot.head_position(0, new_pitch)  # adjust pitch, keep yaw at 0

        # Switch to INTERACT state when close enough
        else:
            logging.info("Reached target object. Switching to INTERACT state.")
            robot.stop()  # Stop any movement
            approaching = False
            return "interact"
        
    # If target is lost
    else:
        # Check if we were approaching it
        if approaching:
            # If we were approaching, wait a bit before switching to INTERACT
            if lost_sight_time == 0:
                lost_sight_time = time.time()
                logging.info("Lost sight of target during approach, finishing movement.")
            elif time.time() - lost_sight_time > 2.5:
                logging.info("Finished movement after losing sight. Switching to INTERACT state.")                
                lost_sight_time = 0
                approaching = False
                return "interact"
            
        # If not approaching, switch back to SEARCH
        else:
            logging.info("Lost sight of target object. Switching back to SEARCH state.")
            return "search"
    return "approach"


# -------------------------------------------------
# INTERACT STATE HANDLER
# -------------------------------------------------
def interact_state_handler(robot: RobotClient,
                          frame_small: np.ndarray) -> str:
    '''Handle INTERACT state logic: Interact, then switch to next target and go to SEARCH.'''
    global current_target_index, TARGET_OBJECT, last_turn_time
    # logging.info("In INTERACT state handler") # for debugging
    
    # Interact with the current target
    if robot is not None:
        # robot.say(f"I have reached the target object, and am now interacting with it. Hello, {TARGET_OBJECT}!")
        robot.turn_counter_clockwise(np.pi/3, block=False)
        robot.head_position(0, 0.2)  # Center head
        last_turn_time = time.time() + 5  # Reset turn timer
    
    # Switch to the next target in the list
    current_target_index = (current_target_index + 1) % len(TARGET_LIST)
    TARGET_OBJECT = TARGET_LIST[current_target_index]
    logging.info(f"Switching target to '{TARGET_OBJECT}' and returning to SEARCH state.")
    
    return "search"

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------

def main():
    global current_img_path
    # -----------------------------------------
    # Load YOLO ONCE in main thread
    # -----------------------------------------
    logging.info(f"Starting Perception-Action system in {'IMAGE' if IMG_MODE else 'ROBOT CAMERA'} mode.")
    logging.info("Loading YOLO model…")
    yolo = YOLO(YOLO_MODEL_PATH)
    # yolo.set_classes(["red chair", "blue chair"])
    try: yolo.fuse()
    except: pass

    if DEVICE != "cpu":
        yolo.model.to(DEVICE)
        torch.cuda.synchronize()
    logging.info(f"YOLO device: {next(iter(yolo.model.parameters())).device}")

    # Depth pipeline (slow, CPU or GPU, but safe in main thread)
    logging.info("Loading Depth pipeline…")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        device=0 if DEVICE != "cpu" else -1,
        use_fast=True,
        verbose=False,
    )
    logging.info("Monocular Metric Depth Estimation pipeline loaded.")
    

    # -----------------------------------------
    # Start Pepper camera (in a thread)
    # -----------------------------------------

    # Initialize robot client if not in image mode
    robot = None
    if not IMG_MODE:
        logging.info("Connecting to robot…")
        robot = RobotClient(ROBOT_IP)
        robot.head_position(0, 0.2)  # Center head
    
    # Initialize Camera thread
    frame_queue = queue.Queue(maxsize=2)
    cam_thread = threading.Thread(
        target=camera_thread_func,
        args=(robot, frame_queue, IMG_MODE),
        daemon=True
    )
    cam_thread.start()

    # -----------------------------------------
    # Warmup for YOLO
    # -----------------------------------------
    dummy = np.zeros((TARGET_W, TARGET_W, 3), dtype=np.uint8)
    for _ in range(3):
        _ = yolo(dummy, conf=0.7, verbose=False)
        if DEVICE != "cpu": torch.cuda.synchronize()

    # -----------------------------------------
    # State machine
    # -----------------------------------------
    all_states = ["search", "approach", "interact"]
    state = all_states[0]

    logging.info("Starting main loop")

    # ------------------------------------------
    # MAIN LOOP
    # ------------------------------------------
    while True:

        # ----------------------------------
        # FRAME ACQUISITION + YOLO
        # ----------------------------------
        
        # Get frame from queue
        try:
            frame = frame_queue.get(timeout=0.5)
        except:
            continue
        
        # Resize for consistent YOLO input
        frame_small = resize_for_yolo(frame, TARGET_W)
        frame_small = np.ascontiguousarray(frame_small, dtype=np.uint8)

        # Get YOLO results
        annotated_frame, boxes, classes, confs, names, best_box, best_conf = run_yolo(yolo, frame_small)

        # ----------------------------------
        # STATE HANDLERS
        # ----------------------------------

        # SEARCH STATE -> look for target object
        if state == "search":
            state = search_state_handler(robot, annotated_frame, best_box, best_conf)
            
        # APPROACH STATE -> move towards target object        
        if state == "approach":
            state = approach_state_handler(robot, frame_small, boxes, 
                                           classes, confs, names, best_box, best_conf, pipe)
        
        # INTERACT STATE -> placeholder for future interaction logic
        if state == "interact":
            state = interact_state_handler(robot, frame_small)
        
        # ----------------------------------
        # KEY HANDLING
        # ----------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif IMG_MODE and key == ord('n'):
            switch_image('next')
        elif IMG_MODE and key == ord('b'):
            switch_image('prev')
        elif IMG_MODE and key == ord('r'):
            state = "interact"

    # --------------------------------------------------
    # Cleanup, exit
    # --------------------------------------------------
    if robot is not None:
        robot.stop_camera()
        robot.shutdown()
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()