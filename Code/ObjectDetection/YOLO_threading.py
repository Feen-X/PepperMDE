from ultralytics import YOLO
import cv2
import threading
from pathlib import Path

# Base directory = the folder containing this script
BASE_DIR = Path(__file__).resolve().parent

# Paths relative to the script
model_path = str(BASE_DIR / "models" / "yolo-world-chair.pt")
imgDir = str(BASE_DIR / "Images" / "Objects.jpg")

# Load YOLO model
model = YOLO(model_path)
# model.set_classes(["chair"])

# Set device to GPU (if available)
device = 0  # 0 for CUDA GPU
model.fuse()  # Optional: fuse Conv + BN layers for slightly faster inference

# Shared variables
img = True
frame = None
annotated_frame = None
lock = threading.Lock()
running = True

# Video capture thread
def capture_thread():
    global frame, running, img
    capture = cv2.VideoCapture(0)
    if not capture.isOpened() or img is True:
        print("Error: Could not open camera. Showing fallback image.")
        img = cv2.imread(imgDir)
        if img is None:
            print(f"Error: Could not load fallback image from {imgDir}.")
            return
        while running:
            with lock:
                frame = img.copy()
            # Sleep a bit to avoid busy loop
            cv2.waitKey(100)
        return

    while running:
        ret, f = capture.read()
        if not ret:
            break
        
        with lock:
            frame = f
    capture.release()

# Inference thread
def inference_thread():
    global frame, annotated_frame, running
    
    while running:
        if frame is not None:
            with lock:
                f_copy = frame.copy()
            
            image_center = (f_copy.shape[1] // 2, f_copy.shape[0] // 2)

            # Run YOLO (no need to resize manually, YOLO handles it)
            results = model(f_copy, device=device, conf=0.6, verbose=False)  

            # Start with the original frame for annotation
            annotated = f_copy.copy()
            
            # List to hold center points of detected objects
            object_centers = []

            # enumerate through detected boxes with the highest confidence box first
            for i, box in enumerate(sorted(results[0].boxes, key=lambda x: x.conf[0], reverse=True)):
                # Bounding box (already mapped back to original image size by Ultralytics)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Confidence
                conf = float(box.conf[0])

                # Class id and name
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                
                # Always draws boxes for the specified classes
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{cls_name} {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                
                # If object is a specific class (e.g., "chair"), and it is most confident
                if cls_name == "chair" and i == 0:
                    # Calculate center point of the box
                    box_center_x = int((x1 + x2) / 2)
                    box_center_y = int((y1 + y2) / 2)
                    object_centers.append((box_center_x, box_center_y))
                    cv2.line(annotated, image_center, (box_center_x, box_center_y), (255, 0, 0), 2)
                    cv2.circle(annotated, (box_center_x, box_center_y), 5, (255, 0, 0), -1)

            # Save annotated frame for display
            annotated_frame = annotated

if __name__ == "__main__":
    # Start threads
    t1 = threading.Thread(target=capture_thread)
    t2 = threading.Thread(target=inference_thread)
    t1.start()
    t2.start()

    # Display loop
    while running:
        if annotated_frame is not None:
            cv2.imshow("YOLO Live", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    # Wait for threads to finish
    t1.join()
    t2.join()
    cv2.destroyAllWindows()
