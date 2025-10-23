import cv2
import threading
import numpy as np
from transformers import pipeline
from PIL import Image

# Load Hugging Face depth estimation pipeline
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",  # You can pick Small/Medium/Large
    device=0,  # set to -1 for CPU, 0 for first CUDA GPU
    use_fast=True
)

# Shared variables for video processing and threading
frame = None
annotated_frame = None
lock = threading.Lock()
running = True

# Video capture thread
def capture_thread():
    global frame, running
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not capture.isOpened():
        print("Error: Could not open camera.")
        running = False
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

            # Convert to PIL for the pipeline
            pil_image = Image.fromarray(cv2.cvtColor(f_copy, cv2.COLOR_BGR2RGB))

            # Run Hugging Face depth estimation
            result = pipe(pil_image)

            # result["depth"] is a PIL image with depth values
            depth_pil = result["depth"]

            # Convert back to numpy array
            depth = np.array(depth_pil, dtype=np.float32)

            # Normalize depth map to 0–255 for visualization
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_uint8 = (depth_norm * 255).astype('uint8')

            # Apply colormap for visualization
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

            # Resize back to original frame shape if needed
            depth_color = cv2.resize(depth_color, (f_copy.shape[1], f_copy.shape[0]))

            with lock:
                annotated_frame = depth_color

# Start threads
t1 = threading.Thread(target=capture_thread)
t2 = threading.Thread(target=inference_thread)
t1.start()
t2.start()

# Display loop
while running:
    if annotated_frame is not None:
        cv2.imshow("DepthAnythingV2 (HF)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# Wait for threads to finish
t1.join()
t2.join()
cv2.destroyAllWindows()
