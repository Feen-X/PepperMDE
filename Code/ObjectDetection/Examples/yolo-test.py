import time
import cv2
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #
MODEL_PATH = "ObjectDetection/models/yolo-world-chair.pt"
IMAGE_PATH = "Images/cabinet1.jpg"
DEVICE = 0   # or "cpu"

# ---------------------------------------- #

print("Loading model...")
model = YOLO(MODEL_PATH)
model.fuse()

print("Loading image...")
img = cv2.imread(IMAGE_PATH)
    
# Resize to the same size you plan to use
TARGET_W = 640
h, w = img.shape[:2]
new_h = int(h * (TARGET_W / w))
img_resized = cv2.resize(img, (TARGET_W, new_h), interpolation=cv2.INTER_AREA)

print(f"Image shape: {img_resized.shape}")

# ---------------- WARMUP ---------------- #
print("Warming up model...")
for _ in range(3):
    _ = model(img_resized, device=DEVICE, conf=0.7, verbose=False)

# ---------------- TIMED RUNS ---------------- #
N = 10
times = []

print("Running timed inference...")
for i in range(N):
    start = time.time()
    _ = model(img_resized, device=DEVICE, conf=0.7, verbose=False)
    end = time.time()
    duration = end - start
    times.append(duration)
    print(f"Inference {i+1}: {duration:.4f} seconds")

avg_time = sum(times) / len(times)
print(f"\nAverage inference time over {N} runs: {avg_time:.4f} seconds")
