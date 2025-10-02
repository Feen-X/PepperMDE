import cv2
import torch
import threading

from DepthEstimation.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Shared variables for video processing and threading
frame = None
annotated_frame = None
lock = threading.Lock()
running = True

# Video capture thread
def capture_thread():
    global frame, running
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
            # Resize for faster inference (optional) and keep aspect ratio
            resized = cv2.resize(f_copy, (640, 640))
            # Run inference for depth estimation
            depth = model.infer_image(resized)  # Should return HxW raw depth map (numpy)
            # Normalize depth map to 0-1
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            # Scale to 0-255 and convert to uint8
            depth_uint8 = (depth_norm * 255).astype('uint8')
            # Apply colormap for visualization
            depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            # Resize to match display window size (optional)
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
        cv2.imshow("Model", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# Wait for threads to finish
t1.join()
t2.join()
cv2.destroyAllWindows()

def resize_with_aspect_ratio(img, target_size=640):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Pad to square
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, (top, bottom, left, right), (new_w, new_h)
