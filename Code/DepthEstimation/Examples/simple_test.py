from transformers import pipeline

pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    device=0,               # set to -1 for CPU, 0 for first CUDA GPU
    use_fast=True,
    verbose=True
)