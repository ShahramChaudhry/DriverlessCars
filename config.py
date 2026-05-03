import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_WEIGHT   = "yolov8n.pt"   # nano = fastest; yolov8s / yolov8m for better accuracy
IMG_SIZE       = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# Mirrors ResNet benchmark philosophy exactly:
#   warmup passes are run but never timed
#   first timed run is discarded before computing stats
WARMUP_FRAMES     = 10
BENCHMARK_REPEATS = 5    # first run discarded → stats from 4 runs
MAX_DEMO_FRAMES   = 100  # cap keeps VRAM usage ~470 MB and latency manageable

# Pruning (structured / unstructured modes) — same default ratio as ResNet experiments
PRUNE_RATIO = 0.40
