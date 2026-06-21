import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_WEIGHT   = "yolov8n.pt"   # nano = fastest; yolov8s / yolov8m for better accuracy
IMG_SIZE       = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# Mirrors ResNet benchmark philosophy exactly:
#   warmup passes are run but never timed
#   first timed run is discarded before computing stats
WARMUP_FRAMES         = 10
COMPILE_WARMUP_FRAMES = 20   # extra untimed frames before compile modes reach steady overlay FPS
BENCHMARK_REPEATS     = 5    # first run discarded → stats from 4 runs
MAX_DEMO_FRAMES       = 100  # cap keeps VRAM usage ~470 MB and latency manageable

# On-video FPS: batched forward throughput (matches fair benchmark methodology).
OVERLAY_FPS_BATCH_SIZE         = 64
OVERLAY_UNTIMED_WARMUP_FRAMES  = 10
COMPILE_OVERLAY_UNTIMED_WARMUP = 64   # one untimed batch-64 pass before timing compile modes
OVERLAY_FPS_DISPLAY_SKIP       = 0    # timed batches before showing EMA

# Pruning (structured / unstructured modes) — stronger default so effects show up in benchmarks
PRUNE_RATIO = 0.50
