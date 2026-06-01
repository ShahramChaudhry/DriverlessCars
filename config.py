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
COMPILE_WARMUP_FRAMES = 50   # torch.compile autotune needs more untimed frames in the demo
BENCHMARK_REPEATS     = 5    # first run discarded → stats from 4 runs
MAX_DEMO_FRAMES       = 100  # cap keeps VRAM usage ~470 MB and latency manageable

# Side-by-side panel benchmarks (CUDA-event timing, mirrors run_demo)
SIDE_BY_SIDE_BENCH_BATCH_SIZE = 64
SIDE_BY_SIDE_BENCH_SCOPE      = "forward"  # use "forward+nms" to include NMS in timed path
COMPILE_BENCHMARK_DISCARD_RUNS = 2   # torch.compile: 2nd timed run can still be autotuning
COMPILE_BENCHMARK_REPEATS      = 7   # extra timed runs so enough remain after discard

# On-video FPS uses same batched forward throughput as JSON benchmark (not per-frame batch=1)
OVERLAY_FPS_BATCH_SIZE         = SIDE_BY_SIDE_BENCH_BATCH_SIZE
OVERLAY_FPS_SCOPE              = SIDE_BY_SIDE_BENCH_SCOPE  # "forward" — NMS/annotate after timer
OVERLAY_UNTIMED_WARMUP_FRAMES  = 10
COMPILE_OVERLAY_UNTIMED_WARMUP = 25
OVERLAY_FPS_BATCH_SKIP         = 1    # timed batches before showing EMA

# Pruning (structured / unstructured modes) — stronger default so effects show up in benchmarks
PRUNE_RATIO = 0.50
