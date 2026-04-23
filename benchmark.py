# """
# Strict per-frame latency benchmarking.

# Direct adaptation of the ResNet50 CUDA-event benchmark:

#   ResNet50 approach                    │ Video-YOLO adaptation
#   ─────────────────────────────────────│────────────────────────────────────────
#   gpu_batches pre-loaded               │ GPU tensors pre-loaded (load_video_frames)
#   warmup loop, then synchronize        │ warmup_model(), then synchronize
#   CUDA start/end events                │ same
#   for batch in gpu_batches: model(x)   │ for tensor in tensors: _forward(t) + NMS
#   times[1:] discards first run         │ run_totals[1:] discards first run
#   mean_throughput = samples / mean_t   │ fps = num_frames / mean_t
# """
# from __future__ import annotations
# import time
# import statistics
# import torch
# from ultralytics.utils.ops import non_max_suppression
# from config import DEVICE, CONF_THRESHOLD, IOU_THRESHOLD, WARMUP_FRAMES, BENCHMARK_REPEATS
# from model_loader import ModelBundle
# from inference import _forward


# @torch.inference_mode()
# def warmup_model(
#     bundle:        ModelBundle,
#     tensors:       list,
#     warmup_frames: int = WARMUP_FRAMES,
# ) -> None:
#     """
#     Run warmup_frames untimed forward passes.
#     Ensures JIT compilation and CUDA kernel caches are warm before measurement —
#     mirrors the warmup loop in the ResNet benchmark.
#     """
#     n = min(warmup_frames, len(tensors))
#     for t in tensors[:n]:
#         _ = _forward(bundle, t)
#     if DEVICE.type == "cuda":
#         torch.cuda.synchronize()
#     print(f"[{bundle.mode}] Warmup complete ({n} frames, not timed).")


# @torch.inference_mode()
# def benchmark_video(
#     bundle:  ModelBundle,
#     tensors: list,
#     repeats: int = BENCHMARK_REPEATS,
# ) -> dict:
#     """
#     Timed benchmark over all GPU tensors, repeated `repeats` times.

#     Timing scope: nn forward pass + NMS (preprocessing excluded).
#     First timed run is discarded before computing statistics —
#     matches valid_times = times[1:] in the ResNet code.

#     Returns a metrics dict ready for display in Gradio.
#     """
#     n          = len(tensors)
#     run_totals = []   # total wall-clock seconds per repeat

#     for r in range(repeats):
#         if DEVICE.type == "cuda":
#             ev_start = torch.cuda.Event(enable_timing=True)
#             ev_end   = torch.cuda.Event(enable_timing=True)

#             ev_start.record()
#             for t in tensors:
#                 pred = _forward(bundle, t)
#                 _    = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)
#             ev_end.record()

#             torch.cuda.synchronize()
#             elapsed = ev_start.elapsed_time(ev_end) / 1000.0   # ms → s
#         else:
#             t0 = time.perf_counter()
#             for t in tensors:
#                 pred = _forward(bundle, t)
#                 _    = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)
#             elapsed = time.perf_counter() - t0

#         run_totals.append(elapsed)
#         print(
#             f"[{bundle.mode}] run {r + 1}/{repeats} — "
#             f"{elapsed:.4f}s total | "
#             f"{elapsed / n * 1000:.2f} ms/frame | "
#             f"{n / elapsed:.1f} FPS"
#         )

#     # Discard first run — mirrors valid_times = times[1:] in ResNet benchmark
#     valid = run_totals[1:] if len(run_totals) > 1 else run_totals

#     mean_total = statistics.mean(valid)
#     ms_runs    = [t / n * 1000 for t in valid]
#     mean_ms    = statistics.mean(ms_runs)
#     median_ms  = statistics.median(ms_runs)
#     std_ms     = statistics.stdev(ms_runs) if len(ms_runs) > 1 else 0.0
#     fps        = n / mean_total

#     return {
#         "mode":                bundle.mode,
#         "device":              str(DEVICE),
#         "num_frames":          n,
#         "repeats_used":        len(valid),
#         "mean_ms_per_frame":   round(mean_ms, 3),
#         "median_ms_per_frame": round(median_ms, 3),
#         "std_ms_per_frame":    round(std_ms, 3),
#         "fps":                 round(fps, 2),
#         "total_s_mean":        round(mean_total, 4),
#         "all_runs_total_s":    [round(t, 4) for t in run_totals],
#     }
"""
Benchmark utilities for fair inference-time measurement.

Rules followed:
- model loading is excluded
- compile time is excluded
- warmup is excluded
- timing focuses on inference only
- CUDA events are used for accurate GPU timing
"""
from __future__ import annotations

import statistics
import torch

from ultralytics.utils import nms

from config import DEVICE, CONF_THRESHOLD, IOU_THRESHOLD
from model_loader import ModelBundle
from inference import _forward


@torch.inference_mode()
def warmup_model(
    bundle: ModelBundle,
    tensors: list[torch.Tensor],
    warmup_frames: int = 5,
    batch_size: int = 1,
    timing_scope: str = "forward+nms",
) -> None:
    """
    Warm up the model so compile/setup overhead is excluded from benchmark timing.
    """
    bundle.nn_model.eval()

    n = min(warmup_frames, len(tensors))
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    for i in range(0, n, batch_size):
        batch = torch.cat(tensors[i : i + batch_size], dim=0)
        pred = _forward(bundle, batch)
        if timing_scope == "forward+nms":
            _ = nms.non_max_suppression(
                pred,
                conf_thres=CONF_THRESHOLD,
                iou_thres=IOU_THRESHOLD,
            )
        elif timing_scope == "forward":
            pass
        else:
            raise ValueError("timing_scope must be 'forward' or 'forward+nms'")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def benchmark_video(
    bundle: ModelBundle,
    tensors: list[torch.Tensor],
    repeats: int = 5,
    batch_size: int = 1,
    timing_scope: str = "forward+nms",
) -> dict:
    """
    Benchmark inference on GPU-resident video tensors.

    Returns metrics dict with:
    - per-run times
    - median latency
    - mean latency
    - fps
    """
    bundle.nn_model.eval()
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if timing_scope not in {"forward", "forward+nms"}:
        raise ValueError("timing_scope must be 'forward' or 'forward+nms'")

    times = []

    for _ in range(repeats):
        if DEVICE.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            import time
            t0 = time.time()

        for i in range(0, len(tensors), batch_size):
            batch = torch.cat(tensors[i : i + batch_size], dim=0)
            pred = _forward(bundle, batch)
            if timing_scope == "forward+nms":
                _ = nms.non_max_suppression(
                    pred,
                    conf_thres=CONF_THRESHOLD,
                    iou_thres=IOU_THRESHOLD,
                )

        if DEVICE.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            elapsed = time.time() - t0

        times.append(elapsed)

    valid_times = times[1:] if len(times) > 1 else times
    median_t = statistics.median(valid_times)
    mean_t = statistics.mean(valid_times)
    fps = len(tensors) / mean_t if mean_t > 0 else 0.0

    ms_per_frame_runs = [(t / len(tensors)) * 1000.0 for t in valid_times]
    mean_ms = statistics.mean(ms_per_frame_runs)
    median_ms = statistics.median(ms_per_frame_runs)
    std_ms = statistics.stdev(ms_per_frame_runs) if len(ms_per_frame_runs) > 1 else 0.0

    return {
        "runs": times,
        "median_seconds": median_t,
        "mean_seconds": mean_t,
        "frames": len(tensors),
        "fps": fps,
        "batch_size": batch_size,
        "timing_scope": timing_scope,
        "mean_ms_per_frame": round(mean_ms, 3),
        "median_ms_per_frame": round(median_ms, 3),
        "std_ms_per_frame": round(std_ms, 3),
    }