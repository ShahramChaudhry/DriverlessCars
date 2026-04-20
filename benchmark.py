"""
Strict per-frame latency benchmarking.

Direct adaptation of the ResNet50 CUDA-event benchmark:

  ResNet50 approach                    │ Video-YOLO adaptation
  ─────────────────────────────────────│────────────────────────────────────────
  gpu_batches pre-loaded               │ GPU tensors pre-loaded (load_video_frames)
  warmup loop, then synchronize        │ warmup_model(), then synchronize
  CUDA start/end events                │ same
  for batch in gpu_batches: model(x)   │ for tensor in tensors: _forward(t) + NMS
  times[1:] discards first run         │ run_totals[1:] discards first run
  mean_throughput = samples / mean_t   │ fps = num_frames / mean_t
"""
from __future__ import annotations
import time
import statistics
import torch
from ultralytics.utils.ops import non_max_suppression
from config import DEVICE, CONF_THRESHOLD, IOU_THRESHOLD, WARMUP_FRAMES, BENCHMARK_REPEATS
from model_loader import ModelBundle
from inference import _forward


@torch.inference_mode()
def warmup_model(
    bundle:        ModelBundle,
    tensors:       list,
    warmup_frames: int = WARMUP_FRAMES,
) -> None:
    """
    Run warmup_frames untimed forward passes.
    Ensures JIT compilation and CUDA kernel caches are warm before measurement —
    mirrors the warmup loop in the ResNet benchmark.
    """
    n = min(warmup_frames, len(tensors))
    for t in tensors[:n]:
        _ = _forward(bundle, t)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    print(f"[{bundle.mode}] Warmup complete ({n} frames, not timed).")


@torch.inference_mode()
def benchmark_video(
    bundle:  ModelBundle,
    tensors: list,
    repeats: int = BENCHMARK_REPEATS,
) -> dict:
    """
    Timed benchmark over all GPU tensors, repeated `repeats` times.

    Timing scope: nn forward pass + NMS (preprocessing excluded).
    First timed run is discarded before computing statistics —
    matches valid_times = times[1:] in the ResNet code.

    Returns a metrics dict ready for display in Gradio.
    """
    n          = len(tensors)
    run_totals = []   # total wall-clock seconds per repeat

    for r in range(repeats):
        if DEVICE.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end   = torch.cuda.Event(enable_timing=True)

            ev_start.record()
            for t in tensors:
                pred = _forward(bundle, t)
                _    = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)
            ev_end.record()

            torch.cuda.synchronize()
            elapsed = ev_start.elapsed_time(ev_end) / 1000.0   # ms → s
        else:
            t0 = time.perf_counter()
            for t in tensors:
                pred = _forward(bundle, t)
                _    = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)
            elapsed = time.perf_counter() - t0

        run_totals.append(elapsed)
        print(
            f"[{bundle.mode}] run {r + 1}/{repeats} — "
            f"{elapsed:.4f}s total | "
            f"{elapsed / n * 1000:.2f} ms/frame | "
            f"{n / elapsed:.1f} FPS"
        )

    # Discard first run — mirrors valid_times = times[1:] in ResNet benchmark
    valid = run_totals[1:] if len(run_totals) > 1 else run_totals

    mean_total = statistics.mean(valid)
    ms_runs    = [t / n * 1000 for t in valid]
    mean_ms    = statistics.mean(ms_runs)
    median_ms  = statistics.median(ms_runs)
    std_ms     = statistics.stdev(ms_runs) if len(ms_runs) > 1 else 0.0
    fps        = n / mean_total

    return {
        "mode":                bundle.mode,
        "device":              str(DEVICE),
        "num_frames":          n,
        "repeats_used":        len(valid),
        "mean_ms_per_frame":   round(mean_ms, 3),
        "median_ms_per_frame": round(median_ms, 3),
        "std_ms_per_frame":    round(std_ms, 3),
        "fps":                 round(fps, 2),
        "total_s_mean":        round(mean_total, 4),
        "all_runs_total_s":    [round(t, 4) for t in run_totals],
    }
