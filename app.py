"""
Gradio demo — Driverless Car Perception with Optimization Mode Comparison.

Call graph:
  load_model(mode)         →  ModelBundle  (torch.compile applied here, NOT timed)
  load_video_frames(path)  →  GPU tensors on CUDA (preprocessing offline, like gpu_batches)
  warmup_model(bundle, …)  →  triggers JIT / kernel caches  (not timed)
  benchmark_video(…)       →  CUDA-event timing on GPU (mirrors ResNet bench)
  run_video_inference(…)   →  annotated frames for visualization
  save_video(…)            →  mp4 for Gradio output
"""
from __future__ import annotations
import os
import tempfile

import gradio as gr
import torch

from config       import (
    DEVICE,
    MODEL_WEIGHT,
    WARMUP_FRAMES,
    COMPILE_WARMUP_FRAMES,
    BENCHMARK_REPEATS,
    MAX_DEMO_FRAMES,
    CONF_THRESHOLD,
    SIDE_BY_SIDE_BENCH_BATCH_SIZE,
    SIDE_BY_SIDE_BENCH_SCOPE,
    COMPILE_BENCHMARK_DISCARD_RUNS,
    COMPILE_BENCHMARK_REPEATS,
)
from model_loader import load_model, OPTIMIZATION_MODES, DEFAULT_COMPARE_MODE
from inference    import (
    load_video_frames,
    run_video_inference,
    run_video_inference_with_fps_overlay,
    save_video,
)
from benchmark    import warmup_model, warmup_full_video_passes, benchmark_video


def _mode_radio_choices() -> list[tuple[str, str]]:
    """(visible label, registry key) — mirrors model_loader.OPTIMIZATION_MODES order."""
    return [(v["label"], k) for k, v in OPTIMIZATION_MODES.items()]


# ── Lazy model cache — prevents re-compiling across Gradio clicks ─────────────
_model_cache: dict = {}

def _get_model(mode: str, weight: str):
    key = f"{weight}::{mode}"
    if key not in _model_cache:
        print(
            f"[INFO] Loading model (weight={weight}, mode={mode}) "
            f"— compile time is excluded from metrics"
        )
        _model_cache[key] = load_model(mode, weight=weight)
    return _model_cache[key]


def _warmup_frames_for_mode(mode: str) -> int:
    """Compiled modes need extra untimed frames before FPS overlay reflects steady state."""
    if OPTIMIZATION_MODES.get(mode, {}).get("compile_mode") and DEVICE.type == "cuda":
        return max(WARMUP_FRAMES, COMPILE_WARMUP_FRAMES)
    return WARMUP_FRAMES


def _bench_batch_size(num_frames: int) -> int:
    return min(SIDE_BY_SIDE_BENCH_BATCH_SIZE, num_frames)


def _benchmark_discard_runs(mode_key: str) -> int:
    if OPTIMIZATION_MODES.get(mode_key, {}).get("compile_mode"):
        return COMPILE_BENCHMARK_DISCARD_RUNS
    return 1


def _benchmark_repeats(mode_key: str) -> int:
    if OPTIMIZATION_MODES.get(mode_key, {}).get("compile_mode"):
        return max(BENCHMARK_REPEATS, COMPILE_BENCHMARK_REPEATS)
    return BENCHMARK_REPEATS


def _warmup_for_benchmark(bundle, mode_key: str, tensors: list) -> None:
    """Warm up at the same batch size / scope as the JSON benchmark."""
    bs = _bench_batch_size(len(tensors))
    scope = SIDE_BY_SIDE_BENCH_SCOPE
    n = _warmup_frames_for_mode(mode_key)
    warmup_model(bundle, tensors, warmup_frames=n, batch_size=bs, timing_scope=scope)
    if OPTIMIZATION_MODES.get(mode_key, {}).get("compile_mode"):
        warmup_full_video_passes(bundle, tensors, passes=3, batch_size=bs, timing_scope=scope)


def _collect_benchmark_metrics(
    bundle,
    mode_key: str,
    tensors: list,
) -> dict:
    """CUDA-event benchmark after warmup; compile modes discard extra cold timed runs."""
    batch_size = _bench_batch_size(len(tensors))
    discard = _benchmark_discard_runs(mode_key)
    metrics = benchmark_video(
        bundle,
        tensors,
        repeats=_benchmark_repeats(mode_key),
        batch_size=batch_size,
        timing_scope=SIDE_BY_SIDE_BENCH_SCOPE,
        discard_runs=discard,
    )
    n_warm = _warmup_frames_for_mode(mode_key)
    metrics.update(
        {
            "mode": mode_key,
            "label": OPTIMIZATION_MODES.get(mode_key, {}).get("label", mode_key),
            "model_weight": MODEL_WEIGHT,
            "warmup_frames": n_warm,
            "benchmark_note": (
                f"Timing scope: {SIDE_BY_SIDE_BENCH_SCOPE}, batch_size={batch_size}. "
                "Preprocessing excluded (pre-loaded to GPU). "
                f"Warmup at benchmark batch size (+ full-video passes for compile). "
                f"First {discard} timed run(s) discarded. "
                "fps uses median of remaining runs (steady state)."
            ),
            "mean_seconds": round(metrics["mean_seconds"], 6),
            "median_seconds": round(metrics["median_seconds"], 6),
        }
    )
    return metrics


# ── Core demo function ────────────────────────────────────────────────────────
def run_demo(
    video_path: str | None,
    mode:       str,
    run_bench:  bool,
    model_weight: str,
    batch_size: int,
    timing_scope: str,
    progress=gr.Progress(),
) -> tuple:
    try:
        # Gradio sliders often pass numeric values as floats (e.g., 8.0)
        batch_size = int(batch_size)

        if video_path is None:
            return None, {"error": "Please upload a video file first."}

        progress(0.05, desc="Loading / retrieving model ...")
        bundle = _get_model(mode, model_weight)

        progress(0.20, desc="Loading & preprocessing frames (letterbox → device tensors) ...")
        raw_frames, tensors, shapes, ratios, pads, fps, orig_wh = load_video_frames(video_path)

        if not tensors:
            return None, {"error": "Could not read any frames from the video."}

        n_warmup = min(WARMUP_FRAMES, len(tensors))
        progress(0.40, desc=f"Warming up ({n_warmup} frames, not timed) ...")
        warmup_model(
            bundle,
            tensors,
            warmup_frames=WARMUP_FRAMES,
            batch_size=batch_size,
            timing_scope=timing_scope,
        )

        metrics: dict = {"mode": mode, "device": str(DEVICE), "num_frames": len(tensors)}

        if run_bench:
            progress(0.55, desc=f"Benchmarking ({BENCHMARK_REPEATS} timed runs, first discarded) ...")
            metrics = benchmark_video(
                bundle,
                tensors,
                repeats=BENCHMARK_REPEATS,
                batch_size=batch_size,
                timing_scope=timing_scope,
            )
            metrics["benchmark_note"] = (
                f"Timing scope: {timing_scope}. "
                f"Preprocessing excluded (pre-loaded to GPU). "
                f"Warmup: {n_warmup} frames. "
                "First timed run discarded (mirrors ResNet benchmark)."
            )
            metrics["model_weight"] = model_weight

        progress(0.80, desc="Running annotated inference pass ...")
        annotated = run_video_inference(bundle, raw_frames, tensors, shapes, ratios, pads)

        progress(0.93, desc="Saving output video ...")
        out_path = tempfile.mktemp(suffix=".mp4")
        save_video(annotated, out_path, fps)

        if not run_bench:
            metrics["note"] = "Enable 'Run benchmark' to get per-frame timing metrics."

        progress(1.0, desc="Done!")
        return out_path, metrics
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, {"error": str(e)}


def side_by_side_videos(
    video_path: str | None,
    mode_right: str,
    progress=gr.Progress(),
) -> tuple[str | None, str | None, dict | None, dict | None]:
    """
    Produce two output videos side-by-side (looping in UI):
      - Left: eager
      - Right: user-selected mode (default: amp_compile)
    Each output has a per-frame (EMA) FPS overlay plus benchmark JSON below.
    """
    if video_path is None:
        return None, None, None, None

    progress(0.05, desc="Loading & preprocessing frames ...")
    raw_frames, tensors, shapes, ratios, pads, fps, _ = load_video_frames(video_path)
    if not tensors:
        return None, None, None, None

    progress(0.15, desc="Loading models (cached) ...")
    bundle_eager = _get_model("eager", MODEL_WEIGHT)
    bundle_right = _get_model(mode_right, MODEL_WEIGHT)

    progress(0.22, desc="Warming up Eager (benchmark batch size) ...")
    _warmup_for_benchmark(bundle_eager, "eager", tensors)

    progress(0.28, desc=f"Warming up {mode_right} (benchmark batch size) ...")
    _warmup_for_benchmark(bundle_right, mode_right, tensors)

    # Per-frame overlay path (batch=1); separate from batched JSON benchmark.
    bs1 = 1
    scope_overlay = "forward+nms"
    warmup_model(
        bundle_eager,
        tensors,
        warmup_frames=_warmup_frames_for_mode("eager"),
        batch_size=bs1,
        timing_scope=scope_overlay,
    )
    warmup_model(
        bundle_right,
        tensors,
        warmup_frames=_warmup_frames_for_mode(mode_right),
        batch_size=bs1,
        timing_scope=scope_overlay,
    )

    progress(0.32, desc="Benchmarking Eager ...")
    metrics_eager = _collect_benchmark_metrics(bundle_eager, "eager", tensors)

    progress(0.38, desc=f"Benchmarking {mode_right} ...")
    metrics_right = _collect_benchmark_metrics(bundle_right, mode_right, tensors)

    progress(0.42, desc="Running Eager video inference ...")
    eager_frames = run_video_inference_with_fps_overlay(
        bundle_eager,
        raw_frames,
        tensors,
        shapes,
        ratios,
        pads,
        overlay_label="Eager",
    )

    mode_name = OPTIMIZATION_MODES.get(mode_right, {}).get("overlay_label", mode_right)
    progress(0.65, desc=f"Running {mode_name} video inference ...")
    right_frames = run_video_inference_with_fps_overlay(
        bundle_right,
        raw_frames,
        tensors,
        shapes,
        ratios,
        pads,
        overlay_label=OPTIMIZATION_MODES.get(mode_right, {}).get(
            "overlay_label", mode_right
        ),
    )

    progress(0.90, desc="Saving output videos ...")
    out_a = tempfile.mktemp(suffix=".mp4")
    out_b = tempfile.mktemp(suffix=".mp4")
    save_video(eager_frames, out_a, fps)
    save_video(right_frames, out_b, fps)

    progress(1.0, desc="Done!")
    return out_a, out_b, metrics_eager, metrics_right

# ── Gradio UI ─────────────────────────────────────────────────────────────────
def _device_label() -> str:
    if DEVICE.type == "cuda":
        return f"**Device:** {torch.cuda.get_device_name(0)} (CUDA)"
    return "**Device:** CPU — compile and AMP modes may fall back to eager"


def _right_mode_dropdown_choices() -> list[tuple[str, str]]:
    """(visible label, registry key) for the right-hand comparison dropdown."""
    out: list[tuple[str, str]] = []
    for key, cfg in OPTIMIZATION_MODES.items():
        if key == "eager":
            continue
        label = cfg["label"]
        if cfg.get("ui_default"):
            label = f"{label} (default)"
        out.append((label, key))
    return out


_GRADIO_CSS = """
.demo-wrap { max-width: 1200px; margin: 0 auto; }
.demo-header { margin-bottom: 0.25rem !important; }
.demo-header p { opacity: 0.88; line-height: 1.5; }
.demo-controls {
  align-items: flex-end;
  gap: 1rem;
  margin: 0.75rem 0 1rem 0 !important;
}
.demo-controls .small-upload button {
  padding: 0.55rem 1.1rem !important;
  font-size: 0.95rem !important;
  min-height: 0 !important;
  border-radius: 8px !important;
}
.demo-videos { gap: 1rem !important; align-items: flex-start !important; }
.demo-videos .video-container { border-radius: 10px; overflow: hidden; }
.demo-panel-title { margin: 0 0 0.35rem 0 !important; font-size: 1rem !important; }
.demo-panel-title p { margin: 0 !important; opacity: 0.92; }
"""

with gr.Blocks(
    title="Driverless Cars — Perception Demo",
    theme=gr.themes.Soft(),
    css=_GRADIO_CSS,
    elem_classes=["demo-wrap"],
) as demo:

    gr.Markdown(
        f"""
# Driverless Cars — real-time perception demo

Compare **YOLOv8n** object detection side by side: a fixed **eager FP32 baseline** (left) vs an
**optimized mode** you choose (right). Each video shows **inference FPS** on the frame (forward + NMS, after warmup). Below each
panel, **benchmark JSON** reports CUDA-timed throughput (`batch_size={SIDE_BY_SIDE_BENCH_BATCH_SIZE}`,
scope `{SIDE_BY_SIDE_BENCH_SCOPE}`) after warmup — first timed run discarded.

{_device_label()} · Model: `{MODEL_WEIGHT}` · Max {MAX_DEMO_FRAMES} frames per clip

Upload a driving clip, pick a mode on the right, and both panels re-run automatically.
        """,
        elem_classes=["demo-header"],
    )

    with gr.Row(elem_classes=["demo-controls"]):
        upload = gr.UploadButton(
            "Upload video",
            file_types=["video"],
            file_count="single",
            type="filepath",
            elem_classes=["small-upload"],
            scale=1,
        )
        right_mode = gr.Dropdown(
            choices=_right_mode_dropdown_choices(),
            value=DEFAULT_COMPARE_MODE,
            label="Right panel mode",
            scale=2,
        )

    with gr.Row(elem_classes=["demo-videos"]):
        with gr.Column():
            gr.Markdown("**Left — Eager (baseline)**", elem_classes=["demo-panel-title"])
            out_left = gr.Video(autoplay=True, loop=True, show_label=False)
            bench_left = gr.JSON(label="Benchmark metrics")
        with gr.Column():
            gr.Markdown("**Right — optimized mode**", elem_classes=["demo-panel-title"])
            out_right = gr.Video(autoplay=True, loop=True, show_label=False)
            bench_right = gr.JSON(label="Benchmark metrics")

    _demo_inputs = [upload, right_mode]
    _demo_outputs = [out_left, out_right, bench_left, bench_right]

    upload.upload(fn=side_by_side_videos, inputs=_demo_inputs, outputs=_demo_outputs)
    right_mode.change(fn=side_by_side_videos, inputs=_demo_inputs, outputs=_demo_outputs)


def _want_gradio_share() -> bool:
    """Public Gradio link when GRADIO_SHARE=1 or when running inside Google Colab."""
    v = os.environ.get("GRADIO_SHARE", "").lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def launch_gradio(*, share: bool | None = None) -> None:
    """Start the Gradio server (used by python app.py and Colab notebooks)."""
    if share is None:
        share = _want_gradio_share()
    demo.launch(
        share=share,
        server_name="0.0.0.0" if share else "127.0.0.1",
    )


if __name__ == "__main__":
    launch_gradio()
