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

from config       import DEVICE, WARMUP_FRAMES, BENCHMARK_REPEATS, MAX_DEMO_FRAMES, CONF_THRESHOLD
from model_loader import load_model, OPTIMIZATION_MODES
from inference    import (
    load_video_frames,
    run_video_inference,
    run_video_inference_with_fps_overlay,
    save_video,
)
from benchmark    import warmup_model, benchmark_video


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
    model_weight: str,
    mode_right: str,
    progress=gr.Progress(),
) -> tuple[str | None, str | None]:
    """
    Produce two output videos side-by-side (looping in UI):
      - Left: eager
      - Right: user-selected mode (default: amp_compile)
    Each output has a per-frame (EMA) FPS overlay.
    """
    if video_path is None:
        return None, None

    progress(0.05, desc="Loading & preprocessing frames ...")
    raw_frames, tensors, shapes, ratios, pads, fps, _ = load_video_frames(video_path)
    if not tensors:
        return None, None

    progress(0.15, desc="Loading models (cached) ...")
    bundle_eager = _get_model("eager", model_weight)
    bundle_right = _get_model(mode_right, model_weight)

    progress(0.30, desc="Running Eager video inference ...")
    eager_frames = run_video_inference_with_fps_overlay(
        bundle_eager,
        raw_frames,
        tensors,
        shapes,
        ratios,
        pads,
        overlay_label="Eager",
    )

    progress(0.65, desc=f"Running {mode_right} video inference ...")
    right_frames = run_video_inference_with_fps_overlay(
        bundle_right,
        raw_frames,
        tensors,
        shapes,
        ratios,
        pads,
        overlay_label=OPTIMIZATION_MODES.get(mode_right, {}).get("label", mode_right),
    )

    progress(0.90, desc="Saving output videos ...")
    out_a = tempfile.mktemp(suffix=".mp4")
    out_b = tempfile.mktemp(suffix=".mp4")
    save_video(eager_frames, out_a, fps)
    save_video(right_frames, out_b, fps)

    progress(1.0, desc="Done!")
    return out_a, out_b

# ── Gradio UI ─────────────────────────────────────────────────────────────────
def _device_label() -> str:
    if DEVICE.type == "cuda":
        return f"GPU — {torch.cuda.get_device_name(0)}"
    return "CPU  (no CUDA GPU — compile modes and AMP fall back to eager / off)"


_mode_table = "| Mode key | Backend |\n|---|---|\n" + "\n".join(
    f"| `{k}` | {v['label']} |" for k, v in OPTIMIZATION_MODES.items()
)

_bench_philosophy = (
    "> **Benchmark philosophy** (mirrors ResNet50 GPU experiment): "
    "model load & compile time excluded · "
    f"{WARMUP_FRAMES} warmup frames discarded · "
    "first timed run discarded · "
    "CUDA events for timing on GPU; wall-clock on CPU · "
    "NMS included in timing · "
    f"max {MAX_DEMO_FRAMES} frames per video"
)

_GRADIO_CSS = (
    ".small-upload button {"
    "  padding: 6px 10px !important;"
    "  font-size: 13px !important;"
    "  min-height: 0 !important;"
    "}"
)

with gr.Blocks(
    title="Side-by-side demo",
    theme=gr.themes.Soft(),
    css=_GRADIO_CSS,
) as demo:

    # Top controls row (small upload button, above videos)
    with gr.Row():
        upload = gr.UploadButton(
            "Upload video",
            file_types=["video"],
            file_count="single",
            type="filepath",
            elem_classes=["small-upload"],
        )
        model_weight = gr.Dropdown(
            choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            value="yolov8n.pt",
            label="Model",
            scale=1,
        )

        right_mode_choices = [
            (v["label"], k) for k, v in OPTIMIZATION_MODES.items() if k != "eager"
        ]
        right_mode = gr.Dropdown(
            choices=right_mode_choices,
            value="amp_compile",
            label="Right-side mode",
            scale=2,
        )

    # Big side-by-side video outputs
    with gr.Row():
        out_left = gr.Video(label="Eager", autoplay=True, loop=True)
        out_right = gr.Video(label="Right-side output", autoplay=True, loop=True)

    # Auto-run after upload; also rerun when mode/model changes (if upload exists).
    upload.upload(
        fn=side_by_side_videos,
        inputs=[upload, model_weight, right_mode],
        outputs=[out_left, out_right],
    )
    model_weight.change(
        fn=side_by_side_videos,
        inputs=[upload, model_weight, right_mode],
        outputs=[out_left, out_right],
    )
    right_mode.change(
        fn=side_by_side_videos,
        inputs=[upload, model_weight, right_mode],
        outputs=[out_left, out_right],
    )

    # Intentionally minimal UI: no extra benchmark/philosophy text.


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
