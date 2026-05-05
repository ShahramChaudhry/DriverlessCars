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


# ── Side-by-side comparison ───────────────────────────────────────────────────
def compare_modes(
    video_path: str | None,
    mode_a:     str,
    mode_b:     str,
    model_weight: str,
    batch_size: int,
    timing_scope: str,
    progress=gr.Progress(),
) -> tuple[dict, dict]:
    if video_path is None:
        return {"error": "Upload a video."}, {"error": "Upload a video."}
    batch_size = int(batch_size)

    progress(0.05, desc="Loading & preprocessing video frames ...")
    _, tensors, *_ = load_video_frames(video_path)

    results: dict[str, dict] = {}
    for i, mode in enumerate([mode_a, mode_b]):
        progress(0.10 + i * 0.45, desc=f"Benchmarking mode: {mode} ...")
        bundle = _get_model(mode, model_weight)
        warmup_model(
            bundle,
            tensors,
            warmup_frames=WARMUP_FRAMES,
            batch_size=batch_size,
            timing_scope=timing_scope,
        )
        results[mode] = benchmark_video(
            bundle,
            tensors,
            repeats=BENCHMARK_REPEATS,
            batch_size=batch_size,
            timing_scope=timing_scope,
        )
        results[mode]["model_weight"] = model_weight

    return results[mode_a], results[mode_b]


def compare_fixed_videos(
    video_path: str | None,
    model_weight: str,
    progress=gr.Progress(),
) -> tuple[str | None, str | None]:
    """
    Produce two output videos side-by-side:
      - Eager
      - AMP + torch.compile (max-autotune-no-cudagraphs)
    Each output has a per-frame FPS overlay.
    """
    if video_path is None:
        return None, None

    progress(0.05, desc="Loading & preprocessing frames ...")
    raw_frames, tensors, shapes, ratios, pads, fps, _ = load_video_frames(video_path)
    if not tensors:
        return None, None

    progress(0.15, desc="Loading models (cached) ...")
    bundle_eager = _get_model("eager", model_weight)
    bundle_ampc = _get_model("amp_compile", model_weight)

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

    progress(0.65, desc="Running AMP+Compile video inference ...")
    ampc_frames = run_video_inference_with_fps_overlay(
        bundle_ampc,
        raw_frames,
        tensors,
        shapes,
        ratios,
        pads,
        overlay_label="AMP + Compile",
    )

    progress(0.90, desc="Saving output videos ...")
    out_a = tempfile.mktemp(suffix=".mp4")
    out_b = tempfile.mktemp(suffix=".mp4")
    save_video(eager_frames, out_a, fps)
    save_video(ampc_frames, out_b, fps)

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

with gr.Blocks(
    title="Driverless Car Perception — Optimisation Demo",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        f"# Driverless Car Perception — Optimisation Demo\n"
        f"**Device:** {_device_label()}  |  "
        f"**Model:** YOLOv8-nano (COCO 80 classes)  |  "
        f"**Conf:** {CONF_THRESHOLD}"
    )
    gr.Markdown(_bench_philosophy)

    with gr.Tabs():

        # ── Tab 1: Single mode ────────────────────────────────────────────────
        with gr.TabItem("Single Mode — Inference + Benchmark"):

            with gr.Row():
                with gr.Column(scale=1):
                    vid_in   = gr.Video(label="Input Driving Video")
                    model_sel = gr.Dropdown(
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                        value="yolov8n.pt",
                        label="Model Weight",
                    )
                    gr.Markdown(
                        "**Modes:** "
                        + " · ".join(f"`{k}`" for k in OPTIMIZATION_MODES)
                    )
                    mode_sel = gr.Radio(
                        choices=_mode_radio_choices(),
                        value="eager",
                        label="Optimisation mode",
                    )
                    timing_sel = gr.Radio(
                        choices=["forward", "forward+nms"],
                        value="forward+nms",
                        label="Benchmark Timing Scope",
                    )
                    batch_sel = gr.Slider(
                        minimum=1,
                        maximum=64,
                        value=1,
                        step=1,
                        label="Batch size (frames per forward call)",
                    )
                    gr.Markdown(_mode_table)
                    bench_cb = gr.Checkbox(
                        value=True,
                        label=f"Run benchmark  ({BENCHMARK_REPEATS} timed runs, adds ~5× inference time)",
                    )
                    run_btn = gr.Button("Run Inference", variant="primary")

                with gr.Column(scale=1):
                    vid_out     = gr.Video(label="Annotated Output")
                    metrics_out = gr.JSON(label="Performance Metrics")

            run_btn.click(
                fn=run_demo,
                inputs=[vid_in, mode_sel, bench_cb, model_sel, batch_sel, timing_sel],
                outputs=[vid_out, metrics_out],
            )

        # ── Tab 2: Side-by-side comparison ───────────────────────────────────
        with gr.TabItem("Compare Two Modes"):
            gr.Markdown(
                "Outputs two looping videos side-by-side with FPS overlay:\n"
                "- **Eager**\n"
                "- **AMP + torch.compile**"
            )
            cmp_model = gr.Dropdown(
                choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                value="yolov8n.pt",
                label="Model Weight",
            )
            cmp_upload = gr.Video(label="Input Video")
            cmp_btn = gr.Button("Run Side-by-side", variant="primary")

            with gr.Row():
                out_eager = gr.Video(label="Eager (FPS overlaid)", autoplay=True, loop=True)
                out_ampc  = gr.Video(label="AMP + Compile (FPS overlaid)", autoplay=True, loop=True)

            cmp_btn.click(
                fn=compare_fixed_videos,
                inputs=[cmp_upload, cmp_model],
                outputs=[out_eager, out_ampc],
            )

    gr.Markdown(
        "---\n"
        "**Tips for a live demo**\n"
        "- **Google Colab:** Runtime → Change runtime type → **GPU**, then upload and run "
        "`colab/DriverlessCars_Colab.ipynb` from this repository\n"
        "- Upload a 10–30 s front-camera clip for fast results\n"
        "- **torch.compile()** modes use `max-autotune-no-cudagraphs` (YOLO-safe); "
        "first-run compile can take several minutes — run once before a live demo\n"
        "- Swap `MODEL_WEIGHT = 'yolov8s.pt'` in `config.py` for better detection accuracy\n"
        "- All compiled models are cached in memory — switching modes after first load is instant\n"
        f"- To process longer clips increase `MAX_DEMO_FRAMES` in `config.py` (currently {MAX_DEMO_FRAMES})"
    )


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
    """Start the Gradio server (used by `python app.py` and Colab notebooks)."""
    if share is None:
        share = _want_gradio_share()
    demo.launch(
        share=share,
        server_name="0.0.0.0" if share else "127.0.0.1",
    )


if __name__ == "__main__":
    launch_gradio()
