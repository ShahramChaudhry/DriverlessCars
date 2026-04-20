"""
Gradio demo — Driverless Car Perception with Optimization Mode Comparison.

Call graph:
  load_model(mode)         →  ModelBundle  (torch.compile applied here, NOT timed)
  load_video_frames(path)  →  GPU tensors  (preprocessing offline, like gpu_batches)
  warmup_model(bundle, …)  →  triggers JIT / kernel caches  (not timed)
  benchmark_video(…)       →  CUDA-event timing  (mirrors ResNet bench)
  run_video_inference(…)   →  annotated frames for visualization
  save_video(…)            →  mp4 for Gradio output
"""
from __future__ import annotations
import tempfile
import gradio as gr
import torch

from config       import DEVICE, WARMUP_FRAMES, BENCHMARK_REPEATS, MAX_DEMO_FRAMES, CONF_THRESHOLD
from model_loader import load_model, OPTIMIZATION_MODES
from inference    import load_video_frames, run_video_inference, save_video
from benchmark    import warmup_model, benchmark_video


# ── Lazy model cache — prevents re-compiling across Gradio clicks ─────────────
_model_cache: dict = {}

def _get_model(mode: str):
    if mode not in _model_cache:
        print(f"[INFO] Loading model (mode={mode}) — compile time is excluded from metrics")
        _model_cache[mode] = load_model(mode)
    return _model_cache[mode]


# ── Core demo function ────────────────────────────────────────────────────────
def run_demo(
    video_path: str | None,
    mode:       str,
    run_bench:  bool,
    progress=gr.Progress(),
) -> tuple:
    if video_path is None:
        return None, {"error": "Please upload a video file first."}

    progress(0.05, desc="Loading / retrieving model ...")
    bundle = _get_model(mode)

    progress(0.20, desc="Loading & preprocessing frames (letterbox → GPU tensors) ...")
    raw_frames, tensors, shapes, ratios, pads, fps, orig_wh = load_video_frames(video_path)

    if not tensors:
        return None, {"error": "Could not read any frames from the video."}

    n_warmup = min(WARMUP_FRAMES, len(tensors))
    progress(0.40, desc=f"Warming up ({n_warmup} frames, not timed) ...")
    warmup_model(bundle, tensors)

    metrics: dict = {"mode": mode, "device": str(DEVICE), "num_frames": len(tensors)}

    if run_bench:
        progress(0.55, desc=f"Benchmarking ({BENCHMARK_REPEATS} timed runs, first discarded) ...")
        metrics = benchmark_video(bundle, tensors)
        metrics["benchmark_note"] = (
            "Timing scope: nn-forward + NMS only. "
            f"Preprocessing excluded (pre-loaded to GPU). "
            f"Warmup: {n_warmup} frames. "
            "First timed run discarded (mirrors ResNet benchmark)."
        )

    progress(0.80, desc="Running annotated inference pass ...")
    annotated = run_video_inference(bundle, raw_frames, tensors, shapes, ratios, pads)

    progress(0.93, desc="Saving output video ...")
    out_path = tempfile.mktemp(suffix=".mp4")
    save_video(annotated, out_path, fps)

    if not run_bench:
        metrics["note"] = "Enable 'Run benchmark' to get per-frame timing metrics."

    progress(1.0, desc="Done!")
    return out_path, metrics


# ── Side-by-side comparison ───────────────────────────────────────────────────
def compare_modes(
    video_path: str | None,
    mode_a:     str,
    mode_b:     str,
    progress=gr.Progress(),
) -> tuple[dict, dict]:
    if video_path is None:
        return {"error": "Upload a video."}, {"error": "Upload a video."}

    progress(0.05, desc="Loading & preprocessing video frames ...")
    _, tensors, *_ = load_video_frames(video_path)

    results: dict[str, dict] = {}
    for i, mode in enumerate([mode_a, mode_b]):
        progress(0.10 + i * 0.45, desc=f"Benchmarking mode: {mode} ...")
        bundle = _get_model(mode)
        warmup_model(bundle, tensors)
        results[mode] = benchmark_video(bundle, tensors)

    return results[mode_a], results[mode_b]


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_device_label = (
    f"GPU — {torch.cuda.get_device_name(0)}"
    if torch.cuda.is_available()
    else "CPU  (compile modes and AMP will fall back to eager)"
)

_mode_table = "| Mode key | Description |\n|---|---|\n" + "\n".join(
    f"| `{k}` | {v['label']} |" for k, v in OPTIMIZATION_MODES.items()
)

_bench_philosophy = (
    "> **Benchmark philosophy** (mirrors ResNet50 GPU experiment): "
    "model load & compile time excluded · "
    f"{WARMUP_FRAMES} warmup frames discarded · "
    "first timed run discarded · "
    "CUDA events for sub-ms accuracy · "
    "NMS included in timing · "
    f"max {MAX_DEMO_FRAMES} frames per video"
)

with gr.Blocks(
    title="Driverless Car Perception — Optimisation Demo",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        f"# Driverless Car Perception — Optimisation Demo\n"
        f"**Device:** {_device_label}  |  "
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
                    mode_sel = gr.Dropdown(
                        choices=list(OPTIMIZATION_MODES.keys()),
                        value="eager",
                        label="Optimisation Mode",
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
                inputs=[vid_in, mode_sel, bench_cb],
                outputs=[vid_out, metrics_out],
            )

        # ── Tab 2: Side-by-side comparison ───────────────────────────────────
        with gr.TabItem("Compare Two Modes"):
            gr.Markdown(
                "Run the **same video** through two optimisation modes "
                "and compare per-frame latency / FPS side-by-side."
            )
            cmp_vid = gr.Video(label="Input Video")
            with gr.Row():
                cmp_a = gr.Dropdown(
                    choices=list(OPTIMIZATION_MODES.keys()),
                    value="eager",
                    label="Mode A",
                )
                cmp_b = gr.Dropdown(
                    choices=list(OPTIMIZATION_MODES.keys()),
                    value="amp_compile_reduce_overhead",
                    label="Mode B",
                )
            cmp_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                cmp_out_a = gr.JSON(label="Mode A — Metrics")
                cmp_out_b = gr.JSON(label="Mode B — Metrics")

            cmp_btn.click(
                fn=compare_modes,
                inputs=[cmp_vid, cmp_a, cmp_b],
                outputs=[cmp_out_a, cmp_out_b],
            )

    gr.Markdown(
        "---\n"
        "**Tips for a live demo**\n"
        "- Upload a 10–30 s front-camera clip for fast results\n"
        "- `compile_max_autotune` has a 5–15 min first-run compile penalty"
        " — trigger it once before the live presentation\n"
        "- Swap `MODEL_WEIGHT = 'yolov8s.pt'` in `config.py` for better detection accuracy\n"
        "- All compiled models are cached in memory — switching modes after first load is instant\n"
        f"- To process longer clips increase `MAX_DEMO_FRAMES` in `config.py` (currently {MAX_DEMO_FRAMES})"
    )


if __name__ == "__main__":
    demo.launch(share=False)
