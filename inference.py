# """
# Preprocessing, forward pass, NMS, annotation, and video I/O.

# Preprocessing (letterbox + tensor creation) is done offline in load_video_frames
# so that GPU tensors are pre-resident before benchmarking begins — the same
# strategy as building gpu_batches in the ResNet50 benchmark.
# """
# from __future__ import annotations
# import tempfile
# import cv2
# import numpy as np
# import torch
# from ultralytics.utils import nms, ops
# from ultralytics.utils.plotting import Annotator, colors as yolo_colors
# from config import DEVICE, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, MAX_DEMO_FRAMES
# from model_loader import ModelBundle


# # ── Preprocessing ──────────────────────────────────────────────────────────────

# def _letterbox(
#     img_rgb: np.ndarray,
#     new_shape: int = 640,
# ) -> tuple[np.ndarray, float, tuple[float, float]]:
#     """
#     Resize + pad to a square while maintaining aspect ratio (grey padding).
#     Returns: (padded_img, scale_ratio, (pad_w/2, pad_h/2))
#     """
#     h, w   = img_rgb.shape[:2]
#     r      = min(new_shape / h, new_shape / w)
#     nw, nh = int(round(w * r)), int(round(h * r))
#     dw     = (new_shape - nw) / 2
#     dh     = (new_shape - nh) / 2
#     resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
#     top, bot = int(round(dh - 0.1)), int(round(dh + 0.1))
#     lft, rgt = int(round(dw - 0.1)), int(round(dw + 0.1))
#     padded = cv2.copyMakeBorder(
#         resized, top, bot, lft, rgt,
#         cv2.BORDER_CONSTANT, value=(114, 114, 114),
#     )
#     return padded, r, (dw, dh)


# def preprocess_frame(
#     frame_bgr: np.ndarray,
#     img_size: int = IMG_SIZE,
# ) -> tuple[torch.Tensor, tuple[int, int], float, tuple[float, float]]:
#     """BGR frame → normalised float32 [1, 3, H, W] tensor + metadata for rescaling."""
#     h0, w0        = frame_bgr.shape[:2]
#     rgb           = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#     padded, r, p  = _letterbox(rgb, img_size)
#     t = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
#     return t.unsqueeze(0), (h0, w0), r, p


# def load_video_frames(
#     video_path: str,
#     img_size:   int = IMG_SIZE,
#     max_frames: int = MAX_DEMO_FRAMES,
# ) -> tuple[list, list, list, list, list, float, tuple[int, int]]:
#     """
#     Read all frames, preprocess them, move tensors to GPU.
#     This is the video equivalent of building gpu_batches in the ResNet benchmark —
#     all preprocessing happens here so it is excluded from timing.

#     Returns:
#         raw_bgr_frames, gpu_tensors, orig_shapes, ratios, pads, fps, (w, h)
#     """
#     cap   = cv2.VideoCapture(video_path)
#     fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     raw_frames, tensors, shapes, ratios, pads = [], [], [], [], []

#     while len(raw_frames) < max_frames:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         t, orig_shape, ratio, pad = preprocess_frame(frame, img_size)
#         raw_frames.append(frame.copy())
#         tensors.append(t.to(DEVICE, non_blocking=True))   # pre-resident on GPU
#         shapes.append(orig_shape)
#         ratios.append(ratio)
#         pads.append(pad)

#     cap.release()
#     return raw_frames, tensors, shapes, ratios, pads, fps, (orig_w, orig_h)


# # ── Forward pass ──────────────────────────────────────────────────────────────

# def _forward(bundle: ModelBundle, tensor: torch.Tensor) -> torch.Tensor:
#     """
#     Single forward pass with optional AMP.
#     The YOLOv8 Detect head returns (pred_tensor, feature_maps) in eval mode;
#     we take only pred_tensor and pass it to NMS.
#     """
#     with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=bundle.use_amp):
#         out = bundle.nn_model(tensor)
#     # Detect head: eval mode returns (predictions, raw_feature_list)
#     return out[0] if isinstance(out, (list, tuple)) else out


# # ── Annotation ────────────────────────────────────────────────────────────────

# def annotate_frame(
#     frame_bgr:  np.ndarray,
#     det:        torch.Tensor,
#     names:      dict,
#     orig_shape: tuple[int, int],
# ) -> np.ndarray:
#     """
#     Draw detections on frame_bgr.
#     det: [N, 6] tensor (x1, y1, x2, y2, conf, cls) in letterboxed-640 space.
#     """
#     annotator = Annotator(frame_bgr.copy())
#     if det is not None and len(det):
#         det_cpu = det.clone()
#         det_cpu[:, :4] = scale_boxes(
#             (IMG_SIZE, IMG_SIZE), det_cpu[:, :4], orig_shape
#         )
#         for *xyxy, conf, cls in det_cpu.tolist():
#             label = f"{names[int(cls)]}  {conf:.2f}"
#             annotator.box_label(xyxy, label, color=yolo_colors(int(cls), bgr=True))
#     return annotator.result()


# # ── Inference pass for visualization ──────────────────────────────────────────

# @torch.inference_mode()
# def run_video_inference(
#     bundle:     ModelBundle,
#     raw_frames: list,
#     tensors:    list,
#     shapes:     list,
#     ratios:     list,
#     pads:       list,
# ) -> list[np.ndarray]:
#     """
#     Full annotated pass over all frames.
#     Not timed — use benchmark.benchmark_video() for timing metrics.
#     """
#     annotated = []
#     for frame, t, orig_shape in zip(raw_frames, tensors, shapes):
#         pred = _forward(bundle, t)
#         det  = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)[0]
#         ann  = annotate_frame(frame, det, bundle.names, orig_shape)
#         annotated.append(ann)
#     return annotated


# # ── Video output ──────────────────────────────────────────────────────────────

# def save_video(frames: list[np.ndarray], out_path: str, fps: float) -> str:
#     """Write annotated frames to mp4. Tries H.264, falls back to mp4v."""
#     if not frames:
#         return out_path
#     h, w = frames[0].shape[:2]
#     writer = None
#     for codec in ("avc1", "mp4v"):
#         fourcc = cv2.VideoWriter_fourcc(*codec)
#         writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
#         if writer.isOpened():
#             break
#     for f in frames:
#         writer.write(f)
#     writer.release()
#     return out_path

"""
Preprocessing, forward pass, NMS, annotation, and video I/O.

Preprocessing (letterbox + tensor creation) is done offline in load_video_frames
so that GPU tensors are pre-resident before benchmarking begins — the same
strategy as building gpu_batches in the ResNet50 benchmark.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from ultralytics.utils import nms, ops
from ultralytics.utils.plotting import Annotator, colors as yolo_colors

from config import DEVICE, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, MAX_DEMO_FRAMES
from model_loader import ModelBundle


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _letterbox(
    img_rgb: np.ndarray,
    new_shape: int = 640,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize + pad to a square while maintaining aspect ratio (grey padding).
    Returns: (padded_img, scale_ratio, (pad_w/2, pad_h/2))
    """
    h, w = img_rgb.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw = (new_shape - nw) / 2
    dh = (new_shape - nh) / 2

    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bot = int(round(dh - 0.1)), int(round(dh + 0.1))
    lft, rgt = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bot,
        lft,
        rgt,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, r, (dw, dh)


def preprocess_frame(
    frame_bgr: np.ndarray,
    img_size: int = IMG_SIZE,
) -> tuple[torch.Tensor, tuple[int, int], float, tuple[float, float]]:
    """BGR frame → normalized float32 [1, 3, H, W] tensor + metadata for rescaling."""
    h0, w0 = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded, r, p = _letterbox(rgb, img_size)
    t = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0), (h0, w0), r, p


def load_video_frames(
    video_path: str,
    img_size: int = IMG_SIZE,
    max_frames: int = MAX_DEMO_FRAMES,
) -> tuple[list, list, list, list, list, float, tuple[int, int]]:
    """
    Read all frames, preprocess them, move tensors to DEVICE (CUDA when available, else CPU).
    This is the video equivalent of building gpu_batches in the ResNet benchmark —
    all preprocessing happens here so it is excluded from timing.

    Returns:
        raw_bgr_frames, gpu_tensors, orig_shapes, ratios, pads, fps, (w, h)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_frames, tensors, shapes, ratios, pads = [], [], [], [], []

    while len(raw_frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t, orig_shape, ratio, pad = preprocess_frame(frame, img_size)
        raw_frames.append(frame.copy())
        tensors.append(
            t.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
        )  # pre-resident on GPU (CUDA) or CPU
        shapes.append(orig_shape)
        ratios.append(ratio)
        pads.append(pad)

    cap.release()
    return raw_frames, tensors, shapes, ratios, pads, fps, (orig_w, orig_h)


# ── Forward pass ──────────────────────────────────────────────────────────────

def _forward(bundle: ModelBundle, tensor: torch.Tensor) -> torch.Tensor:
    """
    Single forward pass with optional AMP.
    The YOLOv8 Detect head may return a tuple; use only prediction tensor for NMS.
    """
    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=bundle.use_amp,
    ):
        out = bundle.nn_model(tensor)

    return out[0] if isinstance(out, (list, tuple)) else out


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_frame(
    frame_bgr: np.ndarray,
    det: torch.Tensor,
    names: dict,
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """
    Draw detections on frame_bgr.
    det: [N, 6] tensor (x1, y1, x2, y2, conf, cls) in letterboxed-IMG_SIZE space.
    """
    annotator = Annotator(frame_bgr.copy())

    if det is not None and len(det):
        det_cpu = det.clone()
        det_cpu[:, :4] = ops.scale_boxes(
            (IMG_SIZE, IMG_SIZE), det_cpu[:, :4], orig_shape
        )

        for *xyxy, conf, cls in det_cpu.tolist():
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(
                xyxy,
                label,
                color=yolo_colors(int(cls), bgr=True),
            )

    return annotator.result()


# ── Inference pass for visualization ──────────────────────────────────────────

@torch.inference_mode()
def run_video_inference(
    bundle: ModelBundle,
    raw_frames: list,
    tensors: list,
    shapes: list,
    ratios: list,
    pads: list,
) -> list[np.ndarray]:
    """
    Full annotated pass over all frames.
    Not timed — use benchmark.benchmark_video() for timing metrics.
    """
    annotated = []

    for frame, t, orig_shape in zip(raw_frames, tensors, shapes):
        pred = _forward(bundle, t)
        det = nms.non_max_suppression(
            pred,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
        )[0]

        ann = annotate_frame(frame, det, bundle.names, orig_shape)
        annotated.append(ann)

    return annotated


def _opencv_safe_text(text: str) -> str:
    """cv2.putText only renders Hershey ASCII reliably (no em-dash, middle dot, etc.)."""
    text = text.replace("\u2014", "-").replace("\u2013", "-").replace("\u00b7", ",")
    return text.encode("ascii", errors="replace").decode("ascii")


def _overlay_text_bottom_left(frame_bgr: np.ndarray, lines: str | list[str]) -> np.ndarray:
    """Draw one or more HUD lines at bottom-left (Gradio chrome stays at top)."""
    if isinstance(lines, str):
        lines = [lines]
    out = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.72, 2
    line_gap = 6
    y = out.shape[0] - 14

    for text in reversed([_opencv_safe_text(ln) for ln in lines]):
        (_, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y -= baseline
        cv2.putText(out, text, (12, y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y -= th + line_gap

    return out


@torch.inference_mode()
def run_video_inference_with_fps_overlay(
    bundle: ModelBundle,
    raw_frames: list,
    tensors: list,
    shapes: list,
    ratios: list,
    pads: list,
    *,
    overlay_label: str,
    bench_fps: float,
    bench_caption: str = "batch 64, forward only",
) -> list[np.ndarray]:
    """Annotated pass with batched benchmark FPS on each frame (matches JSON below video)."""
    hud = f"{overlay_label} | bench: {bench_fps:.0f} FPS ({bench_caption})"
    annotated: list[np.ndarray] = []

    for frame, t, orig_shape in zip(raw_frames, tensors, shapes):
        pred = _forward(bundle, t)
        det = nms.non_max_suppression(
            pred,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
        )[0]
        ann = annotate_frame(frame, det, bundle.names, orig_shape)
        annotated.append(_overlay_text_bottom_left(ann, hud))

    return annotated


# ── Video output ──────────────────────────────────────────────────────────────

def save_video(frames: list[np.ndarray], out_path: str, fps: float) -> str:
    """Write annotated frames to mp4. Tries H.264, falls back to mp4v."""
    if not frames:
        return out_path

    h, w = frames[0].shape[:2]
    writer = None

    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break

    if writer is None or not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter with available codecs")

    for f in frames:
        writer.write(f)

    writer.release()
    return out_path
