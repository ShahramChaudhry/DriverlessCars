"""
Preprocessing, forward pass, NMS, annotation, and video I/O.

Preprocessing (letterbox + tensor creation) is done offline in load_video_frames
so that GPU tensors are pre-resident before benchmarking begins — the same
strategy as building gpu_batches in the ResNet50 benchmark.
"""
from __future__ import annotations
import tempfile
import cv2
import numpy as np
import torch
from ultralytics.utils.ops import non_max_suppression, scale_boxes
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
    h, w   = img_rgb.shape[:2]
    r      = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw     = (new_shape - nw) / 2
    dh     = (new_shape - nh) / 2
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bot = int(round(dh - 0.1)), int(round(dh + 0.1))
    lft, rgt = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bot, lft, rgt,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, r, (dw, dh)


def preprocess_frame(
    frame_bgr: np.ndarray,
    img_size: int = IMG_SIZE,
) -> tuple[torch.Tensor, tuple[int, int], float, tuple[float, float]]:
    """BGR frame → normalised float32 [1, 3, H, W] tensor + metadata for rescaling."""
    h0, w0        = frame_bgr.shape[:2]
    rgb           = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded, r, p  = _letterbox(rgb, img_size)
    t = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0), (h0, w0), r, p


def load_video_frames(
    video_path: str,
    img_size:   int = IMG_SIZE,
    max_frames: int = MAX_DEMO_FRAMES,
) -> tuple[list, list, list, list, list, float, tuple[int, int]]:
    """
    Read all frames, preprocess them, move tensors to GPU.
    This is the video equivalent of building gpu_batches in the ResNet benchmark —
    all preprocessing happens here so it is excluded from timing.

    Returns:
        raw_bgr_frames, gpu_tensors, orig_shapes, ratios, pads, fps, (w, h)
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_frames, tensors, shapes, ratios, pads = [], [], [], [], []

    while len(raw_frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t, orig_shape, ratio, pad = preprocess_frame(frame, img_size)
        raw_frames.append(frame.copy())
        tensors.append(t.to(DEVICE, non_blocking=True))   # pre-resident on GPU
        shapes.append(orig_shape)
        ratios.append(ratio)
        pads.append(pad)

    cap.release()
    return raw_frames, tensors, shapes, ratios, pads, fps, (orig_w, orig_h)


# ── Forward pass ──────────────────────────────────────────────────────────────

def _forward(bundle: ModelBundle, tensor: torch.Tensor) -> torch.Tensor:
    """
    Single forward pass with optional AMP.
    The YOLOv8 Detect head returns (pred_tensor, feature_maps) in eval mode;
    we take only pred_tensor and pass it to NMS.
    """
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=bundle.use_amp):
        out = bundle.nn_model(tensor)
    # Detect head: eval mode returns (predictions, raw_feature_list)
    return out[0] if isinstance(out, (list, tuple)) else out


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_frame(
    frame_bgr:  np.ndarray,
    det:        torch.Tensor,
    names:      dict,
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """
    Draw detections on frame_bgr.
    det: [N, 6] tensor (x1, y1, x2, y2, conf, cls) in letterboxed-640 space.
    """
    annotator = Annotator(frame_bgr.copy())
    if det is not None and len(det):
        det_cpu = det.clone()
        det_cpu[:, :4] = scale_boxes(
            (IMG_SIZE, IMG_SIZE), det_cpu[:, :4], orig_shape
        )
        for *xyxy, conf, cls in det_cpu.tolist():
            label = f"{names[int(cls)]}  {conf:.2f}"
            annotator.box_label(xyxy, label, color=yolo_colors(int(cls), bgr=True))
    return annotator.result()


# ── Inference pass for visualization ──────────────────────────────────────────

@torch.inference_mode()
def run_video_inference(
    bundle:     ModelBundle,
    raw_frames: list,
    tensors:    list,
    shapes:     list,
    ratios:     list,
    pads:       list,
) -> list[np.ndarray]:
    """
    Full annotated pass over all frames.
    Not timed — use benchmark.benchmark_video() for timing metrics.
    """
    annotated = []
    for frame, t, orig_shape in zip(raw_frames, tensors, shapes):
        pred = _forward(bundle, t)
        det  = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)[0]
        ann  = annotate_frame(frame, det, bundle.names, orig_shape)
        annotated.append(ann)
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
    for f in frames:
        writer.write(f)
    writer.release()
    return out_path
