"""
load_model(mode) applies optimization modes to YOLOv8's bare nn.Module.
Compile latency is intentionally excluded here — matches the ResNet50
benchmark philosophy where model loading was never part of the timed loop.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from ultralytics import YOLO
from config import DEVICE, MODEL_WEIGHT

# Registry — add new modes here without touching any other file
OPTIMIZATION_MODES: dict[str, dict] = {
    "eager": {
        "use_amp":      False,
        "compile_mode": None,
        "label":        "Eager Baseline (FP32, no compile)",
    },
    "compile_reduce_overhead": {
        "use_amp":      False,
        "compile_mode": "reduce-overhead",
        "label":        "torch.compile — reduce-overhead",
    },
    "compile_max_autotune": {
        "use_amp":      False,
        "compile_mode": "max-autotune",
        "label":        "torch.compile — max-autotune  ⚠ slow first run",
    },
    "amp": {
        "use_amp":      True,
        "compile_mode": None,
        "label":        "AMP only (FP16 mixed precision)",
    },
    "amp_compile_reduce_overhead": {
        "use_amp":      True,
        "compile_mode": "reduce-overhead",
        "label":        "AMP + torch.compile — reduce-overhead",
    },
}


@dataclass
class ModelBundle:
    nn_model: nn.Module
    mode:     str
    names:    dict
    use_amp:  bool
    stride:   int


def load_model(mode: str = "eager") -> ModelBundle:
    """
    Load YOLOv8, apply the requested optimization, return a ModelBundle.
    torch.compile is called here so compile time is never counted in benchmarks.
    """
    if mode not in OPTIMIZATION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(OPTIMIZATION_MODES)}")

    cfg = OPTIMIZATION_MODES[mode]

    yolo     = YOLO(MODEL_WEIGHT)
    nn_model = yolo.model.to(DEVICE).eval()
    names    = yolo.names
    stride   = int(yolo.model.stride.max())

    # Global speed flags — mirrors ResNet benchmark setup
    torch.backends.cudnn.benchmark = True
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if cfg["compile_mode"] is not None:
        if DEVICE.type != "cuda":
            print(f"[WARN] torch.compile requires CUDA — running eager for mode={mode}")
        else:
            print(
                f"[INFO] Compiling model (mode='{cfg['compile_mode']}') — "
                f"this may take several minutes for max-autotune ..."
            )
            nn_model = torch.compile(nn_model, mode=cfg["compile_mode"])

    # AMP disabled on CPU: float16 unsupported; keeps parity with GPU-only ResNet experiments
    use_amp = cfg["use_amp"] and (DEVICE.type == "cuda")

    return ModelBundle(
        nn_model=nn_model,
        mode=mode,
        names=names,
        use_amp=use_amp,
        stride=stride,
    )
