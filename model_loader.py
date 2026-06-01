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
from config import DEVICE, IMG_SIZE, MODEL_WEIGHT, PRUNE_RATIO

# Registry — add new modes here without touching any other file
# Use max-autotune-no-cudagraphs (not reduce-overhead): YOLOv8 Detect mutates
# buffers during forward; CUDA graphs from reduce-overhead break at runtime.
OPTIMIZATION_MODES: dict[str, dict] = {
    "eager": {
        "use_amp":      False,
        "compile_mode": None,
        "label":        "Eager",
        "overlay_label": "Eager",
    },
    "torch_compile": {
        "use_amp":      False,
        "compile_mode": "max-autotune-no-cudagraphs",
        "label":        "torch.compile()",
        "overlay_label": "torch.compile()",
    },
    "amp": {
        "use_amp":      True,
        "compile_mode": None,
        "label":        "AMP",
        "overlay_label": "AMP",
    },
    "amp_compile": {
        "use_amp":      True,
        "compile_mode": "max-autotune-no-cudagraphs",
        "label":        "AMP + torch.compile()",
        "overlay_label": "AMP + torch.compile()",
        "ui_default":   True,
    },
    "structured_prune": {
        "use_amp":      False,
        "compile_mode": None,
        "label":        "Structured pruning",
        "overlay_label": "Structured pruning",
    },
    "unstructured_prune": {
        "use_amp":      False,
        "compile_mode": None,
        "label":        "Unstructured pruning",
        "overlay_label": "Unstructured pruning",
    },
}

DEFAULT_COMPARE_MODE = "amp_compile"


@dataclass
class ModelBundle:
    nn_model: nn.Module
    mode:     str
    names:    dict
    use_amp:  bool
    stride:   int
    weight:   str


def load_model(mode: str = "eager", weight: str | None = None) -> ModelBundle:
    """
    Load YOLOv8, apply the requested optimization, return a ModelBundle.
    torch.compile is called here so compile time is never counted in benchmarks.
    """
    if mode not in OPTIMIZATION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(OPTIMIZATION_MODES)}")

    cfg = OPTIMIZATION_MODES[mode]
    weight = weight or MODEL_WEIGHT

    yolo     = YOLO(weight)
    nn_model = yolo.model.to(DEVICE).eval()
    names    = yolo.names
    stride   = int(yolo.model.stride.max())

    def _param_count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    if mode in ("unstructured_prune", "structured_prune"):
        n_params_before = _param_count(nn_model)
        print(f"[INFO] Parameter count before prune: {n_params_before:,}")

    if mode == "unstructured_prune":
        from pruning import apply_unstructured_l1

        print(
            "[NOTE] Unstructured prune (no finetune in demo). "
            "Parameter count stays same; weights are sparsified in place."
        )
        apply_unstructured_l1(nn_model, PRUNE_RATIO)
        n_params_after = _param_count(nn_model)
        print(f"[INFO] Parameter count after prune:  {n_params_after:,}")
        if n_params_after == n_params_before:
            print(
                "[INFO] (Expected for unstructured: numel unchanged vs structured, "
                "where channels are removed.)"
            )
    elif mode == "structured_prune":
        from pruning import apply_structured_magnitude

        print("[NOTE] Structured prune (no finetune in demo); requires `pip install torch-pruning`.")
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=torch.float32)
        apply_structured_magnitude(nn_model, example, PRUNE_RATIO)
        n_params_after = _param_count(nn_model)
        print(f"[INFO] Parameter count after prune:  {n_params_after:,}")
        if n_params_after < n_params_before:
            print(f"[INFO] Structured pruning removed {n_params_before - n_params_after:,} parameters.")
        elif n_params_after == n_params_before:
            print(
                "[WARN] Parameter count unchanged after structured prune — "
                "ratio may be too low for this graph or heads absorbed all protection."
            )

    # Global speed flags — mirrors ResNet benchmark setup
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if cfg["compile_mode"] is not None:
        if DEVICE.type != "cuda":
            print(f"[WARN] torch.compile requires CUDA — running eager for mode={mode}")
        else:
            cm = cfg["compile_mode"]
            print(
                f"[INFO] Compiling model (torch.compile mode={cm!r}) — "
                "first run after load can take several minutes …"
            )
            try:
                nn_model = torch.compile(nn_model, mode=cm)
            except Exception as e:
                print(
                    f"[WARN] mode={cm!r} failed ({e}); "
                    "falling back to mode='default' (upgrade PyTorch for no-cudagraphs autotune)."
                )
                nn_model = torch.compile(nn_model, mode="default")

    # AMP disabled on CPU; float16 path is CUDA-only here
    use_amp = cfg["use_amp"] and (DEVICE.type == "cuda")

    return ModelBundle(
        nn_model=nn_model,
        mode=mode,
        names=names,
        use_amp=use_amp,
        stride=stride,
        weight=weight,
    )
