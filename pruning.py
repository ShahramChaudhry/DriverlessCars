"""
Pruning utilities for the bare YOLOv8 nn.Module (mirrors ResNet50 scripts:
unstructured L1 prune + structured magnitude prune via torch-pruning).

Post-prune ImageNet finetune is not run here (demo has no training loader);
see project scripts / your ResNet notebooks for finetune loops.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def apply_unstructured_l1(model: nn.Module, amount: float) -> None:
    """L1 magnitude unstructured prune on every Conv2d / Linear weight, then remove reparam mask."""
    import torch.nn.utils.prune as prune

    count = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")
            count += 1
    print(f"[INFO] Unstructured L1 prune amount={amount} applied to {count} Conv2d/Linear layers.")


def apply_structured_magnitude(
    model: nn.Module,
    example_inputs: torch.Tensor,
    pruning_ratio: float,
) -> None:
    """
    Structured magnitude pruning (torch-pruning), same family as ResNet MagnitudePruner.
    Detection heads are ignored so class/box tensors stay consistent.
    """
    try:
        import torch_pruning as tp
    except ImportError as e:
        raise ImportError(
            "Structured pruning requires `torch-pruning`. Install: pip install torch-pruning"
        ) from e

    head_names = frozenset(
        ("Detect", "Segment", "Pose", "OBB", "Classify", "WorldDetect", "v10Detect")
    )
    ignored: list[nn.Module] = []
    for m in model.modules():
        if type(m).__name__ in head_names:
            ignored.append(m)

    if not ignored:
        print(
            "[WARN] No YOLO head module matched for ignore list — "
            "structured prune may break the graph; aborting."
        )
        raise RuntimeError("structured_prune: could not find Detect (or sibling) head to ignore")

    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored,
    )
    pruner.step()
    print(
        f"[INFO] Structured magnitude prune ratio={pruning_ratio} "
        f"(ignored {len(ignored)} head module(s))."
    )
