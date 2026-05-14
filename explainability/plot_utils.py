"""Publication-style figures for explainability (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset_utils import IMAGENET_MEAN, IMAGENET_STD


def tensor_to_display_rgb(
    x_chw: torch.Tensor,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Denormalize ImageNet-normalized CHW tensor to HWC RGB [0,1]."""
    t = x_chw.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(3, 1, 1)
    t = t * std + mean
    t = t.clamp(0, 1)
    return t.permute(1, 2, 0).numpy()


def overlay_heatmap(
    rgb_hwc: np.ndarray, heatmap_hw: np.ndarray, cmap: str = "jet", alpha: float = 0.45
) -> np.ndarray:
    """Blend RGB (H,W,3) with heatmap (H,W) in [0,1]."""
    col = plt.get_cmap(cmap)(heatmap_hw)[..., :3]
    return (1 - alpha) * rgb_hwc + alpha * col.astype(np.float32)


def savefig_tight(path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
