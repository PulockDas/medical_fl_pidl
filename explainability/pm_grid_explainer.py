"""Physics-guided Perona-Malik grid importance maps (matches training PIDL grid)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from losses.pidl_loss import gridwise_pm_cell_scores


def pm_grid_scores(
    feature_map: torch.Tensor,
    grid_size: int,
    k: float,
    normalize: bool = True,
) -> torch.Tensor:
    """``(B, grid_size, grid_size)`` PM squared-residual energy per cell."""
    return gridwise_pm_cell_scores(
        feature_map, grid_size=grid_size, k=k, normalize=normalize
    )


def normalize01(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min–max per batch item for maps of shape (B, H, W) or (B, gs, gs)."""
    B = t.shape[0]
    flat = t.view(B, -1)
    lo = flat.min(dim=1).values.view(B, 1, 1)
    hi = flat.max(dim=1).values.view(B, 1, 1)
    return (t - lo) / (hi - lo + eps)


def upsample_grid_map(
    grid_scores: torch.Tensor, out_h: int, out_w: int
) -> np.ndarray:
    """``grid_scores`` (B, gs, gs) -> numpy (H, W) [0,1] for first item."""
    g = grid_scores[0:1].unsqueeze(1).float()
    up = F.interpolate(g, size=(out_h, out_w), mode="bilinear", align_corners=False)
    u = up.squeeze()
    u = (u - u.min()) / (u.max() - u.min() + 1e-8)
    return u.cpu().numpy().astype(np.float32)


def grid_statistics(grid_scores: torch.Tensor) -> tuple[float, float, int]:
    """Mean, max over cells, and flat argmax index for first image."""
    g = grid_scores[0].flatten()
    idx = int(torch.argmax(g).item())
    return float(g.mean().item()), float(g.max().item()), idx


def gradcam_pm_iou_top25(cam_hw: np.ndarray, pm_hw: np.ndarray) -> float:
    """Intersection / union of top-25% foreground binary masks."""
    c = cam_hw.astype(np.float64).flatten()
    p = pm_hw.astype(np.float64).flatten()
    tc = np.quantile(c, 0.75)
    tp = np.quantile(p, 0.75)
    bc = c >= tc
    bp = p >= tp
    inter = np.logical_and(bc, bp).sum()
    union = np.logical_or(bc, bp).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)
