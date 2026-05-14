"""Grad-CAM for ResNetPIDL (target: final residual stage / layer4)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Grad-CAM with hooks on a convolutional module (default: ``layer4``)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self, _module: nn.Module, _inp: tuple, out: torch.Tensor
    ) -> None:
        self._activations = out.detach()

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_in: tuple[torch.Tensor, ...],
        grad_out: tuple[torch.Tensor, ...],
    ) -> None:
        g = grad_out[0]
        if g is not None:
            self._gradients = g.detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def compute(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (cam_hw, logits, class_idx). cam_hw is (H', W') spatial map."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x, return_features=False)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=retain_graph)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        acts = self._activations[0]
        grads = self._gradients[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam, logits, class_idx


def upsample_cam(cam_hw: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """Upsample CAM to image size; return float32 numpy (H, W) in [0, 1]."""
    t = cam_hw.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return up.squeeze().cpu().numpy().astype(np.float32)


def cam_statistics(cam_hw: torch.Tensor) -> tuple[float, float]:
    m = cam_hw.detach().float()
    return float(m.mean().item()), float(m.max().item())
