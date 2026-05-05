"""
ResNet18 with intermediate feature-map exposure for PIDL regularization.

Architecture overview
---------------------
The standard ResNet18 backbone is decomposed into five named stages so that
every group's spatial output tensor is available in a single forward pass:

    Input  (B,   3, 224, 224)
    Stem   (B,  64,  56,  56)   conv7×7 → BN → ReLU → MaxPool
    layer1 (B,  64,  56,  56)   2 × BasicBlock, stride 1
    layer2 (B, 128,  28,  28)   2 × BasicBlock, stride 2
    layer3 (B, 256,  14,  14)   2 × BasicBlock, stride 2  ← default PIDL layer
    layer4 (B, 512,   7,   7)   2 × BasicBlock, stride 2
    Head   (B, num_classes)     GlobalAvgPool → Dropout → Linear

Why expose all four layers?
----------------------------
Each layer's output is a spatial feature map F ∈ ℝ^(B×C×H×W).
The Perona-Malik PIDL regularizer treats F as a field defined on the (H×W)
spatial grid and computes the anisotropic diffusion PDE residual:

    L_PIDL = mean( ‖ div( c(|∇F|) · ∇F ) ‖² )

Exposing all four layers lets the researcher choose the trade-off:

    layer1 — 56×56 grid, 64 channels   : sharp spatial detail, low semantics
    layer2 — 28×28 grid, 128 channels  : local patterns
    layer3 — 14×14 grid, 256 channels  : semantic regions  ← best default
    layer4 —  7×7  grid, 512 channels  : high-level semantics, coarse grid

The active PIDL layer is selected via ModelConfig.pidl_feature_layer and
accessed in task.py as feature_maps[model.pidl_feature_layer].

Forward API
-----------
    logits, feature_maps = model(x, return_features=True)   # training
    logits               = model(x, return_features=False)  # inference / eval
"""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn as nn
import torchvision.models as tv_models

from configs.experiment_config import ModelConfig


# ---------------------------------------------------------------------------
# Channel dimensions for each ResNet18 stage
# ---------------------------------------------------------------------------

LAYER_CHANNELS: dict[str, int] = {
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}

# Spatial output size (H = W) for 224×224 input
LAYER_SPATIAL: dict[str, int] = {
    "layer1": 56,
    "layer2": 28,
    "layer3": 14,
    "layer4": 7,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResNetPIDL(nn.Module):
    """ResNet18 classifier with full intermediate feature-map exposure.

    All four residual block groups (layer1–layer4) are captured during every
    forward pass when ``return_features=True``, enabling the caller to pick
    any layer for the grid-wise Perona-Malik PIDL regularization.

    Args:
        num_classes: Number of output classes for the classification head.
        config:      ``ModelConfig`` controlling pretrained weights, dropout,
                     frozen backbone, and the active PIDL feature layer.
    """

    def __init__(self, num_classes: int, config: ModelConfig) -> None:
        super().__init__()

        if config.backbone != "resnet18":
            raise NotImplementedError(
                f"Only 'resnet18' is currently supported; got '{config.backbone}'."
            )
        if config.pidl_feature_layer not in LAYER_CHANNELS:
            raise ValueError(
                f"pidl_feature_layer must be one of {list(LAYER_CHANNELS)}; "
                f"got '{config.pidl_feature_layer}'."
            )

        self.config = config
        self.num_classes = num_classes

        # Which layer's feature map is passed to the PIDL loss by default.
        # task.py reads this attribute via model.pidl_feature_layer.
        self.pidl_feature_layer: str = config.pidl_feature_layer

        # ------------------------------------------------------------------
        # Backbone — decomposed so every stage is a named attribute
        # ------------------------------------------------------------------
        weights = tv_models.ResNet18_Weights.DEFAULT if config.pretrained else None
        _backbone = tv_models.resnet18(weights=weights)

        # ── Stem ───────────────────────────────────────────────────────────
        # conv7×7 (stride 2) → BN → ReLU → MaxPool (stride 2)
        # Input:  (B, 3,  224, 224)
        # Output: (B, 64,  56,  56)
        self.conv1   = _backbone.conv1
        self.bn1     = _backbone.bn1
        self.relu    = _backbone.relu
        self.maxpool = _backbone.maxpool

        # ── Residual block groups ──────────────────────────────────────────
        # Each group applies two BasicBlocks. layer2–4 downsample with stride 2.
        self.layer1 = _backbone.layer1   # (B,  64, 56, 56)
        self.layer2 = _backbone.layer2   # (B, 128, 28, 28)
        self.layer3 = _backbone.layer3   # (B, 256, 14, 14)
        self.layer4 = _backbone.layer4   # (B, 512,  7,  7)

        # ── Pooling ────────────────────────────────────────────────────────
        # Global average pool: (B, 512, 7, 7) → (B, 512, 1, 1)
        self.avgpool = _backbone.avgpool

        # Optionally freeze backbone (train head only for a warm-up phase)
        if config.freeze_backbone:
            for param in self._backbone_parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Classification head  —  replaces ResNet's original fc layer
        # ------------------------------------------------------------------
        # Dropout → Linear(512, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(512, num_classes),
        )
        nn.init.kaiming_normal_(self.classifier[1].weight, nonlinearity="relu")
        nn.init.zeros_(self.classifier[1].bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @overload
    def forward(
        self,
        x: torch.Tensor,
        return_features: Literal[True],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        return_features: Literal[False],
    ) -> torch.Tensor: ...

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        """Run a full forward pass through the network.

        Args:
            x:               ``(B, 3, H, W)`` input image batch. Expects
                             ImageNet-normalised tensors for pretrained weights.
            return_features: If ``True`` (default), also return all four
                             intermediate feature maps as a dict.
                             Set to ``False`` during evaluation to skip the
                             dict allocation and save a small amount of memory.

        Returns:
            ``return_features=True``  →  ``(logits, feature_maps)``

              - ``logits``       : ``(B, num_classes)`` raw class scores.
              - ``feature_maps`` : ``dict[str, Tensor]`` with keys
                ``"layer1"``, ``"layer2"``, ``"layer3"``, ``"layer4"``.
                The tensor at ``feature_maps[model.pidl_feature_layer]``
                is the one consumed by ``CEWithPIDLLoss``.

            ``return_features=False``  →  ``logits`` only (Tensor).
        """
        # ── Stem ───────────────────────────────────────────────────────────
        x = self.conv1(x)     # 7×7 conv, stride 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # 3×3 max pool, stride 2
        # x: (B, 64, 56, 56)

        # ── Residual block group 1 ─────────────────────────────────────────
        # 2 × BasicBlock, no downsampling (stride 1).
        # Captures low-level spatial features: edges, textures, gradients.
        # Grid for PIDL: 56×56 spatial positions.
        feat1 = self.layer1(x)
        # feat1: (B, 64, 56, 56)

        # ── Residual block group 2 ─────────────────────────────────────────
        # 2 × BasicBlock, stride 2 → spatial dims halved.
        # Captures local patterns and simple part-level structures.
        # Grid for PIDL: 28×28 spatial positions.
        feat2 = self.layer2(feat1)
        # feat2: (B, 128, 28, 28)

        # ── Residual block group 3 ─────────────────────────────────────────
        # 2 × BasicBlock, stride 2 → spatial dims halved again.
        # Captures semantic regions with meaningful spatial extent.
        # DEFAULT PIDL layer: 14×14 grid balances semantics and resolution.
        # The Perona-Malik diffusion on a 14×14 grid is computationally
        # cheap and preserves medically relevant boundaries (tumor borders,
        # lesion edges, tissue interfaces).
        feat3 = self.layer3(feat2)
        # feat3: (B, 256, 14, 14)

        # ── Residual block group 4 ─────────────────────────────────────────
        # 2 × BasicBlock, stride 2 → 7×7 spatial grid.
        # Captures high-level class-discriminative semantics.
        # PIDL on 7×7 is coarse but enforces global structure consistency.
        feat4 = self.layer4(feat3)
        # feat4: (B, 512, 7, 7)

        # ── Classification head ────────────────────────────────────────────
        pooled = self.avgpool(feat4)          # (B, 512, 1, 1)
        flat   = torch.flatten(pooled, 1)     # (B, 512)
        logits = self.classifier(flat)        # (B, num_classes)

        # ── Return ─────────────────────────────────────────────────────────
        if not return_features:
            return logits

        # All four intermediate feature maps are returned so the caller can
        # select any layer for PIDL regularization via model.pidl_feature_layer.
        # Only the layer selected in ModelConfig.pidl_feature_layer is passed
        # to CEWithPIDLLoss; the others are available for analysis / ablation.
        feature_maps: dict[str, torch.Tensor] = {
            "layer1": feat1,
            "layer2": feat2,
            "layer3": feat3,
            "layer4": feat4,
        }
        return logits, feature_maps

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _backbone_parameters(self):
        """Yield all backbone (non-classifier) parameters."""
        for module in (self.conv1, self.bn1, self.layer1,
                       self.layer2, self.layer3, self.layer4):
            yield from module.parameters()

    def unfreeze_backbone(self) -> None:
        """Enable gradient updates for all backbone parameters.

        Useful for gradual fine-tuning: freeze backbone for the first few
        FL rounds, then call ``unfreeze_backbone()`` for full fine-tuning.
        """
        for param in self._backbone_parameters():
            param.requires_grad = True

    @property
    def pidl_feature_channels(self) -> int:
        """Channel count of the active PIDL feature layer."""
        return LAYER_CHANNELS[self.pidl_feature_layer]

    @property
    def pidl_feature_spatial(self) -> int:
        """Spatial side-length (H = W) of the active PIDL feature layer
        for a 224×224 input image."""
        return LAYER_SPATIAL[self.pidl_feature_layer]

    def count_parameters(self) -> dict[str, int]:
        """Return trainable and total parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "total": total}

    def layer_info(self) -> str:
        """Return a formatted table of layer output shapes (for 224×224 input)."""
        lines = ["Layer    Channels  Spatial  Grid points"]
        lines.append("─" * 40)
        for name, ch in LAYER_CHANNELS.items():
            sp = LAYER_SPATIAL[name]
            active = " ← PIDL" if name == self.pidl_feature_layer else ""
            lines.append(f"{name:<8} {ch:>7}  {sp:>4}×{sp:<4}  {sp*sp:>6}{active}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        counts = self.count_parameters()
        return (
            f"ResNetPIDL(\n"
            f"  backbone   = {self.config.backbone}  "
            f"(pretrained={self.config.pretrained}, "
            f"frozen={self.config.freeze_backbone})\n"
            f"  num_classes = {self.num_classes}\n"
            f"  pidl_layer  = {self.pidl_feature_layer}  "
            f"({self.pidl_feature_channels} ch, "
            f"{self.pidl_feature_spatial}×{self.pidl_feature_spatial} grid)\n"
            f"  parameters  = {counts['trainable']:,} trainable / "
            f"{counts['total']:,} total\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Factory and Flower parameter helpers
# ---------------------------------------------------------------------------


def build_model(num_classes: int, config: ModelConfig) -> ResNetPIDL:
    """Instantiate a ``ResNetPIDL`` from a ``ModelConfig``.

    Args:
        num_classes: Number of target classes (from ``DatasetConfig``).
        config:      ``ModelConfig`` with backbone and PIDL settings.

    Returns:
        A ``ResNetPIDL`` model ready for training.
    """
    return ResNetPIDL(num_classes=num_classes, config=config)


def get_model_parameters(model: ResNetPIDL) -> list:
    """Extract all parameters as a list of NumPy arrays (Flower format).

    Args:
        model: ``ResNetPIDL`` instance.

    Returns:
        ``[np.ndarray, ...]`` in ``state_dict`` key order.
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model: ResNetPIDL, parameters: list) -> ResNetPIDL:
    """Load a Flower parameter list into a model in-place.

    Args:
        model:      The ``ResNetPIDL`` to update.
        parameters: List of NumPy arrays matching ``model.state_dict()`` order.

    Returns:
        The same model (for chaining).
    """
    import numpy as np

    state_dict = model.state_dict()
    new_state = {
        k: torch.tensor(np.array(v))
        for k, v in zip(state_dict.keys(), parameters)
    }
    model.load_state_dict(new_state, strict=True)
    return model
