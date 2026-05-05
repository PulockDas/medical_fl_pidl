"""
Grid-wise Spatial Perona-Malik PIDL Regularization Loss.

IMPORTANT — what this loss IS and IS NOT
-----------------------------------------
This is NOT a temporal/time-series loss.
There is no time dimension, no tumor growth model, and no sequence of images.

This IS a spatial feature regularizer for static medical images.
It treats each intermediate ResNet feature map F ∈ ℝ^(B×C×H×W) as a
2-D spatial field defined on the (H×W) pixel grid of one scan.

The Perona-Malik PDE is used purely as a spatial smoothness prior:
features in anatomically uniform regions should vary slowly, while
features at tissue/lesion boundaries should be allowed to be sharp.

Why grid-wise (not global)?
----------------------------
Tumor, lesion, and pathology regions are spatially LOCAL — they occupy
a small fraction of the image. A global PM loss is dominated by the
large background region and under-regularizes the diagnostically important
foreground patches. Dividing H×W into grid_size × grid_size non-overlapping
regions enforces spatial coherence at the scale of individual tissue patches,
giving equal weight to every part of the image regardless of local content.

Physics: Perona-Malik anisotropic diffusion (Perona & Malik, 1990)
-------------------------------------------------------------------
PDE:    ∂F/∂t = div( c(|∇F|) · ∇F )

Diffusivity (Lorentzian, default):   c(s) = 1 / (1 + (s/K)²)
At strong edges (s >> K): c → 0  →  no diffusion (edge preserved)
At flat regions (s << K): c → 1  →  full isotropic diffusion

Steady-state PIDL residual:  R(F) = div( c(|∇F|) · ∇F ) = 0
PIDL loss:                   L_PIDL = mean( ‖R(F)‖² )

Isotropic baseline (c = 1 everywhere):
PIDL loss becomes            L_iso  = mean( ‖∇F‖² )   (Tikhonov)

Numerical scheme
----------------
Gradients  : forward  finite differences on the (H×W) grid
Divergence : backward finite differences on the flux field
(adjoint pair → discrete operator is negative semi-definite)

Total loss
----------
total_loss = CrossEntropy(logits, labels) + lambda_pm × reg_loss

where reg_loss is the global or grid-wise PM / isotropic loss.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Spatial gradient computation
# ---------------------------------------------------------------------------


def compute_spatial_gradients(
    feature_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward finite-difference spatial gradients on the (H×W) grid.

    For interior grid points:
        grad_x[b, c, h, w] = F[b,c,h,w+1] − F[b,c,h,w]   (→ right)
        grad_y[b, c, h, w] = F[b,c,h+1,w] − F[b,c,h,w]   (↓ down)

    Boundary values are set to zero (Neumann / no-flux condition).
    This is the standard forward-difference approximation used in
    numerical diffusion PDE solvers.

    Args:
        feature_map: ``(B, C, H, W)`` intermediate ResNet feature map.

    Returns:
        ``(grad_x, grad_y)`` — each ``(B, C, H, W)``.
        ``grad_x`` is the horizontal (width) gradient.
        ``grad_y`` is the vertical (height) gradient.
    """
    grad_x = torch.zeros_like(feature_map)
    grad_y = torch.zeros_like(feature_map)

    # Width direction (last dimension)
    grad_x[..., :-1] = feature_map[..., 1:] - feature_map[..., :-1]

    # Height direction (second-to-last dimension)
    grad_y[..., :-1, :] = feature_map[..., 1:, :] - feature_map[..., :-1, :]

    return grad_x, grad_y


# ---------------------------------------------------------------------------
# 2. Internal helpers
# ---------------------------------------------------------------------------


def _backward_divergence(
    flux_x: torch.Tensor,
    flux_y: torch.Tensor,
) -> torch.Tensor:
    """Backward finite-difference divergence of a 2-D flux field.

    Adjoint of the forward gradient operator. Together with forward
    gradients, this ensures the discrete PM operator (−div·c·grad)
    is positive semi-definite.

    Args:
        flux_x: ``(B, C, H, W)`` flux in the width direction.
        flux_y: ``(B, C, H, W)`` flux in the height direction.

    Returns:
        ``(B, C, H, W)`` divergence tensor.
    """
    div_x = torch.zeros_like(flux_x)
    div_y = torch.zeros_like(flux_y)

    # Backward difference in x: div_x[w] = flux_x[w] − flux_x[w−1]
    div_x[..., 1:] = flux_x[..., 1:] - flux_x[..., :-1]
    div_x[..., 0]  = flux_x[..., 0]   # left-boundary term

    # Backward difference in y: div_y[h] = flux_y[h] − flux_y[h−1]
    div_y[..., 1:, :] = flux_y[..., 1:, :] - flux_y[..., :-1, :]
    div_y[..., 0, :]  = flux_y[..., 0, :]  # top-boundary term

    return div_x + div_y


def _normalize_per_sample(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Min-max normalize each batch element independently to [0, 1].

    Normalizing makes the edge-sensitivity threshold K scale-invariant:
    K=0.1 means "10% of the feature map's dynamic range" regardless of
    the absolute activation magnitude.

    Args:
        x:   ``(B, C, H, W)`` or ``(B', C, ph, pw)`` tensor.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized tensor of the same shape.
    """
    B = x.shape[0]
    flat = x.view(B, -1)
    xmin = flat.min(dim=1).values.view(B, 1, 1, 1)
    xmax = flat.max(dim=1).values.view(B, 1, 1, 1)
    return (x - xmin) / (xmax - xmin + eps)


def _extract_patches(
    feature_map: torch.Tensor,
    grid_size: int,
) -> torch.Tensor | None:
    """Split a feature map into a batch of non-overlapping spatial patches.

    Divides the (H×W) spatial grid into ``grid_size × grid_size`` tiles.
    If H or W is not evenly divisible by ``grid_size``, the feature map is
    **cropped** to the nearest smaller divisible size. The cropped border
    pixels are typically at most ``grid_size − 1`` pixels wide — negligible
    for typical feature map sizes (14×14 with grid_size=2 → no crop needed).

    Args:
        feature_map: ``(B, C, H, W)`` tensor.
        grid_size:   Number of tiles along each spatial axis.
                     Total tiles = grid_size².

    Returns:
        ``(B × grid_size², C, ph, pw)`` patch batch, where
        ``ph = H_crop // grid_size`` and ``pw = W_crop // grid_size``.
        Returns ``None`` if the feature map is too small (H or W < grid_size),
        in which case the caller should fall back to the global loss.
    """
    B, C, H, W = feature_map.shape

    # Crop to nearest multiple of grid_size
    H_crop = (H // grid_size) * grid_size
    W_crop = (W // grid_size) * grid_size

    if H_crop == 0 or W_crop == 0:
        # Feature map is smaller than grid_size in at least one dimension
        return None

    feat = feature_map[:, :, :H_crop, :W_crop]   # (B, C, H_crop, W_crop)

    ph = H_crop // grid_size   # patch height
    pw = W_crop // grid_size   # patch width

    # ── Tile extraction ────────────────────────────────────────────────────
    # Split H into grid_size strips of height ph, W into grid_size strips of pw.
    # view: (B, C, H_crop, W_crop) → (B, C, grid_size, ph, grid_size, pw)
    #   dim2 = tile row index,  dim3 = row within tile
    #   dim4 = tile col index,  dim5 = col within tile
    feat = feat.view(B, C, grid_size, ph, grid_size, pw)

    # permute: bring tile-row and tile-col indices together before channels
    # (B, C, gs_y, ph, gs_x, pw) → (B, gs_y, gs_x, C, ph, pw)
    feat = feat.permute(0, 2, 4, 1, 3, 5).contiguous()

    # Flatten: each patch becomes an independent "batch element"
    # (B, gs_y, gs_x, C, ph, pw) → (B × gs_y × gs_x, C, ph, pw)
    patches = feat.view(B * grid_size * grid_size, C, ph, pw)

    return patches


# ---------------------------------------------------------------------------
# 3. Global (full feature map) loss functions
# ---------------------------------------------------------------------------


def perona_malik_loss(
    feature_map: torch.Tensor,
    k: float = 0.1,
    normalize: bool = True,
) -> torch.Tensor:
    """Perona-Malik PIDL regularization loss over the full feature map.

    Enforces the Perona-Malik PDE steady-state condition as a soft constraint:

        R(F) = div( c(|∇F|) · ∇F ) = 0

    Loss:  L = mean( ‖R(F)‖² )

    Anisotropic diffusivity (Lorentzian):
        c(s) = 1 / (1 + (s/K)²)
    - Large gradients (edges): c → 0  →  boundary preserved
    - Small gradients (interior): c → 1  →  interior smoothed

    NOTE: This is purely spatial. It operates on the 2-D (H×W) grid of a
    single feature map slice. There is no temporal component.

    Args:
        feature_map: ``(B, C, H, W)`` intermediate ResNet feature map.
        k:           Edge-sensitivity threshold. Gradients larger than K
                     are treated as edges and not smoothed.
        normalize:   If True, normalize each batch element to [0, 1] first,
                     making K scale-invariant.

    Returns:
        Scalar PIDL loss.
    """
    F = _normalize_per_sample(feature_map) if normalize else feature_map

    grad_x, grad_y = compute_spatial_gradients(F)

    # Gradient magnitude at each grid point (ε for numerical stability)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # Lorentzian diffusivity: c(s) = 1 / (1 + (s/K)²)
    # Strong edges (grad_mag >> K): c ≈ 0  →  flux ≈ 0  →  edge preserved
    # Flat regions (grad_mag << K): c ≈ 1  →  full diffusion
    c = 1.0 / (1.0 + (grad_mag / (k + 1e-8)) ** 2)

    # Anisotropic flux field: weighted gradient
    flux_x = c * grad_x
    flux_y = c * grad_y

    # PDE residual: div(c · ∇F)
    # At steady state this should be zero everywhere
    residual = _backward_divergence(flux_x, flux_y)

    return (residual ** 2).mean()


def isotropic_diffusion_loss(
    feature_map: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Isotropic (uniform) diffusion regularizer.

    Special case of the Perona-Malik loss with constant diffusivity c = 1.
    Equivalent to Tikhonov L² gradient regularization:

        L = mean( |∇F|² ) = mean( grad_x² + grad_y² )

    Unlike Perona-Malik, this penalizes ALL gradients equally — both edges
    and flat regions. Use this as a simpler baseline to compare against the
    anisotropic PM variant.

    NOTE: This is purely spatial. No temporal component.

    Args:
        feature_map: ``(B, C, H, W)`` intermediate ResNet feature map.
        normalize:   If True, normalize each batch element to [0, 1] first.

    Returns:
        Scalar isotropic regularization loss.
    """
    F = _normalize_per_sample(feature_map) if normalize else feature_map

    grad_x, grad_y = compute_spatial_gradients(F)

    # Penalizes all gradients uniformly — no edge preservation
    return (grad_x ** 2 + grad_y ** 2).mean()


# ---------------------------------------------------------------------------
# 4. Grid-wise loss functions
# ---------------------------------------------------------------------------


def gridwise_perona_malik_loss(
    feature_map: torch.Tensor,
    grid_size: int = 4,
    k: float = 0.1,
    normalize: bool = True,
) -> torch.Tensor:
    """Grid-wise Perona-Malik PIDL loss.

    Divides the (H×W) feature map into ``grid_size × grid_size``
    non-overlapping spatial regions and computes the PM loss independently
    within each region. The final loss is the mean across all regions.

    Why grid-wise?
    ~~~~~~~~~~~~~~
    Tumor, lesion, and pathology regions are LOCAL. They occupy only a
    fraction of the full image. A global PM loss is dominated by the
    (typically much larger) background region, which under-regularizes
    the diagnostically important foreground patches.

    Grid-wise computation gives equal weight to every spatial region —
    whether it contains background, organ tissue, or a lesion — and
    enforces local spatial coherence at the scale of individual patches.

    Handling non-divisible H/W
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    If H or W is not divisible by ``grid_size``, the feature map is cropped
    to the nearest smaller divisible size before tiling. For typical ResNet
    feature map sizes (e.g. 14×14 for layer3 with 224×224 input) and
    common grid sizes (2, 4, 7), cropping is rarely needed.

    If the feature map is smaller than ``grid_size`` in any spatial
    dimension, falls back silently to the global :func:`perona_malik_loss`.

    Args:
        feature_map: ``(B, C, H, W)`` intermediate ResNet feature map.
        grid_size:   Number of tiles along each spatial axis.
                     Total regions = grid_size². E.g. grid_size=4 → 16 patches.
        k:           Perona-Malik edge-sensitivity threshold.
        normalize:   Per-patch normalization (recommended; keeps K scale-invariant).

    Returns:
        Scalar PM loss averaged over all grid_size² patches and all batches.
    """
    patches = _extract_patches(feature_map, grid_size)

    if patches is None:
        # Feature map too small for requested grid — fall back to global loss
        return perona_malik_loss(feature_map, k=k, normalize=normalize)

    # Apply PM loss over the merged (B × grid_size²) batch dimension.
    # Each patch is treated as an independent spatial field of size (ph × pw).
    # Normalizing per-patch makes K relative to each patch's dynamic range.
    return perona_malik_loss(patches, k=k, normalize=normalize)


def gridwise_isotropic_loss(
    feature_map: torch.Tensor,
    grid_size: int = 4,
    normalize: bool = True,
) -> torch.Tensor:
    """Grid-wise isotropic (uniform) diffusion loss.

    Same grid-division strategy as :func:`gridwise_perona_malik_loss` but
    applies the isotropic (c = 1) regularizer within each region.

    Use as a simpler baseline: no diffusivity weighting, all gradients
    penalized equally within every spatial patch.

    Args:
        feature_map: ``(B, C, H, W)`` intermediate ResNet feature map.
        grid_size:   Number of tiles along each spatial axis.
        normalize:   Per-patch normalization.

    Returns:
        Scalar isotropic loss averaged over all patches and batches.
    """
    patches = _extract_patches(feature_map, grid_size)

    if patches is None:
        return isotropic_diffusion_loss(feature_map, normalize=normalize)

    return isotropic_diffusion_loss(patches, normalize=normalize)


# ---------------------------------------------------------------------------
# 5. PIDLLoss — unified nn.Module
# ---------------------------------------------------------------------------


class PIDLLoss(nn.Module):
    """Combined CrossEntropy + spatial PIDL regularization loss.

    Computes:
        total_loss = CE(logits, labels) + lambda_pm × reg_loss

    where reg_loss is selected by ``regularizer_type``:

        "perona_malik"  — anisotropic PM regularizer (edge-preserving)
        "isotropic"     — uniform gradient penalty (Tikhonov baseline)
        "none"          — pure CrossEntropy, no regularization

    When ``use_grid_loss=True`` (recommended), the feature map is divided
    into ``grid_size × grid_size`` local patches and the regularizer is
    applied independently within each patch before averaging.

    This is a SPATIAL regularizer for static images only.
    There is no temporal dimension, no time stepping, and no
    tumor-growth or disease-progression model.

    Args:
        regularizer_type: One of ``"perona_malik"``, ``"isotropic"``, ``"none"``.
        lambda_pm:        Weight of the regularization term. Set to 0 or use
                          ``regularizer_type="none"`` to disable.
        k:                Perona-Malik edge-sensitivity threshold K. Gradients
                          larger than K are treated as edges and not smoothed.
                          Ignored for isotropic mode.
        use_grid_loss:    If True (recommended), apply the regularizer on local
                          grid patches rather than the full feature map.
        grid_size:        Number of spatial tiles along each axis.
                          Total patches = grid_size². E.g. grid_size=4 → 16.
    """

    VALID_TYPES = ("perona_malik", "isotropic", "none")

    def __init__(
        self,
        regularizer_type: str = "perona_malik",
        lambda_pm: float = 0.01,
        k: float = 0.1,
        use_grid_loss: bool = True,
        grid_size: int = 4,
    ) -> None:
        super().__init__()

        if regularizer_type not in self.VALID_TYPES:
            raise ValueError(
                f"regularizer_type must be one of {self.VALID_TYPES}; "
                f"got '{regularizer_type}'."
            )
        if grid_size < 1:
            raise ValueError(f"grid_size must be ≥ 1; got {grid_size}.")

        self.regularizer_type = regularizer_type
        self.lambda_pm = lambda_pm
        self.k = k
        self.use_grid_loss = use_grid_loss
        self.grid_size = grid_size

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total, cross-entropy, and regularization losses.

        Args:
            logits:      ``(B, num_classes)`` raw model output.
            labels:      ``(B,)`` integer ground-truth class indices.
            feature_map: ``(B, C, H, W)`` intermediate ResNet feature map.
                         Must be on the same device as logits and labels.
                         Typically taken from the layer selected by
                         ``ModelConfig.pidl_feature_layer``.

        Returns:
            ``(total_loss, ce_loss, reg_loss)`` — three scalar tensors.

              - ``total_loss``  = ``ce_loss + lambda_pm × reg_loss``
              - ``ce_loss``     — standard cross-entropy on logits
              - ``reg_loss``    — PIDL regularization term (or 0 if disabled)
        """
        ce_loss = self.ce(logits, labels)

        # ── Regularization ─────────────────────────────────────────────────
        if self.regularizer_type == "none" or self.lambda_pm == 0.0:
            # Regularization disabled: reg_loss is a zero scalar on the
            # correct device so it can still be logged without errors.
            reg_loss = torch.zeros(1, device=logits.device).squeeze()

        elif self.regularizer_type == "perona_malik":
            if self.use_grid_loss:
                reg_loss = gridwise_perona_malik_loss(
                    feature_map, grid_size=self.grid_size, k=self.k
                )
            else:
                reg_loss = perona_malik_loss(feature_map, k=self.k)

        else:  # "isotropic"
            if self.use_grid_loss:
                reg_loss = gridwise_isotropic_loss(
                    feature_map, grid_size=self.grid_size
                )
            else:
                reg_loss = isotropic_diffusion_loss(feature_map)

        total_loss = ce_loss + self.lambda_pm * reg_loss
        return total_loss, ce_loss, reg_loss

    def extra_repr(self) -> str:
        return (
            f"regularizer_type={self.regularizer_type!r}, "
            f"lambda_pm={self.lambda_pm}, "
            f"k={self.k}, "
            f"use_grid_loss={self.use_grid_loss}, "
            f"grid_size={self.grid_size}"
        )


# ---------------------------------------------------------------------------
# 6. Backward-compatible alias
# ---------------------------------------------------------------------------


class CEWithPIDLLoss(nn.Module):
    """Backward-compatible wrapper around PIDLLoss.

    Maps the original constructor arguments (lambda_pidl, K, diffusivity_type,
    normalize, class_weights) to the new PIDLLoss interface so that existing
    code in federated/client_app.py continues to work without modification.

    New code should use PIDLLoss directly.
    """

    def __init__(
        self,
        lambda_pidl: float = 0.01,
        K: float = 0.1,
        diffusivity_type: str = "lorentzian",
        normalize: bool = True,
        class_weights: torch.Tensor | None = None,
        use_grid_loss: bool = True,
        grid_size: int = 4,
    ) -> None:
        super().__init__()
        self._loss = PIDLLoss(
            regularizer_type="perona_malik" if lambda_pidl > 0 else "none",
            lambda_pm=lambda_pidl,
            k=K,
            use_grid_loss=use_grid_loss,
            grid_size=grid_size,
        )
        if class_weights is not None:
            self._loss.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._loss(logits, labels, feature_map)
