"""Explainability helpers: Grad-CAM and Perona-Malik grid maps."""

from explainability.gradcam import GradCAM, cam_statistics, upsample_cam
from explainability.pm_grid_explainer import (
    gradcam_pm_iou_top25,
    grid_statistics,
    normalize01,
    pm_grid_scores,
    upsample_grid_map,
)
from explainability.plot_utils import overlay_heatmap, savefig_tight, tensor_to_display_rgb

__all__ = [
    "GradCAM",
    "cam_statistics",
    "upsample_cam",
    "pm_grid_scores",
    "normalize01",
    "upsample_grid_map",
    "grid_statistics",
    "gradcam_pm_iou_top25",
    "tensor_to_display_rgb",
    "overlay_heatmap",
    "savefig_tight",
]
