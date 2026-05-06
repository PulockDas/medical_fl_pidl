"""
Robustness module — attacks, defenses, and configuration.

Disabled by default; enabled via FL_RUN_OVERRIDE in the notebook.
"""

from robustness.robustness_config import RobustnessConfig
from robustness.attacks import wrap_dataset_with_attack
from robustness.defenses import clip_model_update, compute_update_norm

__all__ = [
    "RobustnessConfig",
    "wrap_dataset_with_attack",
    "clip_model_update",
    "compute_update_norm",
]
