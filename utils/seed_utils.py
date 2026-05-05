"""
Reproducibility utilities.

Call set_all_seeds(seed) at the start of any script or notebook cell
that involves randomness (data loading, model init, training).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Also sets PYTHONHASHSEED and enables PyTorch deterministic algorithms
    where possible.

    Args:
        seed: Integer seed. Each federated client should use seed + client_id
              to avoid identical initializations while remaining reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Request deterministic CUDA operations where available.
    # This can slightly slow down training on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
