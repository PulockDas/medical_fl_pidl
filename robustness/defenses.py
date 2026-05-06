"""
Server-side and client-side defenses for federated learning robustness.

Update clipping
---------------
Before a client sends its updated parameters back to the server, the
*parameter delta* (new_params − old_params) is projected onto the L2 ball
of radius ``clip_norm``.  This limits the maximum influence any single client
can exert on the global model — a standard defense against both Byzantine
attacks and gradient inversion.

The clipping is applied **on the client side** inside ``MedicalFLClient.fit``
so that no information about unclipped updates reaches the server.

Usage in client_app.py::

    from robustness.defenses import clip_model_update
    new_params = clip_model_update(old_params, new_params, clip_norm=3.0)

The clipped ``new_params`` list is then returned in the usual
``(params, num_examples, metrics)`` tuple from ``fit``.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Update-clipping defense
# ---------------------------------------------------------------------------


def clip_model_update(
    old_params: List[np.ndarray],
    new_params: List[np.ndarray],
    clip_norm: float = 3.0,
) -> List[np.ndarray]:
    """Clip the L2 norm of (new_params − old_params) to at most *clip_norm*.

    If the update's global L2 norm is already ≤ clip_norm the parameters are
    returned unchanged (no rescaling).

    Algorithm
    ---------
    1. Compute delta_i = new_i − old_i for every layer i.
    2. Compute ||delta|| = sqrt(Σ ||delta_i||²).
    3. If ||delta|| > clip_norm, scale each delta_i by clip_norm / ||delta||.
    4. Return old_i + clipped_delta_i as the new parameters.

    Args:
        old_params: List of parameter arrays **before** local training
                    (the global parameters received from the server).
        new_params: List of parameter arrays **after** local training.
        clip_norm:  Maximum allowed L2 norm of the combined update.

    Returns:
        Clipped parameter list with the same shapes as new_params.

    Raises:
        ValueError: If old_params and new_params have different lengths.
    """
    if len(old_params) != len(new_params):
        raise ValueError(
            f"old_params has {len(old_params)} arrays but "
            f"new_params has {len(new_params)}."
        )

    # Compute per-layer deltas
    deltas = [n - o for n, o in zip(new_params, old_params)]

    # Global L2 norm of the flattened update vector
    global_norm = math.sqrt(
        sum(float(np.sum(d ** 2)) for d in deltas)
    )

    if global_norm <= clip_norm or global_norm == 0.0:
        return new_params  # already within budget — no change needed

    # Scale factor to project onto the L2 ball
    scale = clip_norm / global_norm
    clipped = [o + d * scale for o, d in zip(old_params, deltas)]
    return clipped


# ---------------------------------------------------------------------------
# Utility: compute update norm (for logging)
# ---------------------------------------------------------------------------


def compute_update_norm(
    old_params: List[np.ndarray],
    new_params: List[np.ndarray],
) -> float:
    """Return the L2 norm of (new_params − old_params).

    Useful for logging the raw update magnitude before clipping to
    understand how much clipping is actually applied.

    Args:
        old_params: Parameters before local training.
        new_params: Parameters after local training.

    Returns:
        L2 norm (a non-negative float).
    """
    return math.sqrt(
        sum(float(np.sum((n - o) ** 2)) for n, o in zip(new_params, old_params))
    )
