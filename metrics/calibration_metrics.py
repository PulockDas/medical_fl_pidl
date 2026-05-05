"""
Model calibration metrics for probabilistic medical image classifiers.

What is calibration?
--------------------
A model is *well-calibrated* if its predicted confidence equals its empirical
accuracy: when the model says "80% confident", it should be right ≈80% of the
time.  Poor calibration is common in neural networks fine-tuned on small medical
datasets and critical in clinical settings where uncertainty estimates guide
downstream decisions.

Metrics implemented
-------------------
ECE  (Expected Calibration Error)
    Standard calibration metric. Uses equal-width bins on [0, 1].
    ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

MCE  (Maximum Calibration Error)
    Worst-case calibration gap across bins.
    MCE = max_b |acc(B_b) − conf(B_b)|

ACE  (Average Calibration Error)
    ECE variant using equal-mass bins (each bin contains the same number of
    samples) instead of equal-width bins.  Gives a fairer estimate when
    predictions cluster near high or low confidence.

Brier Score
    Proper scoring rule for probability predictions.
    BS = (1/N) Σ_i Σ_c (p_ic − y_ic)²
    Perfect model: 0.  Uniform random: 1 − 1/C.

Mean Confidence
    Mean of the maximum predicted probability across samples.

Mean Entropy
    Mean Shannon entropy of predicted distributions (nats).
    H(p) = −Σ_c p_c · log(p_c)

Reliability Diagram Data
    Per-bin accuracy, confidence, and sample fraction for plotting.

Input convention
----------------
All public functions accept:
  ``y_prob`` : ``(N, C)`` softmax probability matrix (torch.Tensor or ndarray)
  ``y_true`` : ``(N,)``   integer ground-truth class indices
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Convert a torch.Tensor, list, or ndarray to a CPU NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_conf_correct(
    y_prob: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_confidence, correctness) arrays from probability matrix.

    Args:
        y_prob: ``(N, C)`` probability matrix.
        y_true: ``(N,)`` integer ground-truth labels.

    Returns:
        ``(confidences, correctness)`` — both ``(N,)`` float arrays.
        ``confidences`` = max predicted probability per sample.
        ``correctness`` = 1.0 if argmax == y_true, else 0.0.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    return confidences, correctness


# ---------------------------------------------------------------------------
# Bin statistics — equal-width (for ECE / MCE)
# ---------------------------------------------------------------------------


def _equal_width_bins(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bin accuracy, mean confidence, and sample fraction.

    Equal-width binning: [0, 1/B), [1/B, 2/B), …, [(B−1)/B, 1].

    Args:
        confidences: ``(N,)`` max predicted probabilities.
        correctness: ``(N,)`` binary correctness indicators.
        n_bins:      Number of equal-width bins.

    Returns:
        ``(bin_acc, bin_conf, bin_frac)`` — each ``(n_bins,)`` float array.
        Unpopulated bins have ``bin_acc = bin_conf = bin_frac = 0``.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc   = np.zeros(n_bins)
    bin_conf  = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    n         = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right endpoint in last bin
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        cnt = mask.sum()
        if cnt > 0:
            bin_acc[i]   = correctness[mask].mean()
            bin_conf[i]  = confidences[mask].mean()
            bin_count[i] = cnt

    bin_frac = bin_count / max(n, 1)
    return bin_acc, bin_conf, bin_frac


# ---------------------------------------------------------------------------
# Bin statistics — equal-mass (for ACE)
# ---------------------------------------------------------------------------


def _equal_mass_bins(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bin accuracy, mean confidence, and sample fraction using
    equal-mass (quantile) binning.

    Each bin contains ≈ N / n_bins samples, regardless of the confidence
    distribution.  This avoids empty bins in ECE when most predictions
    cluster near 0 or 1 (common after softmax temperature is low).

    Args:
        confidences: ``(N,)`` max predicted probabilities.
        correctness: ``(N,)`` binary correctness indicators.
        n_bins:      Number of equal-mass bins.

    Returns:
        ``(bin_acc, bin_conf, bin_frac)`` — each ``(n_bins,)`` float array.
    """
    n          = len(confidences)
    sort_idx   = np.argsort(confidences)
    conf_sorted = confidences[sort_idx]
    corr_sorted = correctness[sort_idx]

    # Split sorted arrays into n_bins chunks
    splits = np.array_split(np.arange(n), n_bins)

    bin_acc  = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_frac = np.zeros(n_bins)

    for i, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        bin_acc[i]  = corr_sorted[idx].mean()
        bin_conf[i] = conf_sorted[idx].mean()
        bin_frac[i] = len(idx) / n

    return bin_acc, bin_conf, bin_frac


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def compute_ece(
    y_prob: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) with equal-width bins.

    ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    Lower is better (0 = perfectly calibrated).

    Args:
        y_prob:  ``(N, C)`` softmax probability matrix.
        y_true:  ``(N,)``   integer ground-truth labels.
        n_bins:  Number of equal-width confidence bins (default 15).

    Returns:
        ECE as a float in ``[0, 1]``.
    """
    y_prob_np = _to_numpy(y_prob).astype(float)
    y_true_np = _to_numpy(y_true).astype(int)

    confs, corrs = _extract_conf_correct(y_prob_np, y_true_np)
    bin_acc, bin_conf, bin_frac = _equal_width_bins(confs, corrs, n_bins)
    return float(np.sum(bin_frac * np.abs(bin_acc - bin_conf)))


# ---------------------------------------------------------------------------
# MCE
# ---------------------------------------------------------------------------


def compute_mce(
    y_prob: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error (MCE) — worst-case bin calibration gap.

    MCE = max_b |acc(B_b) − conf(B_b)|

    More sensitive to extreme overconfidence / underconfidence in any bin
    than ECE.  Important for safety-critical screening tools.

    Args:
        y_prob:  ``(N, C)`` softmax probability matrix.
        y_true:  ``(N,)``   integer ground-truth labels.
        n_bins:  Number of equal-width confidence bins.

    Returns:
        MCE as a float in ``[0, 1]``.
    """
    y_prob_np = _to_numpy(y_prob).astype(float)
    y_true_np = _to_numpy(y_true).astype(int)

    confs, corrs = _extract_conf_correct(y_prob_np, y_true_np)
    bin_acc, bin_conf, bin_frac = _equal_width_bins(confs, corrs, n_bins)

    populated = bin_frac > 0
    if not populated.any():
        return 0.0
    return float(np.abs(bin_acc[populated] - bin_conf[populated]).max())


# ---------------------------------------------------------------------------
# ACE
# ---------------------------------------------------------------------------


def compute_ace(
    y_prob: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
) -> float:
    """Average Calibration Error (ACE) with equal-mass bins.

    Like ECE but uses quantile binning so every bin has approximately the
    same number of samples.  ACE is more robust to confidence clustering
    (e.g. when a network is very overconfident on most samples).

    Args:
        y_prob:  ``(N, C)`` softmax probability matrix.
        y_true:  ``(N,)``   integer ground-truth labels.
        n_bins:  Number of equal-mass bins (default 15).

    Returns:
        ACE as a float in ``[0, 1]``.
    """
    y_prob_np = _to_numpy(y_prob).astype(float)
    y_true_np = _to_numpy(y_true).astype(int)

    confs, corrs = _extract_conf_correct(y_prob_np, y_true_np)
    bin_acc, bin_conf, bin_frac = _equal_mass_bins(confs, corrs, n_bins)
    return float(np.sum(bin_frac * np.abs(bin_acc - bin_conf)))


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------


def compute_brier_score(
    y_prob: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
) -> float:
    """Multiclass Brier Score — a proper scoring rule for probabilities.

    BS = (1/N) Σ_i Σ_c (p_ic − y_ic)²

    where ``y_ic = 1`` if sample ``i`` has label ``c``, else 0.
    Lower is better: 0.0 = perfect, ``1 − 1/C`` = uniformly random classifier.

    Args:
        y_prob:  ``(N, C)`` softmax probability matrix.
        y_true:  ``(N,)``   integer label tensor.

    Returns:
        Brier score as a float ≥ 0.
    """
    y_prob_np = _to_numpy(y_prob).astype(np.float64)
    y_true_np = _to_numpy(y_true).astype(int)
    n, c      = y_prob_np.shape

    one_hot          = np.zeros((n, c), dtype=np.float64)
    one_hot[np.arange(n), y_true_np] = 1.0

    return float(np.mean(np.sum((y_prob_np - one_hot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Mean confidence and entropy
# ---------------------------------------------------------------------------


def compute_mean_confidence(
    y_prob: Union[np.ndarray, torch.Tensor],
) -> float:
    """Mean of the maximum predicted probability across all samples.

    Measures average model confidence.  High confidence + low accuracy
    indicates overconfidence — common after local training with limited data
    in federated settings.

    Args:
        y_prob: ``(N, C)`` softmax probability matrix.

    Returns:
        Mean max-confidence in ``[1/C, 1]``.
    """
    y_prob_np = _to_numpy(y_prob).astype(float)
    return float(y_prob_np.max(axis=1).mean())


def compute_mean_entropy(
    y_prob: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-9,
) -> float:
    """Mean Shannon entropy of predicted distributions (nats).

    H(p_i) = −Σ_c p_ic · log(p_ic)

    Low entropy → high confidence (not necessarily correct).
    High entropy → uncertain prediction, possible domain shift.

    Maximum possible entropy = log(C) nats for a uniform distribution
    over C classes.

    Args:
        y_prob: ``(N, C)`` softmax probability matrix.
        eps:    Small constant added before log to prevent log(0).

    Returns:
        Mean entropy across all N samples (nats).
    """
    y_prob_np = _to_numpy(y_prob).astype(float)
    p         = np.clip(y_prob_np, eps, 1.0)
    H         = -(p * np.log(p)).sum(axis=1)
    return float(H.mean())


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------


def reliability_diagram_data(
    y_prob: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
    scheme: str = "equal_width",
) -> dict[str, np.ndarray]:
    """Compute data for a reliability (confidence calibration) diagram.

    A well-calibrated model should have its reliability curve fall along
    the diagonal (confidence = accuracy) in the plot.

    Args:
        y_prob:  ``(N, C)`` softmax probability matrix.
        y_true:  ``(N,)``   integer ground-truth labels.
        n_bins:  Number of bins.
        scheme:  ``"equal_width"`` (standard ECE) or ``"equal_mass"`` (ACE).

    Returns:
        Dict with keys:

          ``"bin_acc"``   — ``(n_bins,)`` mean accuracy per bin.
          ``"bin_conf"``  — ``(n_bins,)`` mean confidence per bin.
          ``"bin_frac"``  — ``(n_bins,)`` fraction of samples per bin.
          ``"bin_edges"`` — ``(n_bins+1,)`` edge values (equal_width only;
                            otherwise an integer range is returned).
          ``"gap"``       — ``(n_bins,)`` signed gap ``(acc − conf)`` per bin.
    """
    y_prob_np = _to_numpy(y_prob).astype(float)
    y_true_np = _to_numpy(y_true).astype(int)

    confs, corrs = _extract_conf_correct(y_prob_np, y_true_np)

    if scheme == "equal_mass":
        bin_acc, bin_conf, bin_frac = _equal_mass_bins(confs, corrs, n_bins)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)  # approximate
    else:
        bin_acc, bin_conf, bin_frac = _equal_width_bins(confs, corrs, n_bins)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    return {
        "bin_acc":   bin_acc,
        "bin_conf":  bin_conf,
        "bin_frac":  bin_frac,
        "bin_edges": bin_edges,
        "gap":       bin_acc - bin_conf,   # positive = underconfident, negative = overconfident
    }


# ---------------------------------------------------------------------------
# Unified calibration report
# ---------------------------------------------------------------------------


def compute_calibration_metrics(
    y_prob: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    n_bins: int = 15,
    prefix: str = "",
) -> dict[str, float]:
    """Compute ECE, MCE, ACE, Brier score, mean confidence, and mean entropy.

    Designed to be called after every FL round alongside
    ``compute_classification_metrics()`` or by itself for quick calibration
    audits.

    Args:
        y_prob:  ``(N, C)`` softmax probability matrix.
        y_true:  ``(N,)``   integer ground-truth labels.
        n_bins:  Number of bins for ECE / MCE / ACE (default 15).
        prefix:  String prepended to every key (e.g. ``"test_"``).

    Returns:
        Flat ``{str: float}`` dict with keys:
          ``ece``, ``mce``, ``ace``, ``brier_score``,
          ``mean_confidence``, ``mean_entropy``
        (all prepended with ``prefix``).
    """
    p = prefix
    return {
        f"{p}ece":              compute_ece(y_prob, y_true, n_bins),
        f"{p}mce":              compute_mce(y_prob, y_true, n_bins),
        f"{p}ace":              compute_ace(y_prob, y_true, n_bins),
        f"{p}brier_score":      compute_brier_score(y_prob, y_true),
        f"{p}mean_confidence":  compute_mean_confidence(y_prob),
        f"{p}mean_entropy":     compute_mean_entropy(y_prob),
    }


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_calibration_json(
    diagram_data: dict[str, np.ndarray],
    scalar_metrics: dict[str, float],
    path: Union[str, Path],
    round_number: Optional[int] = None,
) -> None:
    """Save reliability diagram data and scalar calibration metrics to JSON.

    Args:
        diagram_data:    Return value of ``reliability_diagram_data()``.
        scalar_metrics:  Return value of ``compute_calibration_metrics()``.
        path:            Destination JSON path.
        round_number:    Optional FL round number.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc: dict = {}
    if round_number is not None:
        doc["round"] = round_number

    doc["scalar_metrics"]   = scalar_metrics
    doc["reliability_diagram"] = {
        k: v.tolist() for k, v in diagram_data.items()
    }

    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
