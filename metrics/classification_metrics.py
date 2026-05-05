"""
Comprehensive classification metrics for federated medical image experiments.

Return convention
-----------------
``compute_classification_metrics()`` always returns a dict with three keys:

``flat``
    A plain ``{str: float}`` dict.  Every metric is a scalar — safe to write
    directly to a CSV row or a JSONL round record.  Per-class metrics are
    included as ``"precision_<class_name>"``, ``"recall_<class_name>"``, etc.

``per_class``
    A nested ``{class_name: {metric: float}}`` dict for JSON reports and
    per-class CSV tables.

``confusion_matrix``
    A ``(C, C)`` NumPy int array.  Rows = true class, Columns = predicted class.

Metric glossary
---------------
accuracy           Top-1 accuracy.
balanced_accuracy  Mean per-class recall (robust to class imbalance).
f1_macro           Unweighted mean of per-class F1.
f1_micro           Global TP / (TP + 0.5*(FP+FN)) — equals accuracy for
                   multi-class when all classes are considered.
f1_weighted        Frequency-weighted mean of per-class F1.
precision_macro    Unweighted mean of per-class precision.
recall_macro       Unweighted mean of per-class recall  = sensitivity_macro.
sensitivity_macro  Alias for recall_macro (TP / (TP + FN)).
specificity_macro  Unweighted mean of per-class specificity (TN / (TN + FP)).
roc_auc_macro      Macro-average OvR AUROC.  ``nan`` if only one class is present.
pr_auc_macro       Macro-average OvR Average Precision (area under PR curve).
                   ``nan`` if only one class is present.
mean_confidence    Mean of max predicted probability across all samples.
mean_entropy       Mean Shannon entropy of predicted distributions (nats).
ece                Expected Calibration Error (15 equal-width bins).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Convert torch.Tensor, list, or ndarray to a CPU NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _fill_class_names(class_names: Optional[list[str]], num_classes: int) -> list[str]:
    """Return class_names, falling back to "class_0", "class_1", … if None."""
    if class_names is not None and len(class_names) == num_classes:
        return class_names
    return [f"class_{i}" for i in range(num_classes)]


def _safe_nan(value: float) -> float:
    """Return value unchanged, converting any Python NaN to float('nan')."""
    return float(value)


# ---------------------------------------------------------------------------
# Per-class specificity from confusion matrix
# ---------------------------------------------------------------------------


def _specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """Compute per-class specificity from a square confusion matrix.

    For class c:
        TN_c = total samples - (row_sum_c + col_sum_c - cm[c, c])
        FP_c = col_sum_c - cm[c, c]
        specificity_c = TN_c / (TN_c + FP_c)

    Args:
        cm: ``(C, C)`` confusion matrix (row = true, col = predicted).

    Returns:
        ``(C,)`` float array of per-class specificities.
    """
    num_classes = cm.shape[0]
    total       = cm.sum()
    spec        = np.zeros(num_classes, dtype=float)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp   # predicted as c but actually something else
        fn = cm[c, :].sum() - tp   # actually c but predicted as something else
        tn = total - tp - fp - fn
        denom = tn + fp
        spec[c] = tn / denom if denom > 0 else 0.0

    return spec


# ---------------------------------------------------------------------------
# Per-class OvR ROC-AUC and PR-AUC
# ---------------------------------------------------------------------------


def _per_class_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Per-class OvR AUROC.  Returns ``nan`` for classes with no positive samples."""
    y_bin   = label_binarize(y_true, classes=list(range(num_classes)))
    aucs    = np.full(num_classes, float("nan"))
    if num_classes == 2:
        # label_binarize returns (N, 1) for binary; fix shape
        y_bin = np.hstack([1 - y_bin, y_bin])
    for c in range(num_classes):
        if y_bin[:, c].sum() == 0:
            continue
        try:
            aucs[c] = roc_auc_score(y_bin[:, c], y_prob[:, c])
        except ValueError:
            pass
    return aucs


def _per_class_prauc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Per-class OvR Average Precision (area under PR curve).  Returns ``nan`` for empty classes."""
    y_bin   = label_binarize(y_true, classes=list(range(num_classes)))
    prs     = np.full(num_classes, float("nan"))
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    for c in range(num_classes):
        if y_bin[:, c].sum() == 0:
            continue
        try:
            prs[c] = average_precision_score(y_bin[:, c], y_prob[:, c])
        except ValueError:
            pass
    return prs


# ---------------------------------------------------------------------------
# Confidence and entropy
# ---------------------------------------------------------------------------


def compute_mean_confidence(y_prob: np.ndarray) -> float:
    """Mean of the maximum predicted probability across all samples.

    A perfectly confident model scores 1.0; a uniform distribution over C
    classes scores 1/C.  High mean confidence with low accuracy indicates
    overconfidence — an important calibration signal for medical classifiers.

    Args:
        y_prob: ``(N, C)`` probability matrix.

    Returns:
        Mean max-confidence, float in [1/C, 1].
    """
    return float(y_prob.max(axis=1).mean())


def compute_mean_entropy(y_prob: np.ndarray, eps: float = 1e-9) -> float:
    """Mean Shannon entropy of predicted distributions (nats).

    H(p) = -Σ_c p_c · log(p_c)

    A uniform distribution over C classes has maximum entropy log(C) nats.
    Near-zero entropy means the model is very confident (not necessarily
    correct).  High entropy in predictions often signals domain shift or
    classes the model has not seen during local training.

    Args:
        y_prob: ``(N, C)`` probability matrix.
        eps:    Small constant to avoid log(0).

    Returns:
        Mean entropy across all N samples (nats).
    """
    p     = np.clip(y_prob, eps, 1.0)
    H     = -(p * np.log(p)).sum(axis=1)
    return float(H.mean())


# ---------------------------------------------------------------------------
# ECE (imported here so compute_classification_metrics is self-contained)
# ---------------------------------------------------------------------------


def _compute_ece(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Local copy of ECE to avoid circular imports with calibration_metrics."""
    confidences  = y_prob.max(axis=1)
    predictions  = y_prob.argmax(axis=1)
    correctness  = (predictions == y_true).astype(float)
    bin_edges    = np.linspace(0.0, 1.0, n_bins + 1)
    ece          = 0.0
    n            = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if mask.sum() > 0:
            acc  = correctness[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor, list],
    y_prob: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor, list, None] = None,
    class_names: Optional[list[str]] = None,
    prefix: str = "",
    n_calibration_bins: int = 15,
) -> dict[str, Any]:
    """Compute a full suite of classification and calibration metrics.

    Args:
        y_true:             ``(N,)`` integer ground-truth class indices.
        y_prob:             ``(N, C)`` softmax probability matrix.
        y_pred:             ``(N,)`` predicted class indices.  If ``None``,
                            computed as ``y_prob.argmax(axis=1)``.
        class_names:        List of C class name strings.  Defaults to
                            ``["class_0", "class_1", …]``.
        prefix:             String prepended to every key in the ``flat`` dict
                            (e.g. ``"test_"`` → ``"test_accuracy"``).
        n_calibration_bins: Number of equal-width bins for ECE.

    Returns:
        Dict with three keys:

        ``"flat"``            ``{str: float}`` — all scalar metrics, ready for CSV.
        ``"per_class"``       ``{class_name: {metric: float}}`` — for JSON / per-class CSV.
        ``"confusion_matrix"`` ``(C, C)`` NumPy int array.
    """
    # ── Convert inputs ────────────────────────────────────────────────────
    y_true_np: np.ndarray = _to_numpy(y_true).astype(int)
    y_prob_np: np.ndarray = _to_numpy(y_prob).astype(float)
    y_pred_np: np.ndarray = (
        _to_numpy(y_pred).astype(int)
        if y_pred is not None
        else y_prob_np.argmax(axis=1)
    )

    num_classes  = y_prob_np.shape[1]
    names        = _fill_class_names(class_names, num_classes)
    all_labels   = list(range(num_classes))

    p = prefix  # shorthand

    flat:     dict[str, float] = {}
    per_class: dict[str, dict[str, float]] = {n: {} for n in names}

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true_np, y_pred_np, labels=all_labels)

    # ── Overall accuracy ──────────────────────────────────────────────────
    flat[f"{p}accuracy"]          = float(accuracy_score(y_true_np, y_pred_np))
    flat[f"{p}balanced_accuracy"] = float(balanced_accuracy_score(y_true_np, y_pred_np))

    # ── F1 scores ─────────────────────────────────────────────────────────
    flat[f"{p}f1_macro"]    = float(f1_score(y_true_np, y_pred_np, average="macro",    zero_division=0))
    flat[f"{p}f1_micro"]    = float(f1_score(y_true_np, y_pred_np, average="micro",    zero_division=0))
    flat[f"{p}f1_weighted"] = float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0))

    # ── Precision / Recall (macro) ────────────────────────────────────────
    flat[f"{p}precision_macro"] = float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    flat[f"{p}recall_macro"]    = float(recall_score(   y_true_np, y_pred_np, average="macro", zero_division=0))

    # sensitivity = recall (TP / (TP + FN)), kept explicit for medical convention
    flat[f"{p}sensitivity_macro"] = flat[f"{p}recall_macro"]

    # ── Specificity (macro) — from confusion matrix ───────────────────────
    spec_per_class = _specificity_per_class(cm)
    flat[f"{p}specificity_macro"] = float(spec_per_class.mean())

    # ── ROC-AUC (macro OvR) ───────────────────────────────────────────────
    try:
        if num_classes == 2:
            roc_auc_val = roc_auc_score(y_true_np, y_prob_np[:, 1])
        else:
            roc_auc_val = roc_auc_score(
                y_true_np, y_prob_np, multi_class="ovr", average="macro"
            )
        flat[f"{p}roc_auc_macro"] = float(roc_auc_val)
    except ValueError:
        flat[f"{p}roc_auc_macro"] = float("nan")

    # ── PR-AUC (macro OvR average precision) ─────────────────────────────
    try:
        if num_classes == 2:
            pr_auc_val = average_precision_score(y_true_np, y_prob_np[:, 1])
        else:
            pr_auc_val = average_precision_score(
                y_true_np, y_prob_np, average="macro"
            )
        flat[f"{p}pr_auc_macro"] = float(pr_auc_val)
    except ValueError:
        flat[f"{p}pr_auc_macro"] = float("nan")

    # ── Confidence and entropy ────────────────────────────────────────────
    flat[f"{p}mean_confidence"] = compute_mean_confidence(y_prob_np)
    flat[f"{p}mean_entropy"]    = compute_mean_entropy(y_prob_np)

    # ── ECE ───────────────────────────────────────────────────────────────
    flat[f"{p}ece"] = _compute_ece(y_prob_np, y_true_np, n_bins=n_calibration_bins)

    # ── Per-class metrics ─────────────────────────────────────────────────
    pc_precision = precision_score(y_true_np, y_pred_np, labels=all_labels, average=None, zero_division=0)
    pc_recall    = recall_score(   y_true_np, y_pred_np, labels=all_labels, average=None, zero_division=0)
    pc_f1        = f1_score(       y_true_np, y_pred_np, labels=all_labels, average=None, zero_division=0)
    pc_support   = cm.sum(axis=1)               # true counts per class
    pc_auroc     = _per_class_auroc(y_true_np, y_prob_np, num_classes)
    pc_prauc     = _per_class_prauc(y_true_np, y_prob_np, num_classes)

    for i, name in enumerate(names):
        # Flat keys — safe class name (replace spaces/slashes for CSV headers)
        safe = name.replace(" ", "_").replace("/", "_")

        flat[f"{p}precision_{safe}"]    = float(pc_precision[i])
        flat[f"{p}recall_{safe}"]       = float(pc_recall[i])
        flat[f"{p}f1_{safe}"]           = float(pc_f1[i])
        flat[f"{p}specificity_{safe}"]  = float(spec_per_class[i])
        flat[f"{p}support_{safe}"]      = int(pc_support[i])

        # Nested per-class dict
        per_class[name] = {
            "precision":   float(pc_precision[i]),
            "recall":      float(pc_recall[i]),
            "sensitivity": float(pc_recall[i]),   # medical alias
            "f1":          float(pc_f1[i]),
            "specificity": float(spec_per_class[i]),
            "support":     int(pc_support[i]),
            "roc_auc":     _safe_nan(pc_auroc[i]),
            "pr_auc":      _safe_nan(pc_prauc[i]),
        }

    return {
        "flat":             flat,
        "per_class":        per_class,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Convenience: confusion matrix only
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor, list],
    y_pred: Union[np.ndarray, torch.Tensor, list],
    num_classes: int,
) -> np.ndarray:
    """Return a ``(C, C)`` confusion matrix.

    Args:
        y_true:      ``(N,)`` integer ground-truth labels.
        y_pred:      ``(N,)`` integer predicted labels.
        num_classes: Total number of classes C.

    Returns:
        NumPy int array of shape ``(C, C)``.
    """
    return confusion_matrix(
        _to_numpy(y_true).astype(int),
        _to_numpy(y_pred).astype(int),
        labels=list(range(num_classes)),
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_metrics_csv(
    flat_metrics: dict[str, float],
    path: Union[str, Path],
    append: bool = False,
) -> None:
    """Write a flat metrics dict to a CSV file (one row per call).

    If the file already exists and ``append=True``, a new row is appended
    without repeating the header.  Use ``append=True`` to accumulate one
    row per FL round.

    Args:
        flat_metrics: Dict of ``{metric_name: scalar}`` (from ``result["flat"]``).
        path:         Destination CSV path.
        append:       If True, append a data row; if False, create/overwrite.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode    = "a" if (append and path.exists()) else "w"
    do_header = not (append and path.exists())

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_metrics.keys()))
        if do_header:
            writer.writeheader()
        # Cast NaN to string "nan" so pandas reads it correctly
        writer.writerow({k: v for k, v in flat_metrics.items()})


def save_per_class_csv(
    per_class_metrics: dict[str, dict[str, float]],
    path: Union[str, Path],
) -> None:
    """Write per-class metrics to a CSV file with one row per class.

    Columns: ``class_name, precision, recall, sensitivity, f1,
    specificity, support, roc_auc, pr_auc``.

    Args:
        per_class_metrics: Nested dict from ``result["per_class"]``.
        path:              Destination CSV path.
    """
    if not per_class_metrics:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "class_name", "precision", "recall", "sensitivity", "f1",
        "specificity", "support", "roc_auc", "pr_auc",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, m in per_class_metrics.items():
            writer.writerow({"class_name": class_name, **m})


def save_metrics_json(
    result: dict[str, Any],
    path: Union[str, Path],
    round_number: Optional[int] = None,
) -> None:
    """Write the full metrics result (flat + per_class + confusion matrix) to JSON.

    The confusion matrix is serialised as a nested list so JSON can represent it.

    Args:
        result:       Return value of ``compute_classification_metrics()``.
        path:         Destination JSON path.
        round_number: Optional FL round number added under the ``"round"`` key.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc: dict[str, Any] = {}
    if round_number is not None:
        doc["round"] = round_number
    doc["flat"]             = result["flat"]
    doc["per_class"]        = result["per_class"]
    doc["confusion_matrix"] = result["confusion_matrix"].tolist()

    # Replace Python float nan → None for valid JSON
    doc_str = json.dumps(doc, indent=2, default=lambda v: None if (isinstance(v, float) and np.isnan(v)) else v)
    path.write_text(doc_str, encoding="utf-8")


# ---------------------------------------------------------------------------
# FL history aggregation
# ---------------------------------------------------------------------------


def aggregate_round_metrics(
    history: list[dict[str, Any]],
    keys: Optional[list[str]] = None,
) -> dict[str, list[float]]:
    """Collect per-round flat metric dicts into per-metric lists for plotting.

    Args:
        history: List of per-round metric dicts (e.g. from
                 ``LoggingFedAvg.get_history()`` or a loaded JSONL file).
        keys:    Metric keys to extract.  ``None`` → extract every numeric key
                 from the first record.

    Returns:
        ``{metric_name: [val_round1, val_round2, …]}``
    """
    if not history:
        return {}

    if keys is None:
        keys = [
            k for k, v in history[0].items()
            if isinstance(v, (int, float))
        ]

    out: dict[str, list[float]] = {k: [] for k in keys}
    for record in history:
        for k in keys:
            out[k].append(float(record.get(k, float("nan"))))
    return out
