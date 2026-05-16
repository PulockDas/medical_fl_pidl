"""
Core training and evaluation functions for federated PIDL learning.

These functions are shared between client_app.py and server_app.py.
They are deliberately stateless (no global experiment config) so they
can be called safely from multiple Flower client processes / threads.

Shared data cache
-----------------
``get_federated_data()`` builds the full federated DataLoader set once
and caches it by (data_root, num_clients, batch_size, image_size, seed,
test_split, partitioning, dirichlet_alpha).
In Flower simulation mode all client_fn and server_fn calls run in the
same process, so every call after the first is a free dict lookup.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet_pidl import ResNetPIDL


# ---------------------------------------------------------------------------
# Shared data-loading cache (thread-safe)
# ---------------------------------------------------------------------------

_DATA_CACHE: dict[tuple, dict] = {}
_DATA_CACHE_LOCK = threading.Lock()


def get_federated_data(
    data_root: str,
    num_clients: int,
    batch_size: int,
    image_size: int,
    random_seed: int,
    test_split: float = 0.2,
    augment: bool = True,
    num_workers: int = 0,
    save_summary_to: str | None = None,
    partitioning: str = "iid",
    dirichlet_alpha: float = 0.5,
) -> dict:
    """Return (cached) federated DataLoaders and dataset info.

    On the first call for a given set of parameters, this calls
    ``build_federated_dataloaders`` from ``data.dataset_utils`` and stores
    the result. Subsequent calls with the same key return immediately.

    Args:
        data_root:      Path to the ImageFolder-compatible dataset root.
        num_clients:    Number of federated clients / data partitions.
        batch_size:     Mini-batch size for all DataLoaders.
        image_size:     Square image size (pixels) passed to transforms.
        random_seed:    Seed for reproducible splits and partitions.
        test_split:     Fraction of data to hold out as global test set.
        augment:        Enable random augmentations on training splits.
        num_workers:    DataLoader worker processes (0 = main process).
        save_summary_to: Optional directory path for ``dataset_summary.json``.
        partitioning:    ``"iid"`` (default) or ``"dirichlet"`` for non-IID splits.
        dirichlet_alpha: Dirichlet concentration when ``partitioning="dirichlet"``.

    Returns:
        Dict with keys:
          - ``client_train_loaders`` : list[DataLoader]
          - ``global_test_loader``   : DataLoader
          - ``num_classes``          : int
          - ``class_names``          : list[str]
          - ``dataset_summary``      : dict
    """
    from data.dataset_utils import build_federated_dataloaders

    key = (
        data_root,
        num_clients,
        batch_size,
        image_size,
        random_seed,
        test_split,
        partitioning,
        round(float(dirichlet_alpha), 8),
    )
    with _DATA_CACHE_LOCK:
        if key not in _DATA_CACHE:
            print(
                f"[task] Building federated data for {num_clients} clients "
                f"from: {data_root} (partitioning={partitioning})"
            )
            _DATA_CACHE[key] = build_federated_dataloaders(
                image_root=data_root,
                num_clients=num_clients,
                test_split=test_split,
                batch_size=batch_size,
                image_size=(image_size, image_size),
                augment=augment,
                random_seed=random_seed,
                num_workers=num_workers,
                partitioning=partitioning,  # type: ignore[arg-type]
                dirichlet_alpha=dirichlet_alpha,
                save_summary_to=save_summary_to,
            )
        return _DATA_CACHE[key]


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def build_optimizer(
    model: ResNetPIDL,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build an optimizer from explicit hyperparameters.

    Args:
        model:          The model whose parameters to optimise.
        lr:             Learning rate.
        optimizer_type: One of ``"adam"``, ``"adamw"``, ``"sgd"``.
        weight_decay:   L2 regularization coefficient.
        momentum:       SGD momentum (ignored for Adam / AdamW).

    Returns:
        Configured optimizer (only trainable parameters included).
    """
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_type}'. "
                         "Choose 'adam', 'adamw', or 'sgd'.")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: ResNetPIDL,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 1,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> dict[str, float]:
    """Run local training for ``num_epochs`` and return aggregated metrics.

    The criterion must be a ``PIDLLoss`` (or compatible) that accepts
    ``(logits, labels, feature_map)`` and returns
    ``(total_loss, ce_loss, pidl_loss)``.

    The model's ``pidl_feature_layer`` attribute selects which intermediate
    ResNet layer's feature map is passed to the PIDL regularizer.

    Args:
        model:       ResNetPIDL (already on device).
        dataloader:  Client training DataLoader.
        criterion:   PIDLLoss composite loss.
        optimizer:   Configured optimizer.
        device:      Compute device.
        num_epochs:  Number of local epochs per FL round.
        scheduler:   Optional LR scheduler (stepped once per epoch).

    Returns:
        Dict with keys:
          ``train_loss``, ``train_ce_loss``, ``train_pidl_loss``,
          ``train_accuracy`` — all epoch-averaged floats.
    """
    model.train()
    total_loss = total_ce = total_pidl = total_correct = total_samples = 0

    for _epoch in range(num_epochs):
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # return_features=True: get all four intermediate feature maps.
            # The active PIDL layer is chosen by model.pidl_feature_layer,
            # which is set when the model is built from the run config.
            logits, feature_maps = model(images, return_features=True)
            feature_map = feature_maps[model.pidl_feature_layer]

            loss, ce_loss, pidl_loss = criterion(logits, labels, feature_map)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss    += loss.item()      * bs
            total_ce      += ce_loss.item()   * bs
            total_pidl    += pidl_loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += bs

        if scheduler is not None:
            scheduler.step()

    n = max(total_samples, 1)
    return {
        "train_loss":      total_loss    / n,
        "train_ce_loss":   total_ce      / n,
        "train_pidl_loss": total_pidl    / n,
        "train_accuracy":  total_correct / n,
    }


# ---------------------------------------------------------------------------
# Evaluation loop (client-side, CE only)
# ---------------------------------------------------------------------------


def evaluate(
    model: ResNetPIDL,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on a DataLoader using cross-entropy only (no PIDL).

    Args:
        model:      ResNetPIDL (already on device).
        dataloader: Validation DataLoader.
        criterion:  CrossEntropyLoss.
        device:     Compute device.

    Returns:
        Dict with keys: ``val_loss``, ``val_accuracy``, ``num_samples``.
    """
    model.eval()
    total_loss = total_correct = total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # return_features=False: no feature-map allocation at eval time.
            logits = model(images, return_features=False)
            loss   = criterion(logits, labels)

            bs = labels.size(0)
            total_loss    += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += bs

    n = max(total_samples, 1)
    return {
        "val_loss":     total_loss    / n,
        "val_accuracy": total_correct / n,
        "num_samples":  total_samples,
    }


# ---------------------------------------------------------------------------
# Full evaluation (server-side, returns raw logits for downstream metrics)
# ---------------------------------------------------------------------------


def evaluate_full(
    model: ResNetPIDL,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """Full evaluation returning loss, accuracy, and raw tensors for metrics.

    Use this for server-side evaluation where you need F1, AUC, ECE, etc.

    Args:
        model:       ResNetPIDL (already on device).
        dataloader:  Test DataLoader.
        device:      Compute device.
        num_classes: Number of target classes.

    Returns:
        Dict with:
          ``loss``, ``accuracy``, ``num_samples``,
          ``all_logits`` → ``(N, C)`` float32 CPU tensor,
          ``all_labels`` → ``(N,)`` int64 CPU tensor,
          ``all_probs``  → ``(N, C)`` float32 CPU tensor (softmax).
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = total_correct = total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images, return_features=False)
            loss   = ce(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            bs = labels.size(0)
            total_loss    += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += bs

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    n = max(total_samples, 1)

    return {
        "loss":        total_loss    / n,
        "accuracy":    total_correct / n,
        "num_samples": total_samples,
        "all_logits":  logits_cat,
        "all_labels":  labels_cat,
        "all_probs":   F.softmax(logits_cat, dim=1),
    }


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve ``"auto"`` to the best available device.

    Args:
        device_str: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        ``torch.device``.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
