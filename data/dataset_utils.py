"""
Federated DataLoader builder for KaggleHub-downloaded medical image datasets.

Primary entry point
-------------------
::

    result = build_federated_dataloaders(
        image_root   = "/path/from/find_image_root()",
        num_clients  = 3,
        test_split   = 0.20,
        batch_size   = 32,
        image_size   = (224, 224),
        augment      = True,
        random_seed  = 42,
        save_summary_to = "results/run_01",   # optional
    )

    loaders     = result["client_train_loaders"]   # list[DataLoader], one per client
    test_loader = result["global_test_loader"]
    n_classes   = result["num_classes"]
    names       = result["class_names"]
    summary     = result["dataset_summary"]        # also saved as dataset_summary.json

Design notes
------------
- Uses ``torchvision.datasets.ImageFolder`` for automatic class detection.
- Grayscale images are converted to 3-channel RGB so ResNet18 weights apply.
- Train/test split is stratified (sklearn) to preserve class proportions.
- Client partitioning is stratified IID by default (each client mirrors the
  global class distribution). Non-IID Dirichlet is wired in but optional.
- ``_TransformDataset`` applies transforms lazily so train/test splits can
  share the same underlying ImageFolder without duplicate data.

Colab tip
---------
If DataLoader is slow or hangs in Colab, pass ``num_workers=0``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data.partitioning import (
    compute_client_class_distribution,
    partition_indices,
    partition_stats,
)


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (pretrained ResNet backbone)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Lazy-transform wrapper
# ---------------------------------------------------------------------------


class _TransformDataset(Dataset):
    """Apply a transform to a Subset without modifying the parent dataset.

    Using this wrapper lets the train split and test split share one
    underlying ``ImageFolder`` (with ``transform=None``) while each
    receiving their own augmentation / normalisation pipeline.

    Args:
        subset:    A ``torch.utils.data.Subset`` (or any Dataset) that
                   returns raw PIL Images.
        transform: Transform applied in ``__getitem__``.
    """

    def __init__(self, subset: Dataset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Grayscale detection
# ---------------------------------------------------------------------------


def _is_grayscale_dataset(dataset: ImageFolder, n_check: int = 5) -> bool:
    """Sample the first ``n_check`` images to determine if the source is grayscale.

    Args:
        dataset:  An ``ImageFolder`` instance with ``transform=None``.
        n_check:  Number of images to inspect (first n are checked).

    Returns:
        ``True`` if all sampled images are single-channel (mode 'L' or 'LA').
    """
    if len(dataset) == 0:
        return False

    grayscale_modes = {"L", "LA"}
    for i in range(min(n_check, len(dataset))):
        img_path, _ = dataset.samples[i]
        try:
            with Image.open(img_path) as img:
                if img.mode not in grayscale_modes:
                    return False
        except Exception:
            return False

    return True


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------


def build_train_transform(
    image_size: tuple[int, int],
    is_grayscale: bool = False,
    augment: bool = True,
) -> transforms.Compose:
    """Build the training augmentation + normalisation pipeline.

    Grayscale images are replicated across 3 channels *before* augmentation
    so that ``ColorJitter`` and ``Normalize`` receive a 3-channel tensor.

    Args:
        image_size:   ``(H, W)`` resize target.
        is_grayscale: If True, insert a ``Grayscale(3)`` step after resize.
        augment:      If True, add random flips, rotation, and colour jitter.

    Returns:
        A ``transforms.Compose`` pipeline.
    """
    h, w = image_size
    steps: list = [transforms.Resize((h, w))]

    if is_grayscale:
        # Replicate single channel → 3 identical channels (RGB-compatible)
        steps.append(transforms.Grayscale(num_output_channels=3))

    if augment:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(steps)


def build_test_transform(
    image_size: tuple[int, int],
    is_grayscale: bool = False,
) -> transforms.Compose:
    """Build the test / validation normalisation-only pipeline.

    Args:
        image_size:   ``(H, W)`` resize target.
        is_grayscale: If True, insert a ``Grayscale(3)`` step after resize.

    Returns:
        A ``transforms.Compose`` pipeline.
    """
    h, w = image_size
    steps: list = [transforms.Resize((h, w))]

    if is_grayscale:
        steps.append(transforms.Grayscale(num_output_channels=3))

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(steps)


# Backward-compatible aliases used by existing federated/ code
def build_val_transform(config) -> transforms.Compose:  # type: ignore[override]
    """Alias for build_test_transform, accepts a DatasetConfig."""
    is_gray = getattr(config, "_is_grayscale", False)
    return build_test_transform(config.image_size, is_grayscale=is_gray)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _class_counts_for_indices(
    indices: np.ndarray,
    all_targets: np.ndarray,
    class_names: list[str],
) -> dict[str, int]:
    """Return {class_name: count} for a subset of dataset indices."""
    subset_targets = all_targets[indices]
    counts = {name: 0 for name in class_names}
    for lbl in subset_targets:
        counts[class_names[int(lbl)]] += 1
    return counts


def _build_summary(
    image_root: str,
    all_targets: np.ndarray,
    class_names: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    client_index_lists: list[list[int]],
    strategy: str,
    is_grayscale: bool,
    image_size: tuple[int, int],
    batch_size: int,
    augment: bool,
    random_seed: int,
    test_split: float,
    num_workers: int,
) -> dict[str, Any]:
    """Assemble the dataset_summary dict."""
    num_classes = len(class_names)

    # Global class counts
    global_counts = _class_counts_for_indices(
        np.arange(len(all_targets)), all_targets, class_names
    )
    train_counts = _class_counts_for_indices(train_indices, all_targets, class_names)
    test_counts = _class_counts_for_indices(test_indices, all_targets, class_names)

    # Per-client distributions
    client_dists = compute_client_class_distribution(
        client_index_lists, all_targets, class_names
    )
    client_sizes = [len(idxs) for idxs in client_index_lists]

    return {
        "image_root": str(image_root),
        "num_classes": num_classes,
        "class_names": class_names,
        "total_images": int(len(all_targets)),
        "class_counts": global_counts,
        "is_grayscale_source": is_grayscale,
        "splits": {
            "train": {
                "total": int(len(train_indices)),
                "class_counts": train_counts,
            },
            "test": {
                "total": int(len(test_indices)),
                "class_counts": test_counts,
            },
        },
        "partitioning": {
            "strategy": strategy,
            "num_clients": len(client_index_lists),
            "client_sizes": client_sizes,
            "client_class_distributions": client_dists,
        },
        "settings": {
            "image_size": list(image_size),
            "batch_size": batch_size,
            "augment": augment,
            "random_seed": random_seed,
            "test_split": test_split,
            "num_workers": num_workers,
        },
    }


def _save_summary(summary: dict[str, Any], save_dir: str | Path) -> Path:
    """Write ``dataset_summary.json`` to ``save_dir``."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[dataset_utils] Summary saved → {out_path}")
    return out_path


def print_dataset_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable overview of the dataset summary dict."""
    SEP = "─" * 58
    print(f"\n{SEP}")
    print(f"  Dataset root : {Path(summary['image_root']).name}")
    print(f"  Classes ({summary['num_classes']}): {summary['class_names']}")
    print(f"  Total images : {summary['total_images']:,}")
    print(f"  Grayscale src: {summary['is_grayscale_source']}")
    print(f"  {SEP}")

    print("  Global class distribution:")
    for cls, cnt in summary["class_counts"].items():
        frac = cnt / max(summary["total_images"], 1)
        print(f"    {cls:<20} {cnt:>6,}  ({frac:.1%})")

    sp = summary["splits"]
    print(f"\n  Train: {sp['train']['total']:,} images")
    print(f"  Test : {sp['test']['total']:,} images")

    pt = summary["partitioning"]
    print(f"\n  Partitioning : {pt['strategy']}  ({pt['num_clients']} clients)")
    for cid, (size, dist) in enumerate(
        zip(pt["client_sizes"], pt["client_class_distributions"])
    ):
        top = sorted(dist.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{k}:{v}" for k, v in top)
        print(f"    Client {cid}: {size:>5,} samples  [{top_str}{'…' if len(dist) > 3 else ''}]")

    print(f"{SEP}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_federated_dataloaders(
    image_root: str | Path,
    num_clients: int,
    test_split: float = 0.20,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    augment: bool = True,
    random_seed: int = 42,
    num_workers: int = 2,
    partitioning: Literal["iid", "dirichlet"] = "iid",
    dirichlet_alpha: float = 0.5,
    save_summary_to: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Convert a KaggleHub-downloaded dataset into federated DataLoaders.

    Takes the path returned by ``find_image_root()`` and produces one
    DataLoader per federated client plus a shared global test DataLoader.

    Args:
        image_root:       ImageFolder-compatible root directory where each
                          subdirectory is a class name (use ``find_image_root()``
                          from ``data.kaggle_loader`` to resolve this path).
        num_clients:      Number of federated clients (typically 3, 4, or 5).
        test_split:       Fraction of the full dataset held out as the global
                          test set (default 0.20).
        batch_size:       Batch size for all DataLoaders.
        image_size:       ``(H, W)`` resize target (default (224, 224) for ResNet18).
        augment:          If True, apply random augmentation to training batches.
        random_seed:      Seed for all stochastic operations (split, shuffle,
                          partition). Pass the same seed to reproduce a run.
        num_workers:      CPU workers per DataLoader. Use 0 in Colab if
                          multiprocessing causes issues.
        partitioning:     ``"iid"`` (stratified, default) or ``"dirichlet"``
                          (non-IID, for heterogeneity experiments).
        dirichlet_alpha:  Concentration for non-IID Dirichlet split (lower =
                          more heterogeneous). Ignored when partitioning='iid'.
        save_summary_to:  If provided, writes ``dataset_summary.json`` to this
                          directory (e.g. the run's results folder).

    Returns:
        Dict with keys:

        ``client_train_loaders``
            ``list[DataLoader]`` – one per client, shuffled, with augmentation.

        ``global_test_loader``
            ``DataLoader`` – shared stratified test set, not shuffled.

        ``num_classes``
            ``int`` – number of classes detected from the folder structure.

        ``class_names``
            ``list[str]`` – class names in label-index order.

        ``dataset_summary``
            ``dict`` – full statistics (see :func:`print_dataset_summary`).

    Raises:
        FileNotFoundError: If ``image_root`` does not exist.
        RuntimeError:      If fewer than 2 class directories are found.
        ValueError:        If ``test_split`` is outside (0, 1) or any client
                           ends up with 0 samples.

    Example::

        from data.kaggle_loader import download_kaggle_dataset, find_image_root
        from data.dataset_utils import build_federated_dataloaders, print_dataset_summary

        raw_path = download_kaggle_dataset(
            "masoudnickparvar/brain-tumor-mri-dataset", "Brain Tumor MRI"
        )
        img_root = find_image_root(raw_path)

        result = build_federated_dataloaders(
            image_root=img_root,
            num_clients=3,
            test_split=0.20,
            batch_size=32,
            augment=True,
            random_seed=42,
            save_summary_to="results/run_01",
        )
        print_dataset_summary(result["dataset_summary"])
    """
    # ── Validation ────────────────────────────────────────────────────────
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"image_root does not exist: '{image_root}'")
    if not (0.0 < test_split < 1.0):
        raise ValueError(f"test_split must be in (0, 1); got {test_split}.")
    if num_clients < 1:
        raise ValueError(f"num_clients must be ≥ 1; got {num_clients}.")

    print(f"[build_federated_dataloaders] Loading ImageFolder from '{image_root}' …")

    # ── Step 1: Load ImageFolder with no transforms (raw PIL images) ──────
    # transform=None means __getitem__ returns PIL Image → we apply transforms
    # via _TransformDataset, allowing train/test to share the same Folder.
    base_dataset = ImageFolder(root=str(image_root), transform=None)

    num_classes: int = len(base_dataset.classes)
    class_names: list[str] = list(base_dataset.classes)
    all_targets: np.ndarray = np.array(base_dataset.targets)

    if num_classes < 2:
        raise RuntimeError(
            f"ImageFolder found only {num_classes} class(es) in '{image_root}'.\n"
            f"  At least 2 class subdirectories containing images are required.\n"
            f"  Call preview_dataset_structure('{image_root}') to inspect."
        )

    print(f"  → {len(base_dataset):,} images  |  {num_classes} classes: {class_names}")

    # ── Step 2: Detect grayscale source images ────────────────────────────
    is_grayscale = _is_grayscale_dataset(base_dataset)
    if is_grayscale:
        print("  → Grayscale source detected. Images will be converted to 3-channel RGB.")

    # ── Step 3: Build transforms ──────────────────────────────────────────
    train_tf = build_train_transform(image_size, is_grayscale=is_grayscale, augment=augment)
    test_tf = build_test_transform(image_size, is_grayscale=is_grayscale)

    # ── Step 4: Stratified global train / test split ──────────────────────
    all_indices = np.arange(len(base_dataset))

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_split,
        stratify=all_targets,
        random_state=random_seed,
    )

    print(
        f"  → Split: {len(train_indices):,} train  |  {len(test_indices):,} test  "
        f"(stratified, seed={random_seed})"
    )

    # ── Step 5: Partition training indices across clients ─────────────────
    client_index_lists = partition_indices(
        train_indices=train_indices,
        all_targets=all_targets,
        num_clients=num_clients,
        strategy=partitioning,
        dirichlet_alpha=dirichlet_alpha,
        seed=random_seed,
    )

    sizes_str = "  |  ".join(f"client {i}: {len(c):,}" for i, c in enumerate(client_index_lists))
    print(f"  → Partitioning ({partitioning}): {sizes_str}")

    # ── Step 6: Build per-client train DataLoaders ────────────────────────
    pin = torch.cuda.is_available()

    client_train_loaders: list[DataLoader] = []
    for cid, idx_list in enumerate(client_index_lists):
        client_subset = Subset(base_dataset, idx_list)
        client_ds = _TransformDataset(client_subset, train_tf)
        loader = DataLoader(
            client_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=False,
        )
        client_train_loaders.append(loader)

    # ── Step 7: Build global test DataLoader ──────────────────────────────
    test_subset = Subset(base_dataset, test_indices.tolist())
    test_ds = _TransformDataset(test_subset, test_tf)
    global_test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # ── Step 8: Build dataset summary ─────────────────────────────────────
    summary = _build_summary(
        image_root=str(image_root),
        all_targets=all_targets,
        class_names=class_names,
        train_indices=train_indices,
        test_indices=test_indices,
        client_index_lists=client_index_lists,
        strategy=partitioning,
        is_grayscale=is_grayscale,
        image_size=image_size,
        batch_size=batch_size,
        augment=augment,
        random_seed=random_seed,
        test_split=test_split,
        num_workers=num_workers,
    )

    if save_summary_to is not None:
        _save_summary(summary, save_summary_to)

    print(
        f"[build_federated_dataloaders] Done. "
        f"{num_clients} client loaders + 1 test loader ready.\n"
    )

    return {
        "client_train_loaders": client_train_loaders,
        "global_test_loader": global_test_loader,
        "num_classes": num_classes,
        "class_names": class_names,
        "dataset_summary": summary,
    }


# ---------------------------------------------------------------------------
# Standalone helpers (kept for backward compatibility with federated/ code)
# ---------------------------------------------------------------------------


def compute_class_weights(
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a DataLoader.

    Useful for imbalanced datasets (e.g. HAM10000).

    Args:
        loader:      Training DataLoader.
        num_classes: Number of classes.
        device:      Target device.

    Returns:
        ``(num_classes,)`` float tensor; mean weight ≈ 1.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in loader:
        for lbl in labels:
            counts[int(lbl)] += 1.0
    weights = counts.sum() / (num_classes * counts.clamp(min=1.0))
    return weights.to(device)
