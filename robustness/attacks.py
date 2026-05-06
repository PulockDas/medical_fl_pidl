"""
Data-poisoning attacks for federated learning robustness experiments.

Both attacks wrap an existing ``torch.utils.data.Dataset`` and are applied
only on the designated malicious clients.  The server and the rest of the
pipeline remain unchanged — attacks are purely client-side data corruption.

Attack types
------------
GaussianNoiseDataset
    Adds i.i.d. Gaussian noise to each image tensor after the standard
    transforms have been applied.  Operates in normalised pixel space so
    the scale of ``noise_std`` is relative to the ImageNet-normalised range
    (roughly ±3σ after normalisation).

LabelFlipDataset
    Replaces each label with a uniformly random *wrong* class with probability
    ``flip_probability``.  Only wrong classes are picked — the true label is
    never returned as the flipped label.

Usage
-----
In ``client_app.py``, when a client is identified as malicious::

    from robustness.attacks import wrap_dataset_with_attack
    train_loader = wrap_dataset_with_attack(
        original_loader,
        attack_type="gaussian_noise",
        noise_std=0.5,
        num_classes=4,
        label_flip_probability=0.3,
        seed=42 + client_id,   # different seed per client
    )
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Gaussian noise attack
# ---------------------------------------------------------------------------


class GaussianNoiseDataset(Dataset):
    """Wraps a dataset and adds Gaussian noise to every image.

    Args:
        base_dataset: The original ``Dataset`` to wrap.
        noise_std:    Standard deviation of the additive noise (in
                      normalised image space, e.g. 0.5 ≈ moderate noise).
        seed:         Optional per-client RNG seed for reproducibility.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        noise_std: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.noise_std    = noise_std
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        noise = torch.randn_like(image, generator=self._rng) * self.noise_std
        return image + noise, label


# ---------------------------------------------------------------------------
# Label-flip attack
# ---------------------------------------------------------------------------


class LabelFlipDataset(Dataset):
    """Wraps a dataset and randomly replaces labels with wrong classes.

    Args:
        base_dataset:      The original ``Dataset`` to wrap.
        num_classes:       Total number of classes.
        flip_probability:  Probability each label is replaced (0.0–1.0).
        seed:              Optional per-client RNG seed for reproducibility.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        num_classes: int,
        flip_probability: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for label flipping.")
        self.base_dataset     = base_dataset
        self.num_classes      = num_classes
        self.flip_probability = flip_probability
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        if self._rng.random() < self.flip_probability:
            # Pick uniformly from wrong classes only
            wrong_classes = [c for c in range(self.num_classes) if c != label]
            label = self._rng.choice(wrong_classes)
        return image, label


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def wrap_dataset_with_attack(
    original_loader: DataLoader,
    attack_type: str,
    noise_std: float = 0.5,
    num_classes: int = 4,
    label_flip_probability: float = 0.3,
    seed: int = 42,
) -> DataLoader:
    """Return a new DataLoader whose dataset is wrapped with the attack.

    The returned DataLoader preserves all settings (batch_size, shuffle,
    num_workers, etc.) from *original_loader*.

    Args:
        original_loader:        The clean client DataLoader to corrupt.
        attack_type:            ``"gaussian_noise"`` or ``"label_flip"``.
        noise_std:              Std-dev for Gaussian noise attack.
        num_classes:            Number of classes for label-flip attack.
        label_flip_probability: Flip probability for label-flip attack.
        seed:                   RNG seed (use ``base_seed + client_id``
                                to give each malicious client a different seed).

    Returns:
        DataLoader wrapping an attacked dataset.

    Raises:
        ValueError: If ``attack_type`` is unrecognised.
    """
    base_dataset = original_loader.dataset

    if attack_type == "gaussian_noise":
        attacked_dataset: Dataset = GaussianNoiseDataset(
            base_dataset,
            noise_std=noise_std,
            seed=seed,
        )
    elif attack_type == "label_flip":
        attacked_dataset = LabelFlipDataset(
            base_dataset,
            num_classes=num_classes,
            flip_probability=label_flip_probability,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown attack_type '{attack_type}'. "
            "Choose 'gaussian_noise' or 'label_flip'."
        )

    # Reconstruct DataLoader preserving original settings
    return DataLoader(
        attacked_dataset,
        batch_size=original_loader.batch_size,
        shuffle=True,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=original_loader.drop_last,
    )
