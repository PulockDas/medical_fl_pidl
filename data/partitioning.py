"""
Federated data partitioning strategies.

All functions operate on raw integer index arrays so they are completely
decoupled from PyTorch Dataset objects and can be unit-tested independently.

Strategies
----------
stratified_iid_partition  (default)
    Each client receives approximately 1/num_clients of *every* class.
    Guarantees balanced class representation across all clients.

dirichlet_partition  (future / optional)
    Class probabilities are drawn from Dir(alpha). Low alpha produces
    highly heterogeneous splits (some clients see only 1–2 classes).

Usage
-----
::

    from data.partitioning import stratified_iid_partition, partition_stats

    client_index_lists = stratified_iid_partition(
        train_indices=train_idx,   # np.ndarray of global dataset indices
        all_targets=targets,       # full dataset label array
        num_clients=3,
        seed=42,
    )
    # client_index_lists[i] → list[int] of global indices for client i
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Stratified IID partitioning (default)
# ---------------------------------------------------------------------------


def stratified_iid_partition(
    train_indices: np.ndarray,
    all_targets: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> list[list[int]]:
    """Divide training indices evenly across clients, preserving class ratios.

    For each class independently:
      1. Collect all training indices that belong to that class.
      2. Shuffle them with a seeded RNG.
      3. Split into ``num_clients`` approximately equal chunks.
      4. Give chunk *i* to client *i*.

    The result is that every client's local dataset mirrors the global class
    distribution (IID), and sizes differ by at most 1 sample per class.

    Args:
        train_indices: 1-D array of global dataset index integers belonging
                       to the training split.
        all_targets:   Full-dataset label array (length = total dataset size).
                       ``all_targets[i]`` is the integer class label for sample i.
        num_clients:   Number of federated clients.
        seed:          Random seed for reproducibility.

    Returns:
        List of ``num_clients`` lists, each containing global dataset indices
        for one client. Indices within each list are shuffled.

    Raises:
        ValueError: If any client would receive zero samples.
    """
    rng = np.random.default_rng(seed)
    train_targets = all_targets[train_indices]
    classes = np.unique(train_targets)

    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        # All training indices that belong to this class
        cls_mask = train_targets == cls
        cls_indices = train_indices[cls_mask]
        rng.shuffle(cls_indices)

        # Split evenly; numpy handles unequal sizes gracefully
        chunks = np.array_split(cls_indices, num_clients)
        for cid, chunk in enumerate(chunks):
            client_buckets[cid].extend(chunk.tolist())

    # Final shuffle within each client to interleave classes
    for cid in range(num_clients):
        arr = np.array(client_buckets[cid])
        rng.shuffle(arr)
        client_buckets[cid] = arr.tolist()

    # Sanity check
    for cid, bucket in enumerate(client_buckets):
        if len(bucket) == 0:
            raise ValueError(
                f"Client {cid} received 0 training samples after partitioning.\n"
                f"  Reduce num_clients or increase the dataset size."
            )

    return client_buckets


# ---------------------------------------------------------------------------
# Non-IID Dirichlet partitioning (future / optional)
# ---------------------------------------------------------------------------


def dirichlet_partition(
    train_indices: np.ndarray,
    all_targets: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 10,
    seed: int = 42,
) -> list[list[int]]:
    """Non-IID partition using a Dirichlet distribution over class labels.

    For each class, sample allocation proportions from Dir(alpha) and assign
    that fraction of the class's training samples to each client.

    Lower alpha → more heterogeneous (one client dominates each class).
    Higher alpha (e.g. 100) → approaches the IID distribution.

    Args:
        train_indices:         Global training index array.
        all_targets:           Full-dataset label array.
        num_clients:           Number of federated clients.
        alpha:                 Dirichlet concentration parameter.
        min_samples_per_client: Raise if any client receives fewer samples.
        seed:                  Random seed.

    Returns:
        List of per-client index lists.

    Raises:
        ValueError: If min_samples_per_client constraint cannot be met.
    """
    rng = np.random.default_rng(seed)
    train_targets = all_targets[train_indices]
    classes = np.unique(train_targets)

    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_indices = train_indices[train_targets == cls]
        rng.shuffle(cls_indices)
        n = len(cls_indices)
        if n == 0:
            continue

        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        splits = (proportions * n).astype(int)
        splits[-1] = n - splits[:-1].sum()

        ptr = 0
        for cid in range(num_clients):
            take = int(splits[cid])
            client_buckets[cid].extend(cls_indices[ptr : ptr + take].tolist())
            ptr += take

    for cid, bucket in enumerate(client_buckets):
        if len(bucket) < min_samples_per_client:
            raise ValueError(
                f"Client {cid} received only {len(bucket)} samples "
                f"(minimum: {min_samples_per_client}).\n"
                f"  Try increasing alpha (less heterogeneous) or num_samples."
            )

    return client_buckets


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def partition_indices(
    train_indices: np.ndarray,
    all_targets: np.ndarray,
    num_clients: int,
    strategy: Literal["iid", "dirichlet"] = "iid",
    dirichlet_alpha: float = 0.5,
    min_samples_per_client: int = 5,
    seed: int = 42,
) -> list[list[int]]:
    """Dispatch to the appropriate partitioning strategy.

    Args:
        train_indices:         Global training index array.
        all_targets:           Full-dataset label array.
        num_clients:           Number of federated clients.
        strategy:              ``"iid"`` (default) or ``"dirichlet"``.
        dirichlet_alpha:       Concentration for non-IID Dirichlet split.
        min_samples_per_client: Minimum samples each client must receive.
        seed:                  Random seed.

    Returns:
        List of per-client index lists.
    """
    if strategy == "iid":
        return stratified_iid_partition(
            train_indices, all_targets, num_clients, seed=seed
        )
    elif strategy == "dirichlet":
        return dirichlet_partition(
            train_indices, all_targets, num_clients,
            alpha=dirichlet_alpha,
            min_samples_per_client=min_samples_per_client,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown partitioning strategy '{strategy}'. "
            "Choose 'iid' or 'dirichlet'."
        )


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------


def compute_client_class_distribution(
    client_index_lists: list[list[int]],
    all_targets: np.ndarray,
    class_names: list[str],
) -> list[dict[str, int]]:
    """Compute per-client class sample counts.

    Args:
        client_index_lists: Output of any ``*_partition`` function.
        all_targets:        Full-dataset label array.
        class_names:        Ordered list of class name strings.

    Returns:
        List of dicts, one per client.
        Each dict maps class_name → sample count for that client.
    """
    distributions: list[dict[str, int]] = []
    for indices in client_index_lists:
        client_targets = all_targets[np.array(indices, dtype=int)]
        counts = {name: 0 for name in class_names}
        for lbl in client_targets:
            counts[class_names[int(lbl)]] += 1
        distributions.append(counts)
    return distributions


def partition_stats(
    client_index_lists: list[list[int]],
    all_targets: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    """Return a human-readable list of per-client statistics.

    Args:
        client_index_lists: Output of any ``*_partition`` function.
        all_targets:        Full-dataset label array.
        class_names:        Ordered class name strings.

    Returns:
        List of dicts with keys:
          ``client_id``, ``n_samples``, ``class_counts``, ``class_fractions``.
    """
    distributions = compute_client_class_distribution(
        client_index_lists, all_targets, class_names
    )
    stats = []
    for cid, dist in enumerate(distributions):
        total = sum(dist.values())
        fractions = {k: round(v / max(total, 1), 4) for k, v in dist.items()}
        stats.append(
            {
                "client_id": cid,
                "n_samples": total,
                "class_counts": dist,
                "class_fractions": fractions,
            }
        )
    return stats
