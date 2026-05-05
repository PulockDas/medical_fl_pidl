"""
Hierarchical experiment configuration for federated PIDL training.

All hyperparameters live here so notebooks and scripts only need to
instantiate ExperimentConfig and pass it through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from configs.dataset_configs import BRAIN_TUMOR, DatasetConfig


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class FederatedConfig:
    """Flower federation settings.

    SecAgg+ parameters are auto-adjusted when num_clients changes (see
    ExperimentConfig.finalize()). Override manually if needed.
    """

    num_clients: int = 3           # Supports 3, 4, or 5 out of the box
    num_rounds: int = 20
    fraction_fit: float = 1.0      # Fraction of clients sampled per round
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 3
    min_evaluate_clients: int = 3
    min_available_clients: int = 3

    # --- SecAgg+ ---
    use_secagg_plus: bool = True
    # Total shares each client generates for secret splitting
    secagg_num_shares: int = 3
    # Minimum shares required to reconstruct; dropout tolerance = num_shares - threshold
    secagg_reconstruction_threshold: int = 2
    # Clipping bound for model weight vectors (SecAgg+ requirement)
    secagg_clipping_range: float = 8.0
    # Maximum weight across all clients
    secagg_quantization_range: int = 4


@dataclass
class ModelConfig:
    """ResNet backbone and head settings."""

    backbone: Literal["resnet18"] = "resnet18"  # Extendable to resnet34, etc.
    pretrained: bool = True                      # Use ImageNet weights
    freeze_backbone: bool = False                # Fine-tune entire network
    dropout_rate: float = 0.3
    # Which ResNet layer's feature map is fed to the PIDL loss.
    # layer1: 56×56 (edges)  layer2: 28×28 (patterns)
    # layer3: 14×14 (semantic regions) ← default
    # layer4:  7×7  (high-level semantics)
    pidl_feature_layer: Literal["layer1", "layer2", "layer3", "layer4"] = "layer3"


@dataclass
class TrainingConfig:
    """Local client training settings."""

    batch_size: int = 32
    local_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    # SGD-specific
    sgd_momentum: float = 0.9
    # LR scheduler (applied per local training run)
    use_lr_scheduler: bool = False
    lr_scheduler_step_size: int = 1
    lr_scheduler_gamma: float = 0.9


@dataclass
class PIDLConfig:
    """Perona-Malik Physics-Informed Regularization settings.

    The PIDL loss enforces the Perona-Malik anisotropic diffusion PDE as a
    soft constraint on the chosen ResNet feature map layer.

    PDE (steady-state): div( c(|∇F|) · ∇F ) = 0
    PIDL loss:          L_PIDL = mean( ||div(c(|∇F|)·∇F)||² )
    Total loss:         L = L_CE + lambda_pidl * L_PIDL
    """

    enabled: bool = True
    # Weight of the PIDL regularization term
    lambda_pidl: float = 0.01
    # Edge-sensitivity threshold K in the diffusivity function c(s)
    diffusivity_k: float = 0.1
    # 'lorentzian': c(s) = 1/(1+(s/K)²)  |  'gaussian': c(s) = exp(-(s/K)²)
    diffusivity_type: Literal["lorentzian", "gaussian"] = "lorentzian"
    # Normalize feature maps to [0,1] before computing gradients (scale-invariant)
    normalize_features: bool = True


@dataclass
class PartitioningConfig:
    """Federated data partitioning strategy."""

    strategy: Literal["iid", "dirichlet"] = "iid"
    # Dirichlet concentration parameter (lower → more heterogeneous)
    dirichlet_alpha: float = 0.5
    # Minimum samples per client per class to avoid degenerate splits
    min_samples_per_client: int = 10


@dataclass
class LoggingConfig:
    """Metric logging and checkpointing settings."""

    log_every_n_rounds: int = 1
    save_best_model: bool = True
    # Output sub-directory inside results_dir
    run_name: Optional[str] = None   # None = auto-generated from timestamp
    log_format: Literal["json", "csv", "both"] = "both"


# ---------------------------------------------------------------------------
# Master experiment config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level configuration object passed to all pipeline components."""

    experiment_name: str = "federated_pidl"
    dataset: DatasetConfig = field(default_factory=lambda: BRAIN_TUMOR)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pidl: PIDLConfig = field(default_factory=PIDLConfig)
    partitioning: PartitioningConfig = field(default_factory=PartitioningConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    seed: int = 42
    results_dir: str = "results"
    # "auto" = cuda if available, then mps, then cpu
    device: str = "auto"

    def finalize(self) -> "ExperimentConfig":
        """Adjust dependent parameters after num_clients is set.

        Call this after changing num_clients or other mutually-dependent fields.
        Returns self for chaining.
        """
        n = self.federated.num_clients
        # SecAgg+ shares must be ≤ num_clients; keep 1-client dropout tolerance
        self.federated.secagg_num_shares = n
        self.federated.secagg_reconstruction_threshold = max(2, n - 1)

        # FL selection thresholds must not exceed num_clients
        self.federated.min_fit_clients = min(self.federated.min_fit_clients, n)
        self.federated.min_evaluate_clients = min(self.federated.min_evaluate_clients, n)
        self.federated.min_available_clients = n
        return self

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON logging."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def make_config(
    dataset_key: str = "brain_tumor_mri",
    num_clients: int = 3,
    num_rounds: int = 20,
    lambda_pidl: float = 0.01,
    use_secagg_plus: bool = True,
    local_epochs: int = 3,
    partitioning: Literal["iid", "dirichlet"] = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> ExperimentConfig:
    """Build and finalize an ExperimentConfig from the most common parameters.

    Args:
        dataset_key: Short key from AVAILABLE_DATASETS (e.g. "chest_xray").
        num_clients: 3, 4, or 5.
        num_rounds: Total FL rounds.
        lambda_pidl: Weight for the PIDL regularization term.
        use_secagg_plus: Enable Flower SecAgg+ secure aggregation.
        local_epochs: Local training epochs per round.
        partitioning: "iid" or "dirichlet".
        dirichlet_alpha: Concentration for non-IID Dirichlet split.
        seed: Random seed.
        **kwargs: Extra fields applied to the top-level ExperimentConfig.

    Returns:
        A fully-finalized ExperimentConfig.
    """
    from configs.dataset_configs import get_dataset_config

    cfg = ExperimentConfig(
        dataset=get_dataset_config(dataset_key),
        federated=FederatedConfig(
            num_clients=num_clients,
            num_rounds=num_rounds,
            use_secagg_plus=use_secagg_plus,
        ),
        training=TrainingConfig(local_epochs=local_epochs),
        pidl=PIDLConfig(lambda_pidl=lambda_pidl),
        partitioning=PartitioningConfig(
            strategy=partitioning,
            dirichlet_alpha=dirichlet_alpha,
        ),
        seed=seed,
        **kwargs,
    )
    return cfg.finalize()
