"""
Flower ClientApp — local training with CE + grid-wise Perona-Malik PIDL loss.

Per-client flow each round
--------------------------
1. ``client_fn`` is called by the Flower runtime with a ``Context`` carrying
   the run config and a ``partition-id`` node config key that uniquely
   identifies this client's data shard.
2. ``get_federated_data()`` is called (returns cached result on all but the
   first call) to obtain the pre-built per-client DataLoaders.
3. A ``ResNetPIDL`` model is built and a ``PIDLLoss`` criterion is created
   using the run-config hyperparameters.
4. The resulting ``MedicalFLClient`` handles:
     - ``get_parameters`` → return current weights as NumPy arrays
     - ``fit``            → load global weights, train locally, return updates
     - ``evaluate``       → load global weights, validate on local val split

SecAgg+ on the client
---------------------
``secaggplus_mod`` is added to the ``ClientApp`` as a middleware mod.
It intercepts the ``fit`` response, secret-shares the model update among
the other clients, and sends only the encrypted shares to the server.
The server reconstructs the aggregate without seeing any individual update.
No differential privacy noise is added.

Partition-ID assignment
-----------------------
Flower simulation sets ``context.node_config["partition-id"]`` to an
integer in ``[0, num_clients)``. The same seed used by the server ensures
that ``get_federated_data()`` returns the same splits on the client side.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from configs.experiment_config import ModelConfig
from federated.task import (
    build_optimizer,
    evaluate,
    get_federated_data,
    resolve_device,
    train,
)
from losses.pidl_loss import PIDLLoss
from models.resnet_pidl import build_model, get_model_parameters, set_model_parameters


# ---------------------------------------------------------------------------
# secaggplus_mod import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from flwr.client.mod import secaggplus_mod
    _SECAGG_MOD_AVAILABLE = True
except ImportError:
    _SECAGG_MOD_AVAILABLE = False
    secaggplus_mod = None  # type: ignore[assignment]
    warnings.warn(
        "flwr.client.mod.secaggplus_mod not found. "
        "Install flwr>=1.9.0 for SecAgg+ client support. "
        "Running without SecAgg+ client mod.",
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Run-config parser (mirrors server_app._parse_run_config)
# ---------------------------------------------------------------------------


def _parse_run_config(run_config: dict) -> dict:
    """Cast run_config values to Python types (same logic as server_app).

    Also merges ``FL_RUN_OVERRIDE`` env var (JSON) set by the notebook before
    each ``run_simulation()`` call so experiment-specific config reaches the
    Ray workers that Flower spawns.
    """
    import json as _json
    import os as _os

    merged = dict(run_config)
    override_json = _os.environ.get("FL_RUN_OVERRIDE", "")
    if override_json:
        try:
            merged.update(_json.loads(override_json))
        except _json.JSONDecodeError:
            pass

    def _get(key: str, default, cast):
        val = merged.get(key, default)
        try:
            if cast is bool:
                if isinstance(val, bool):
                    return val
                return str(val).lower() in ("true", "1", "yes")
            return cast(val)
        except (ValueError, TypeError):
            return default

    return {
        "dataset_name":     _get("dataset_name",     "brain_tumor_mri", str),
        "data_root":        _get("data_root",         "",                str),
        "log_dir":          _get("log_dir",           "results/run_001", str),
        "num_classes":      _get("num_classes",       4,                 int),
        "num_clients":      _get("num_clients",       3,                 int),
        "min_fit_clients":  _get("min_fit_clients",   3,                 int),
        "local_epochs":     _get("local_epochs",      2,                 int),
        "batch_size":       _get("batch_size",        32,                int),
        "learning_rate":    _get("learning_rate",     1e-3,              float),
        "image_size":       _get("image_size",        224,               int),
        "feature_layer":    _get("feature_layer",     "layer3",          str),
        "regularizer_type": _get("regularizer_type",  "perona_malik",    str),
        "lambda_pm":        _get("lambda_pm",         0.01,              float),
        "use_grid_loss":    _get("use_grid_loss",     True,              bool),
        "grid_size":        _get("grid_size",         4,                 int),
        "k":                _get("k",                 0.1,               float),
        "random_seed":      _get("random_seed",       42,                int),
    }


# ---------------------------------------------------------------------------
# MedicalFLClient — NumPyClient implementation
# ---------------------------------------------------------------------------


class MedicalFLClient(NumPyClient):
    """Flower NumPyClient for one federated participant.

    Holds all state needed for local training and local evaluation.
    One instance is created per ``client_fn`` call by the Flower runtime.

    Args:
        model:        ResNetPIDL (NOT yet on device — moved in ``__init__``).
        train_loader: DataLoader for this client's training partition.
        val_loader:   DataLoader for this client's validation set.
        criterion:    PIDLLoss composite loss (CE + grid-wise PM PIDL).
        device:       Compute device (``cpu``, ``cuda``, or ``mps``).
        client_id:    Integer client index used for logging.
        cfg:          Parsed run config dict.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: PIDLLoss,
        device: torch.device,
        client_id: int,
        cfg: dict,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.ce_only      = nn.CrossEntropyLoss()  # evaluation uses CE only
        self.device       = device
        self.client_id    = client_id
        self.cfg          = cfg

    # ------------------------------------------------------------------
    # NumPyClient protocol
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Any]) -> list:
        """Return current model weights as a list of NumPy arrays."""
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: list,
        config: dict[str, Any],
    ) -> tuple[list, int, dict[str, Any]]:
        """Load global weights, run local training, return updated weights.

        Steps:
        1. Load the aggregated global parameters into the local model.
        2. Build a fresh optimizer (no state carry-over between rounds).
        3. Train for ``local_epochs`` using CE + grid-wise PM PIDL loss.
        4. Return updated parameters, number of training samples, and metrics.
        """
        set_model_parameters(self.model, parameters)

        optimizer = build_optimizer(
            self.model,
            lr=self.cfg["learning_rate"],
            optimizer_type="adam",
            weight_decay=1e-4,
        )

        metrics = train(
            model=self.model,
            dataloader=self.train_loader,
            criterion=self.criterion,
            optimizer=optimizer,
            device=self.device,
            num_epochs=self.cfg["local_epochs"],
        )

        num_examples = len(self.train_loader.dataset)
        metrics["num_examples"] = num_examples
        metrics["client_id"]    = self.client_id

        return (
            get_model_parameters(self.model),
            num_examples,
            metrics,
        )

    def evaluate(
        self,
        parameters: list,
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Load global weights and evaluate on the local validation set.

        PIDL loss is NOT applied during evaluation — only cross-entropy.
        This matches the server-side evaluation logic and avoids confounding
        the validation loss with the regularization weight λ.
        """
        set_model_parameters(self.model, parameters)

        metrics = evaluate(
            model=self.model,
            dataloader=self.val_loader,
            criterion=self.ce_only,
            device=self.device,
        )

        num_examples = metrics.pop("num_samples")
        metrics["client_id"]    = self.client_id
        metrics["num_examples"] = num_examples

        return float(metrics["val_loss"]), num_examples, metrics


# ---------------------------------------------------------------------------
# client_fn — entry point called once per simulated client per round
# ---------------------------------------------------------------------------


def client_fn(context: Context):
    """Build and return one federated client.

    Called by the Flower runtime for every simulated client.  Reads
    ``context.run_config`` for hyperparameters and
    ``context.node_config["partition-id"]`` for the data shard index.

    Args:
        context: Flower runtime context.

    Returns:
        A Flower ``Client`` (via ``MedicalFLClient.to_client()``).
    """
    cfg = _parse_run_config(context.run_config)

    # Partition-id: Flower simulation assigns each virtual node an integer
    # identifier in [0, num_supernodes).
    partition_id  = int(context.node_config.get("partition-id", 0))
    num_partitions = int(
        context.node_config.get("num-partitions", cfg["num_clients"])
    )

    device = resolve_device("auto")

    # ── Data ─────────────────────────────────────────────────────────────
    # The same call signature as the server → same splits / same cache.
    # _DATA_CACHE in task.py means this is only computed once per run.
    data = get_federated_data(
        data_root=cfg["data_root"],
        num_clients=num_partitions,
        batch_size=cfg["batch_size"],
        image_size=cfg["image_size"],
        random_seed=cfg["random_seed"],
    )

    client_train_loaders = data["client_train_loaders"]
    global_test_loader   = data["global_test_loader"]   # used as val set on clients
    num_classes          = data.get("num_classes", cfg["num_classes"])

    # Guard: partition_id must be within the list of client loaders
    partition_id = partition_id % len(client_train_loaders)

    train_loader = client_train_loaders[partition_id]
    val_loader   = global_test_loader   # clients share the global test split as val

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(pidl_feature_layer=cfg["feature_layer"])  # type: ignore[arg-type]
    model     = build_model(num_classes=num_classes, config=model_cfg)

    # ── Loss ──────────────────────────────────────────────────────────────
    # PIDLLoss = CrossEntropy + grid-wise Perona-Malik spatial regularizer.
    # Grid-wise: the feature map is divided into grid_size × grid_size
    # non-overlapping local regions. The PM PDE residual is enforced
    # independently within each region (pathology is LOCAL, not global).
    # There is NO temporal component — this is purely spatial.
    criterion = PIDLLoss(
        regularizer_type=cfg["regularizer_type"],
        lambda_pm=cfg["lambda_pm"],
        k=cfg["k"],
        use_grid_loss=cfg["use_grid_loss"],
        grid_size=cfg["grid_size"],
    )

    return MedicalFLClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        client_id=partition_id,
        cfg=cfg,
    ).to_client()


# ---------------------------------------------------------------------------
# Module-level ClientApp — referenced by pyproject.toml
# ---------------------------------------------------------------------------
# secaggplus_mod intercepts the fit response and encrypts the model update
# before it reaches the server.  The server accumulates encrypted shares and
# reconstructs only the aggregate, never individual client updates.

if _SECAGG_MOD_AVAILABLE:
    app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
else:
    app = ClientApp(client_fn=client_fn)
