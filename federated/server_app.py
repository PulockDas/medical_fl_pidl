"""
Flower ServerApp — FedAvg + SecAgg+ + server-side global evaluation.

How it works
------------
``server_fn`` is called once by the Flower runtime when the simulation
starts. It reads all hyperparameters from ``context.run_config`` (set in
``pyproject.toml`` or overridden on the ``flwr run`` command line), builds
the global test DataLoader, creates the ``LoggingFedAvg`` strategy with an
``evaluate_fn`` that evaluates the shared global model after every round,
wraps everything in ``SecAggPlusWorkflow``, and returns
``ServerAppComponents``.

SecAgg+ parameters
------------------
Derived automatically from ``num_clients`` and ``min_fit_clients``:

  num_shares               = num_clients
  reconstruction_threshold = min_fit_clients  (≥ 1 surviving client required)
  clipping_range           = 8.0              (fixed; guards weight magnitude)
  quantization_range       = 4               (fixed; SecAgg+ encoding range)

These choices give dropout tolerance of ``num_clients − min_fit_clients``
without any additional run-config knobs.

No differential privacy noise is added anywhere in this pipeline.

Usage — flwr run
-----------------
::

    cd medical_fl_pidl
    flwr run .

Or with overrides::

    flwr run . --run-config "num_clients=5 num_server_rounds=20"
"""

from __future__ import annotations

import atexit
import time as _time
import warnings
from pathlib import Path
from typing import Optional

from flwr.common import Context, NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from configs.experiment_config import ModelConfig
from federated.strategy_logging import LoggingFedAvg
from federated.task import evaluate_full, get_federated_data, resolve_device
from models.resnet_pidl import ResNetPIDL, build_model, get_model_parameters, set_model_parameters


# ---------------------------------------------------------------------------
# SecAgg+ import (graceful fallback to DefaultWorkflow if unavailable)
# ---------------------------------------------------------------------------

try:
    from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
    _SECAGG_AVAILABLE = True
except ImportError:
    _SECAGG_AVAILABLE = False
    warnings.warn(
        "flwr.server.workflow.SecAggPlusWorkflow not found. "
        "Install flwr>=1.9.0 to enable SecAgg+. "
        "Falling back to DefaultWorkflow.",
        stacklevel=1,
    )
    try:
        from flwr.server.workflow import DefaultWorkflow
    except ImportError:
        DefaultWorkflow = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Run-config parser
# ---------------------------------------------------------------------------


def _parse_run_config(run_config: dict) -> dict:
    """Cast run_config values to Python types.

    Flower delivers ``context.run_config`` values as their TOML types
    (int, float, bool, str) but may occasionally deliver them all as str
    when overridden on the command line.  This function coerces every
    key to the expected type so downstream code never needs to cast.

    When called from a notebook via ``run_simulation()`` (Flower 1.29+),
    ``context.run_config`` only contains the ``pyproject.toml`` defaults.
    Experiment-specific overrides are injected via the ``FL_RUN_OVERRIDE``
    environment variable (JSON string set by the notebook before each run).
    Ray workers inherit environment variables from the parent process.

    Args:
        run_config: Raw ``context.run_config`` dict.

    Returns:
        Typed config dict with all expected keys present.
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
        "dataset_name":       _get("dataset_name",       "brain_tumor_mri", str),
        "data_root":          _get("data_root",           "",                str),
        "log_dir":            _get("log_dir",             "results/run_001", str),
        "num_classes":        _get("num_classes",         4,                 int),
        "num_server_rounds":  _get("num_server_rounds",   10,                int),
        "num_clients":        _get("num_clients",         3,                 int),
        "min_fit_clients":    _get("min_fit_clients",     3,                 int),
        "local_epochs":       _get("local_epochs",        2,                 int),
        "batch_size":         _get("batch_size",          32,                int),
        "learning_rate":      _get("learning_rate",       1e-3,              float),
        "image_size":         _get("image_size",          224,               int),
        "feature_layer":      _get("feature_layer",       "layer3",          str),
        "regularizer_type":   _get("regularizer_type",    "perona_malik",    str),
        "lambda_pm":          _get("lambda_pm",           0.01,              float),
        "use_grid_loss":      _get("use_grid_loss",       True,              bool),
        "grid_size":          _get("grid_size",           4,                 int),
        "k":                             _get("k",                             0.1,      float),
        "random_seed":                   _get("random_seed",                   42,       int),
        # SecAgg+ — read explicitly so pyproject.toml values are honoured
        "secagg_num_shares":               _get("secagg_num_shares",               3,        int),
        "secagg_reconstruction_threshold": _get("secagg_reconstruction_threshold", 2,        int),
        "secagg_max_weight":               _get("secagg_max_weight",               1048575,  int),
    }


# ---------------------------------------------------------------------------
# Internal: back-fill ce_loss / reg_loss into ExperimentLogger history
# ---------------------------------------------------------------------------


def _backfill_training_losses(exp_logger, strategy_history: list[dict]) -> None:
    """Merge per-round training losses from LoggingFedAvg history into
    the ExperimentLogger's in-memory round records.

    ``evaluate_fn`` runs after aggregation; at that point the strategy's
    ``_current_fit_metrics`` has already been stored for the same round.
    We match by round number and fill ``ce_loss`` / ``reg_loss`` /
    ``training_time_sec`` into the logger's history before ``finalize()``
    writes ``fl_rounds.csv``.
    """
    hist_by_round = {r.get("round", -1): r for r in strategy_history}
    for row in exp_logger._rounds_history:
        rnd = row.get("round", -1)
        if rnd in hist_by_round:
            rec = hist_by_round[rnd]
            row["ce_loss"]           = float(rec.get("train_ce_loss",   row.get("ce_loss",  0.0)))
            row["reg_loss"]          = float(rec.get("train_pidl_loss", row.get("reg_loss", 0.0)))
            row["training_time_sec"] = float(rec.get("elapsed_seconds", row.get("training_time_sec", 0.0)))


# ---------------------------------------------------------------------------
# server_fn — entry point called once by the Flower runtime
# ---------------------------------------------------------------------------


def server_fn(context: Context) -> ServerAppComponents:
    """Build and return the server-side components for one federated run.

    Called automatically by Flower when ``flwr run .`` starts.
    Reads ``context.run_config``, builds the global test DataLoader for
    server-side evaluation, constructs ``LoggingFedAvg``, and wraps it in
    ``SecAggPlusWorkflow``.

    Args:
        context: Flower runtime context. ``context.run_config`` contains the
                 run parameters from ``pyproject.toml`` and CLI overrides.

    Returns:
        ``ServerAppComponents`` with strategy, server config, and workflow.
    """
    cfg = _parse_run_config(context.run_config)

    # ── Directories ──────────────────────────────────────────────────────
    log_dir   = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path  = log_dir / "round_metrics.jsonl"

    # ── Device ───────────────────────────────────────────────────────────
    device = resolve_device("auto")
    print(f"[Server] Device: {device}  |  Log dir: {log_dir}")

    # ── Global test DataLoader ────────────────────────────────────────────
    # The server builds the same partitioned dataset as the clients
    # (identical seed → identical splits) and uses only the held-out
    # global test loader for end-of-round evaluation.
    data = get_federated_data(
        data_root=cfg["data_root"],
        num_clients=cfg["num_clients"],
        batch_size=cfg["batch_size"],
        image_size=cfg["image_size"],
        random_seed=cfg["random_seed"],
        save_summary_to=str(log_dir),
    )
    global_test_loader = data["global_test_loader"]
    num_classes        = data.get("num_classes", cfg["num_classes"])
    class_names        = data.get("class_names", [])

    print(
        f"[Server] Dataset  : {cfg['dataset_name']}  "
        f"({num_classes} classes: {class_names})\n"
        f"[Server] Test set : {len(global_test_loader.dataset)} samples"
    )

    # ── Initial model parameters ──────────────────────────────────────────
    # All clients receive the same initialisation so training starts from
    # an identical checkpoint regardless of simulation order.
    model_cfg   = ModelConfig(pidl_feature_layer=cfg["feature_layer"])  # type: ignore[arg-type]
    init_model  = build_model(num_classes=num_classes, config=model_cfg)
    init_params = ndarrays_to_parameters(get_model_parameters(init_model))
    del init_model  # free memory before the simulation loop starts

    # ── ExperimentLogger — writes fl_rounds.csv, per_class_metrics.csv, etc.
    # Imported inside server_fn to keep the module importable even when
    # metrics / utils packages are not yet on sys.path.
    from metrics.calibration_metrics import compute_calibration_metrics
    from metrics.classification_metrics import compute_classification_metrics
    from utils.logging_utils import ExperimentLogger

    exp_logger = ExperimentLogger(
        log_dir=log_dir,
        dataset_name=cfg["dataset_name"],
        num_clients=cfg["num_clients"],
    )
    exp_logger.save_config(cfg)
    if "dataset_summary" in data:
        exp_logger.save_dataset_summary(data["dataset_summary"])

    # ── Server-side evaluate_fn ───────────────────────────────────────────
    # Called by LoggingFedAvg.evaluate() after every aggregation step.
    # Evaluates the global model, computes comprehensive classification and
    # calibration metrics, logs them via ExperimentLogger (→ fl_rounds.csv,
    # per_class_metrics.csv, fl_eval.json), and returns (loss, metrics) to
    # the Flower framework.
    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict,
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        model = build_model(num_classes=num_classes, config=model_cfg).to(device)
        set_model_parameters(model, list(parameters))

        t0 = _time.time()
        results = evaluate_full(model, global_test_loader, device, num_classes)
        infer_time = _time.time() - t0
        del model  # avoid accumulating GPU memory across rounds

        clf = compute_classification_metrics(
            y_true=results["all_labels"],
            y_prob=results["all_probs"],
            class_names=class_names if class_names else None,
        )
        cal = compute_calibration_metrics(
            y_prob=results["all_probs"],
            y_true=results["all_labels"],
        )

        # ce_loss / reg_loss are filled at atexit time from strategy history
        exp_logger.log_round(
            server_round=server_round,
            clf_result=clf,
            cal_metrics=cal,
            global_test_loss=float(results["loss"]),
            inference_time_sec=infer_time,
        )

        return float(results["loss"]), {
            "accuracy":           float(results["accuracy"]),
            "num_samples":        int(results["num_samples"]),
            # Surface key metrics so LoggingFedAvg can include them in its JSONL
            "f1_macro":           float(clf["flat"].get("f1_macro",         0.0)),
            "balanced_accuracy":  float(clf["flat"].get("balanced_accuracy", 0.0)),
            "ece":                float(cal.get("ece",                       0.0)),
        }

    # ── Strategy ─────────────────────────────────────────────────────────
    nc = cfg["num_clients"]
    mf = cfg["min_fit_clients"]

    # Mutable holder so the atexit handler can reach the strategy after
    # it is created below.
    _strategy_ref: list = []

    strategy = LoggingFedAvg(
        log_path=log_path,
        log_dir=log_dir,
        # Server-side evaluation after every round
        evaluate_fn=evaluate_fn,
        # FedAvg parameters
        initial_parameters=init_params,
        fraction_fit=1.0,            # use all available clients every round
        fraction_evaluate=1.0,
        min_fit_clients=mf,
        min_evaluate_clients=mf,
        min_available_clients=nc,
        fit_metrics_aggregation_fn=None,      # handled by aggregate_fit override
        evaluate_metrics_aggregation_fn=None,
    )

    _strategy_ref.append(strategy)

    # ── atexit: merge training stats + finalize all output files ──────────
    # Runs automatically when `flwr run .` exits, ensuring fl_rounds.csv,
    # fl_summary.json etc. are always written even if the process crashes
    # mid-run (partial records are still useful).
    def _finalize_experiment() -> None:
        if _strategy_ref:
            hist = _strategy_ref[0].get_history()
            if hist:
                # Back-fill ce_loss / reg_loss from per-round strategy history
                _backfill_training_losses(exp_logger, hist)
                exp_logger.log_client_rounds_from_history(hist)
        exp_logger.finalize()

    atexit.register(_finalize_experiment)

    # ── Server config ─────────────────────────────────────────────────────
    server_config = ServerConfig(num_rounds=cfg["num_server_rounds"])

    # ── SecAgg+ workflow ──────────────────────────────────────────────────
    # Parameters come from run_config (set in pyproject.toml).  Defaults
    # fall back to num_clients / min_fit_clients if not explicitly provided.
    # No differential privacy noise is added at any point.
    num_shares               = cfg.get("secagg_num_shares",               nc)
    reconstruction_threshold = cfg.get("secagg_reconstruction_threshold", max(1, mf))
    max_weight               = cfg.get("secagg_max_weight",               1048575)

    if _SECAGG_AVAILABLE:
        print(
            f"\n[SecAgg+] Enabled\n"
            f"  num_shares               = {num_shares}\n"
            f"  reconstruction_threshold = {reconstruction_threshold}\n"
            f"  max_weight               = {max_weight}\n"
            f"  (No DP noise)\n"
        )
        # Flower 1.9+ uses max_weight; older builds use clipping_range +
        # quantization_range.  Try the newer API first and fall back gracefully.
        try:
            workflow = SecAggPlusWorkflow(
                num_shares=num_shares,
                reconstruction_threshold=reconstruction_threshold,
                max_weight=max_weight,
            )
        except TypeError:
            # Older Flower build — fall back to clipping_range + quantization_range
            workflow = SecAggPlusWorkflow(
                num_shares=num_shares,
                reconstruction_threshold=reconstruction_threshold,
                clipping_range=8.0,
                quantization_range=4,
            )
    elif DefaultWorkflow is not None:
        print("[SecAgg+] Not available — using DefaultWorkflow (no SecAgg+).")
        workflow = DefaultWorkflow()
    else:
        workflow = None

    print(
        f"\n{'='*62}\n"
        f"  Federated run starting\n"
        f"  Dataset      : {cfg['dataset_name']}\n"
        f"  Clients      : {nc}   Rounds: {cfg['num_server_rounds']}\n"
        f"  PIDL layer   : {cfg['feature_layer']}  "
        f"λ={cfg['lambda_pm']}  type={cfg['regularizer_type']}\n"
        f"  SecAgg+      : {_SECAGG_AVAILABLE}\n"
        f"  Log dir      : {log_dir}\n"
        f"{'='*62}\n"
    )

    components = ServerAppComponents(strategy=strategy, config=server_config)
    if workflow is not None:
        components.workflow = workflow

    return components


# ---------------------------------------------------------------------------
# Module-level ServerApp — referenced by pyproject.toml
# ---------------------------------------------------------------------------

app = ServerApp(server_fn=server_fn)
