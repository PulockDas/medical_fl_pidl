"""
Experiment output manager for federated PIDL medical image experiments.

``ExperimentLogger`` is the single class responsible for writing every output
file needed for later tables, plots, and paper results.

Output files
------------
::

  <log_dir>/
    config.json              — experiment hyperparameters snapshot
    dataset_summary.json     — class counts, partition stats, image root
    fl_rounds.csv            — one row per FL round (global metrics)
    fl_clients.csv           — one row per (round, client_id)
    fl_eval.json             — complete per-round evaluation records (JSON array)
    fl_summary.json          — best/final metrics + final confusion matrix
    per_class_metrics.csv    — one row per (round, class)
    final_model.pth          — model state dict (optional)

Typical usage from a notebook
------------------------------
::

    logger = ExperimentLogger(
        log_dir="results/run_001",
        dataset_name="brain_tumor_mri",
        num_clients=3,
    )
    logger.save_config(run_config_dict)
    logger.save_dataset_summary(data["dataset_summary"])

    # After flwr.run_simulation finishes, iterate over strategy.get_history():
    for rec in strategy.get_history():
        server_round = rec["round"]

        # Evaluate the final global model on the test set
        eval_out  = evaluate_full(model, test_loader, device, num_classes)
        clf       = compute_classification_metrics(
                        y_true=eval_out["all_labels"],
                        y_prob=eval_out["all_probs"],
                        class_names=class_names)
        cal       = compute_calibration_metrics(
                        y_prob=eval_out["all_probs"],
                        y_true=eval_out["all_labels"])

        logger.log_round(
            server_round=server_round,
            clf_result=clf,
            cal_metrics=cal,
            global_test_loss=rec.get("server_loss", 0.0),
            ce_loss=rec.get("train_ce_loss", 0.0),
            reg_loss=rec.get("train_pidl_loss", 0.0),
            training_time_sec=rec.get("elapsed_seconds", 0.0),
        )
        logger.log_client_round(server_round, client_metrics_for_this_round)

    logger.save_model(model)   # optional
    logger.finalize()

For a simpler post-hoc workflow (single final evaluation):
::

    logger.log_final_eval(clf_result, cal_metrics, strategy_history)
    logger.save_model(model)
    logger.finalize()
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Column definitions — single source of truth for CSV headers
# ---------------------------------------------------------------------------

_ROUNDS_COLS: list[str] = [
    "dataset_name", "num_clients", "round",
    "global_test_acc", "balanced_accuracy", "global_test_loss",
    "ce_loss", "reg_loss",
    "f1_macro", "f1_micro", "f1_weighted",
    "precision_macro", "recall_macro", "specificity_macro", "sensitivity_macro",
    "roc_auc_macro", "pr_auc_macro",
    "mean_confidence", "mean_entropy", "ece",
    "inference_time_sec", "training_time_sec", "aggregation_time_sec",
]

_CLIENTS_COLS: list[str] = [
    "dataset_name", "num_clients", "round",
    "client_id", "num_samples",
    "train_loss", "train_accuracy",
    "train_ce_loss", "train_reg_loss",
    "train_time_sec", "class_distribution",
]

_PER_CLASS_COLS: list[str] = [
    "dataset_name", "num_clients", "round",
    "class_name", "precision", "recall", "f1", "support",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nan_safe(v: Any, default: float = float("nan")) -> Any:
    """Return float('nan') → None so JSON serialises cleanly; otherwise pass through."""
    if isinstance(v, float) and (v != v):   # NaN check
        return None
    return v


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback: ndarray → list, nan → None, anything else → str."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (f != f) else f   # NaN → None
    if isinstance(obj, float) and (obj != obj):
        return None
    return str(obj)


def _write_row(
    path: Path,
    row: dict[str, Any],
    columns: list[str],
    write_header: bool,
) -> None:
    """Append one CSV row, writing the header first if required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        # Replace Python NaN with empty string for CSV readability
        safe_row = {
            k: ("" if isinstance(v, float) and (v != v) else v)
            for k, v in row.items()
        }
        writer.writerow(safe_row)


def _best_round(history: list[dict], metric: str) -> tuple[float, int]:
    """Return (best_value, best_round) for a metric across the history list."""
    best_val = float("-inf")
    best_rnd = -1
    for rec in history:
        v = rec.get(metric)
        if v is not None and not (isinstance(v, float) and v != v):
            if float(v) > best_val:
                best_val = float(v)
                best_rnd = rec.get("round", -1)
    return best_val, best_rnd


# ---------------------------------------------------------------------------
# Run directory factory
# ---------------------------------------------------------------------------


def make_run_dir(
    results_dir: Union[str, Path],
    experiment_name: str,
    run_name: Optional[str] = None,
) -> Path:
    """Create and return a unique timestamped run directory.

    Args:
        results_dir:     Top-level results directory (e.g. ``"results/"``).
        experiment_name: Short name used in the directory name.
        run_name:        If given, use this exact subdirectory name instead of
                         generating a timestamped one.

    Returns:
        ``Path`` to the newly created run directory.
    """
    results_dir = Path(results_dir)
    if run_name:
        run_dir = results_dir / run_name
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_dir / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------


class ExperimentLogger:
    """Central experiment output manager.

    Writes all CSV, JSON, and model checkpoint files for one federated run.
    All public methods are safe to call even if their inputs contain NaN.

    Args:
        log_dir:      Directory where all output files are written.
        dataset_name: Dataset name string embedded in every CSV row.
        num_clients:  Number of FL clients embedded in every CSV row.
        append:       If True, open existing CSVs in append mode (useful for
                      resuming a run).  If False, overwrite (default).
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        dataset_name: str,
        num_clients: int,
        append: bool = False,
    ) -> None:
        self.log_dir      = Path(log_dir)
        self.dataset_name = dataset_name
        self.num_clients  = num_clients

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.path_rounds_csv    = self.log_dir / "fl_rounds.csv"
        self.path_clients_csv   = self.log_dir / "fl_clients.csv"
        self.path_eval_json     = self.log_dir / "fl_eval.json"
        self.path_summary_json  = self.log_dir / "fl_summary.json"
        self.path_per_class_csv = self.log_dir / "per_class_metrics.csv"

        # In-memory accumulators
        self._rounds_history:     list[dict[str, Any]] = []
        self._eval_history:       list[dict[str, Any]] = []
        self._total_train_sec:    float = 0.0
        self._total_infer_sec:    float = 0.0
        self._last_cm:            Optional[np.ndarray] = None

        # Track whether CSV headers have been written
        if append:
            self._rounds_header_written    = self.path_rounds_csv.exists()
            self._clients_header_written   = self.path_clients_csv.exists()
            self._per_class_header_written = self.path_per_class_csv.exists()
        else:
            # Wipe existing files so headers are re-written on first log_round
            for p in [self.path_rounds_csv, self.path_clients_csv,
                      self.path_per_class_csv]:
                if p.exists():
                    p.unlink()
            self._rounds_header_written    = False
            self._clients_header_written   = False
            self._per_class_header_written = False

        print(f"[ExperimentLogger] Output directory: {self.log_dir}")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_) -> None:
        self.finalize()

    # ------------------------------------------------------------------
    # Primary logging: one FL round
    # ------------------------------------------------------------------

    def log_round(
        self,
        server_round: int,
        clf_result: dict[str, Any],
        cal_metrics: Optional[dict[str, float]] = None,
        global_test_loss: float = 0.0,
        ce_loss: float = 0.0,
        reg_loss: float = 0.0,
        training_time_sec: float = 0.0,
        inference_time_sec: float = 0.0,
        aggregation_time_sec: float = 0.0,
    ) -> None:
        """Log one round of global evaluation results.

        Args:
            server_round:        FL round number (1-indexed).
            clf_result:          Return value of
                                 ``compute_classification_metrics()``.
                                 Must have ``"flat"``, ``"per_class"``, and
                                 ``"confusion_matrix"`` keys.
            cal_metrics:         Optional return value of
                                 ``compute_calibration_metrics()``.  If omitted,
                                 ECE is taken from ``clf_result["flat"]`` if
                                 present.
            global_test_loss:    CE loss on the global test set.
            ce_loss:             Training CE loss (from strategy history).
            reg_loss:            Training PIDL regularization loss.
            training_time_sec:   Wall-clock seconds for client training this round.
            inference_time_sec:  Wall-clock seconds for server evaluation.
            aggregation_time_sec: Wall-clock seconds for parameter aggregation.
        """
        flat      = clf_result.get("flat", {})
        per_class = clf_result.get("per_class", {})
        cm        = clf_result.get("confusion_matrix")
        if cm is not None:
            self._last_cm = cm

        # ECE may come from classification result or separate cal_metrics
        ece = flat.get("ece", (cal_metrics or {}).get("ece", float("nan")))

        # ── fl_rounds.csv row ────────────────────────────────────────────
        round_row: dict[str, Any] = {
            "dataset_name":        self.dataset_name,
            "num_clients":         self.num_clients,
            "round":               server_round,
            "global_test_acc":     flat.get("accuracy",          float("nan")),
            "balanced_accuracy":   flat.get("balanced_accuracy", float("nan")),
            "global_test_loss":    global_test_loss,
            "ce_loss":             ce_loss,
            "reg_loss":            reg_loss,
            "f1_macro":            flat.get("f1_macro",          float("nan")),
            "f1_micro":            flat.get("f1_micro",          float("nan")),
            "f1_weighted":         flat.get("f1_weighted",       float("nan")),
            "precision_macro":     flat.get("precision_macro",   float("nan")),
            "recall_macro":        flat.get("recall_macro",      float("nan")),
            "specificity_macro":   flat.get("specificity_macro", float("nan")),
            "sensitivity_macro":   flat.get("sensitivity_macro", float("nan")),
            "roc_auc_macro":       flat.get("roc_auc_macro",     float("nan")),
            "pr_auc_macro":        flat.get("pr_auc_macro",      float("nan")),
            "mean_confidence":     flat.get("mean_confidence",   float("nan")),
            "mean_entropy":        flat.get("mean_entropy",      float("nan")),
            "ece":                 ece,
            "inference_time_sec":  inference_time_sec,
            "training_time_sec":   training_time_sec,
            "aggregation_time_sec": aggregation_time_sec,
        }
        _write_row(
            self.path_rounds_csv, round_row, _ROUNDS_COLS,
            write_header=not self._rounds_header_written,
        )
        self._rounds_header_written = True
        self._rounds_history.append(round_row)

        # ── per_class_metrics.csv rows ───────────────────────────────────
        for class_name, cm_vals in per_class.items():
            pc_row: dict[str, Any] = {
                "dataset_name": self.dataset_name,
                "num_clients":  self.num_clients,
                "round":        server_round,
                "class_name":   class_name,
                "precision":    cm_vals.get("precision",  float("nan")),
                "recall":       cm_vals.get("recall",     float("nan")),
                "f1":           cm_vals.get("f1",         float("nan")),
                "support":      cm_vals.get("support",    0),
            }
            _write_row(
                self.path_per_class_csv, pc_row, _PER_CLASS_COLS,
                write_header=not self._per_class_header_written,
            )
            self._per_class_header_written = True

        # ── fl_eval rich record (in-memory; flushed in finalize) ─────────
        eval_record: dict[str, Any] = {
            "round":           server_round,
            "dataset_name":    self.dataset_name,
            "num_clients":     self.num_clients,
            "flat_metrics":    {k: _nan_safe(v) for k, v in flat.items()},
            "calibration":     {k: _nan_safe(v) for k, v in (cal_metrics or {}).items()},
            "per_class":       {
                cn: {mk: _nan_safe(mv) for mk, mv in cv.items()}
                for cn, cv in per_class.items()
            },
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "global_test_loss": _nan_safe(global_test_loss),
        }
        self._eval_history.append(eval_record)

        # ── Running totals for summary ────────────────────────────────────
        self._total_train_sec += training_time_sec
        self._total_infer_sec += inference_time_sec

    # ------------------------------------------------------------------
    # Per-client metrics for one round
    # ------------------------------------------------------------------

    def log_client_round(
        self,
        server_round: int,
        client_metrics: list[dict[str, Any]],
    ) -> None:
        """Log per-client training metrics for one FL round.

        Args:
            server_round:    FL round number.
            client_metrics:  List of per-client metric dicts.  Each dict
                             should have keys matching the Flower fit metrics
                             returned by ``MedicalFLClient.fit()``:
                             ``client_id``, ``num_examples``,
                             ``train_loss``, ``train_accuracy``,
                             ``train_ce_loss``, ``train_pidl_loss``,
                             optionally ``train_time_sec``,
                             optionally ``class_distribution``.
        """
        for cm in client_metrics:
            row: dict[str, Any] = {
                "dataset_name":   self.dataset_name,
                "num_clients":    self.num_clients,
                "round":          server_round,
                "client_id":      cm.get("client_id",         "?"),
                "num_samples":    cm.get("num_examples",       cm.get("num_samples", 0)),
                "train_loss":     cm.get("train_loss",         float("nan")),
                "train_accuracy": cm.get("train_accuracy",     float("nan")),
                "train_ce_loss":  cm.get("train_ce_loss",      float("nan")),
                # train_pidl_loss is renamed to train_reg_loss in CSV
                "train_reg_loss": cm.get("train_pidl_loss",    cm.get("train_reg_loss", float("nan"))),
                "train_time_sec": cm.get("train_time_sec",     float("nan")),
                # class_distribution may be a dict; serialise as JSON string
                "class_distribution": json.dumps(cm["class_distribution"])
                    if "class_distribution" in cm and cm["class_distribution"]
                    else "",
            }
            _write_row(
                self.path_clients_csv, row, _CLIENTS_COLS,
                write_header=not self._clients_header_written,
            )
            self._clients_header_written = True

    # ------------------------------------------------------------------
    # Convenience: batch-log from strategy history
    # ------------------------------------------------------------------

    def log_client_rounds_from_history(
        self,
        strategy_history: list[dict[str, Any]],
    ) -> None:
        """Batch-log client metrics from ``LoggingFedAvg.get_history()``.

        Each history record must have been emitted by ``LoggingFedAvg`` which
        stores weighted-average train metrics (not per-client).  This method
        creates a single synthetic "aggregate" client row per round with
        ``client_id="aggregate"`` so the CSV is still populated.

        For true per-client rows, call ``log_client_round()`` inside the
        simulation's ``client_fn`` or from a custom strategy hook.

        Args:
            strategy_history: List of per-round dicts from
                              ``LoggingFedAvg.get_history()``.
        """
        for rec in strategy_history:
            synthetic = [{
                "client_id":        "aggregate",
                "num_examples":     rec.get("num_clients_fit", self.num_clients),
                "train_loss":       rec.get("train_loss",       float("nan")),
                "train_accuracy":   rec.get("train_accuracy",   float("nan")),
                "train_ce_loss":    rec.get("train_ce_loss",    float("nan")),
                "train_pidl_loss":  rec.get("train_pidl_loss",  float("nan")),
            }]
            self.log_client_round(rec.get("round", 0), synthetic)

    # ------------------------------------------------------------------
    # Convenience: single final evaluation (post-hoc logging)
    # ------------------------------------------------------------------

    def log_final_eval(
        self,
        clf_result: dict[str, Any],
        cal_metrics: Optional[dict[str, float]] = None,
        strategy_history: Optional[list[dict[str, Any]]] = None,
        server_round: int = 1,
        global_test_loss: float = 0.0,
    ) -> None:
        """Log a single final evaluation result (post-hoc workflow).

        Use this when you only have one evaluation (at the end of training),
        rather than evaluating the model after every round.

        Args:
            clf_result:       ``compute_classification_metrics()`` result.
            cal_metrics:      ``compute_calibration_metrics()`` result.
            strategy_history: Full strategy history (used to fill ce_loss /
                              reg_loss / timing from the last round).
            server_round:     Round number to label this row (default 1 if
                              single evaluation, or the last round number).
            global_test_loss: CE loss on the global test set.
        """
        # Extract last-round training stats from strategy history if available
        ce_loss = reg_loss = train_sec = 0.0
        if strategy_history:
            last = strategy_history[-1]
            ce_loss   = float(last.get("train_ce_loss",   0.0))
            reg_loss  = float(last.get("train_pidl_loss", 0.0))
            train_sec = float(last.get("elapsed_seconds", 0.0))
            server_round = int(last.get("round", server_round))

        self.log_round(
            server_round=server_round,
            clf_result=clf_result,
            cal_metrics=cal_metrics,
            global_test_loss=global_test_loss,
            ce_loss=ce_loss,
            reg_loss=reg_loss,
            training_time_sec=train_sec,
        )

        if strategy_history:
            self.log_client_rounds_from_history(strategy_history)

    # ------------------------------------------------------------------
    # One-time saves
    # ------------------------------------------------------------------

    def save_config(
        self,
        config: Union[dict[str, Any], Any],
    ) -> Path:
        """Serialise the experiment config to ``config.json``.

        Accepts a plain dict, an ``ExperimentConfig`` dataclass instance
        (via ``asdict``), or any object with a ``to_dict()`` / ``__dict__``
        method.

        Args:
            config: Experiment configuration to save.

        Returns:
            Path to the saved file.
        """
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = vars(config)

        path = self.log_dir / "config.json"
        path.write_text(
            json.dumps(config_dict, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(f"[ExperimentLogger] Config saved → {path}")
        return path

    def save_dataset_summary(
        self,
        summary: dict[str, Any],
    ) -> Path:
        """Write the dataset summary dict to ``dataset_summary.json``.

        Args:
            summary: Dict returned by ``build_federated_dataloaders()``
                     under the ``"dataset_summary"`` key, or any compatible dict.

        Returns:
            Path to the saved file.
        """
        path = self.log_dir / "dataset_summary.json"
        path.write_text(
            json.dumps(summary, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(f"[ExperimentLogger] Dataset summary saved → {path}")
        return path

    def save_model(
        self,
        model: Any,
        filename: str = "final_model.pth",
    ) -> Path:
        """Save the model state dict to ``final_model.pth``.

        Args:
            model:    A ``torch.nn.Module`` instance.
            filename: Override the output filename if desired.

        Returns:
            Path to the saved file.
        """
        import torch  # local import keeps the module importable without torch

        path = self.log_dir / filename
        torch.save(model.state_dict(), path)
        print(f"[ExperimentLogger] Model checkpoint saved → {path}")
        return path

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """Write ``fl_eval.json`` and ``fl_summary.json``, print a summary table.

        Should be called once, after all rounds have been logged.
        Safe to call multiple times (subsequent calls overwrite the files).

        Returns:
            The ``fl_summary.json`` dict for inspection in the notebook.
        """
        # ── fl_eval.json — full per-round eval array ──────────────────────
        self.path_eval_json.write_text(
            json.dumps(self._eval_history, indent=2, default=_json_default),
            encoding="utf-8",
        )

        # ── fl_summary.json ───────────────────────────────────────────────
        summary = self._build_summary()
        self.path_summary_json.write_text(
            json.dumps(summary, indent=2, default=_json_default),
            encoding="utf-8",
        )

        self._print_summary(summary)

        print(
            f"\n[ExperimentLogger] All outputs written to: {self.log_dir}\n"
            f"  fl_rounds.csv        ({len(self._rounds_history)} rounds)\n"
            f"  per_class_metrics.csv\n"
            f"  fl_eval.json\n"
            f"  fl_summary.json"
        )
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict[str, Any]:
        """Compute best/final metrics across all logged rounds."""
        h = self._rounds_history

        if not h:
            return {
                "dataset_name": self.dataset_name,
                "num_clients":  self.num_clients,
                "num_rounds":   0,
            }

        last = h[-1]

        best_acc,   best_acc_rnd   = _best_round(h, "global_test_acc")
        best_f1,    best_f1_rnd    = _best_round(h, "f1_macro")
        best_bal,   best_bal_rnd   = _best_round(h, "balanced_accuracy")
        best_auc,   best_auc_rnd   = _best_round(h, "roc_auc_macro")
        best_prauc, best_prauc_rnd = _best_round(h, "pr_auc_macro")

        summary: dict[str, Any] = {
            "dataset_name":              self.dataset_name,
            "num_clients":               self.num_clients,
            "num_rounds":                len(h),
            # Best metrics
            "best_accuracy":             _nan_safe(best_acc),
            "best_accuracy_round":       best_acc_rnd,
            "final_accuracy":            _nan_safe(last.get("global_test_acc")),
            "best_balanced_accuracy":    _nan_safe(best_bal),
            "best_balanced_accuracy_round": best_bal_rnd,
            "final_balanced_accuracy":   _nan_safe(last.get("balanced_accuracy")),
            "best_macro_f1":             _nan_safe(best_f1),
            "best_macro_f1_round":       best_f1_rnd,
            "final_macro_f1":            _nan_safe(last.get("f1_macro")),
            "best_roc_auc_macro":        _nan_safe(best_auc),
            "best_roc_auc_macro_round":  best_auc_rnd,
            "final_roc_auc_macro":       _nan_safe(last.get("roc_auc_macro")),
            "best_pr_auc_macro":         _nan_safe(best_prauc),
            "best_pr_auc_macro_round":   best_prauc_rnd,
            "final_pr_auc_macro":        _nan_safe(last.get("pr_auc_macro")),
            "final_ece":                 _nan_safe(last.get("ece")),
            "final_mean_confidence":     _nan_safe(last.get("mean_confidence")),
            "final_mean_entropy":        _nan_safe(last.get("mean_entropy")),
            # Timing
            "total_training_time_sec":   round(self._total_train_sec, 2),
            "total_inference_time_sec":  round(self._total_infer_sec, 2),
            # Final confusion matrix (from last round with available data)
            "final_confusion_matrix": (
                self._last_cm.tolist() if self._last_cm is not None else None
            ),
        }
        return summary

    def _print_summary(self, summary: dict[str, Any]) -> None:
        """Print a formatted experiment summary table."""
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  EXPERIMENT SUMMARY")
        print(f"  Dataset:  {self.dataset_name}   Clients: {self.num_clients}   "
              f"Rounds: {summary.get('num_rounds', '?')}")
        print(f"{sep}")

        rows = [
            ("Best Accuracy",          summary.get("best_accuracy"),          summary.get("best_accuracy_round")),
            ("Final Accuracy",         summary.get("final_accuracy"),          None),
            ("Best Balanced Acc",      summary.get("best_balanced_accuracy"),  summary.get("best_balanced_accuracy_round")),
            ("Final Balanced Acc",     summary.get("final_balanced_accuracy"), None),
            ("Best Macro F1",          summary.get("best_macro_f1"),          summary.get("best_macro_f1_round")),
            ("Final Macro F1",         summary.get("final_macro_f1"),         None),
            ("Best ROC-AUC",           summary.get("best_roc_auc_macro"),     summary.get("best_roc_auc_macro_round")),
            ("Best PR-AUC",            summary.get("best_pr_auc_macro"),      summary.get("best_pr_auc_macro_round")),
            ("Final ECE",              summary.get("final_ece"),              None),
            ("Train time (total)",     summary.get("total_training_time_sec"), None),
            ("Infer time (total)",     summary.get("total_inference_time_sec"), None),
        ]

        for label, val, rnd in rows:
            if val is None:
                continue
            if isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = str(val)
            rnd_str = f"  (round {rnd})" if rnd is not None and rnd > 0 else ""
            print(f"  {label:<25} {cell}{rnd_str}")

        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Standalone reader helpers (backward compatible)
# ---------------------------------------------------------------------------


def load_fl_rounds_csv(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Load ``fl_rounds.csv`` into a list of typed dicts.

    Args:
        path: Path to ``fl_rounds.csv``.

    Returns:
        List of per-round metric dicts.  Numeric columns are cast to float.
    """
    import csv as _csv

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v != "" else float("nan")
                except (ValueError, TypeError):
                    parsed[k] = v
            records.append(parsed)
    return records


def load_fl_summary(path: Union[str, Path]) -> dict[str, Any]:
    """Load ``fl_summary.json`` into a dict.

    Args:
        path: Path to ``fl_summary.json``.

    Returns:
        Summary dict.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_fl_eval(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Load ``fl_eval.json`` (JSON array) into a list of per-round dicts.

    Args:
        path: Path to ``fl_eval.json``.

    Returns:
        List of per-round evaluation records.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Backward-compatible standalone functions
# ---------------------------------------------------------------------------


def save_config_snapshot(config_dict: dict[str, Any], run_dir: Union[str, Path]) -> Path:
    """Write a config dict to ``config.json`` in ``run_dir``.

    Backward-compatible wrapper around ``ExperimentLogger.save_config()``.
    """
    path = Path(run_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_dict, indent=2, default=str), encoding="utf-8")
    return path


def print_experiment_summary(
    final_metrics: dict[str, Any],
    config_dict: Optional[dict[str, Any]] = None,
) -> None:
    """Print a compact experiment summary.  Backward-compatible utility."""
    sep = "=" * 55
    print(f"\n{sep}\n  EXPERIMENT SUMMARY")
    if config_dict:
        print(f"  Dataset: {config_dict.get('dataset_name', config_dict.get('dataset', '?'))}")
        print(f"  Clients: {config_dict.get('num_clients', '?')}")
    print(sep)
    keys = [
        ("accuracy",          "Accuracy"),
        ("balanced_accuracy", "Balanced Acc"),
        ("f1_macro",          "F1 Macro"),
        ("roc_auc_macro",     "ROC-AUC"),
        ("pr_auc_macro",      "PR-AUC"),
        ("ece",               "ECE"),
        ("brier_score",       "Brier Score"),
        ("loss",              "Test Loss"),
    ]
    for key, label in keys:
        val = final_metrics.get(f"test_{key}", final_metrics.get(key))
        if val is not None:
            print(f"  {label:<20} {val:.4f}" if isinstance(val, float) else f"  {label:<20} {val}")
    print(f"{sep}\n")
