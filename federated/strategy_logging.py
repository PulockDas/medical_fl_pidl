"""
LoggingFedAvg — FedAvg strategy with per-round metric logging and
optional server-side global model evaluation.

Per-round record layout (one JSON line in round_metrics.jsonl)
--------------------------------------------------------------
{
  "round"              : int,
  "elapsed_seconds"    : float,
  "num_clients_fit"    : int,
  "num_failures_fit"   : int,
  "train_loss"         : float,   # weighted average across clients
  "train_ce_loss"      : float,
  "train_pidl_loss"    : float,
  "train_accuracy"     : float,
  "num_clients_eval"   : int,
  "num_failures_eval"  : int,
  "val_loss"           : float,   # weighted average across clients
  "val_accuracy"       : float,
  "server_loss"        : float,   # server-side global test loss (if evaluate_fn set)
  "server_accuracy"    : float,   # server-side global test accuracy
  "server_num_samples" : int
}

A final ``summary.json`` is written to ``log_dir`` when ``close()`` is called
(or the object is garbage-collected). It records best/last metrics and the
full history list.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Union

from flwr.common import (
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------


def _simulate_secagg_time(
    results: list[tuple[ClientProxy, FitRes]],
    num_shares: int,
) -> float:
    """Measure wall-clock time for SecAgg+ mask operations on the actual parameters.

    Simulates the two dominant SecAgg+ costs per round:
      1. *Mask generation* — each client produces ``num_shares - 1`` additive
         random masks (one per share), one per parameter tensor.
      2. *Mask summation* — the server cancels out the masks by summing all
         masked updates (equivalent to the reconstruction step).

    This is run on the real parameter tensors so the timing reflects actual
    model size.  It does NOT simulate cryptographic key exchange (a small
    constant overhead independent of model size).

    Args:
        results:    Flower ``(ClientProxy, FitRes)`` result list from this round.
        num_shares: SecAgg+ ``num_shares`` config value.

    Returns:
        Wall-clock seconds consumed by the simulated SecAgg+ operations.
    """
    import numpy as _np

    if not results or num_shares <= 1:
        return 0.0

    t0 = time.time()
    for _client, fit_res in results:
        tensors = fit_res.parameters.tensors
        for tensor_bytes in tensors:
            arr = _np.frombuffer(tensor_bytes, dtype=_np.float32)
            # Generate num_shares-1 random additive masks (client-side cost)
            masks = [_np.random.rand(arr.size).astype(_np.float32)
                     for _ in range(num_shares - 1)]
            # Reconstruct masked update (server-side cost)
            masked = arr.copy()
            for m in masks:
                masked = masked + m      # noqa: augmented assignment on view fails
            del masks, masked
    return time.time() - t0


def _weighted_average(
    results: list[tuple[ClientProxy, Union[FitRes, EvaluateRes]]],
    metric_keys: list[str],
    weight_key: str = "num_examples",
) -> dict[str, float]:
    """Compute weighted average of scalar metrics across client results.

    Args:
        results:     Flower ``(ClientProxy, *Res)`` result list.
        metric_keys: Which keys to aggregate from ``result.metrics``.
        weight_key:  The metric key used as weight (default ``num_examples``).

    Returns:
        Dict mapping each key to its weighted-average value.
    """
    totals: dict[str, float] = {k: 0.0 for k in metric_keys}
    total_weight = 0

    for _client, res in results:
        metrics = res.metrics or {}
        weight = int(metrics.get(weight_key, res.num_examples))
        total_weight += weight
        for k in metric_keys:
            if k in metrics:
                totals[k] += float(metrics[k]) * weight

    if total_weight == 0:
        return {k: 0.0 for k in metric_keys}
    return {k: v / total_weight for k, v in totals.items()}


# ---------------------------------------------------------------------------
# LoggingFedAvg
# ---------------------------------------------------------------------------


class LoggingFedAvg(FedAvg):
    """FedAvg strategy with per-round metric logging and server-side eval.

    Extends ``FedAvg`` with three logging features:

    1. **Fit metrics** — weighted average of client training metrics
       (loss, CE loss, PIDL loss, accuracy) written in ``aggregate_fit``.
    2. **Server eval** — if ``evaluate_fn`` is set, the strategy's
       ``evaluate()`` method is overridden to capture and log the global
       model's test performance after every round.
    3. **Client eval** — weighted average of client validation metrics
       written in ``aggregate_evaluate``. The full round record (fit +
       server eval + client eval) is written here as a single JSONL line.

    Persistence:
        - ``<log_dir>/round_metrics.jsonl`` — one JSON object per round.
        - ``<log_dir>/summary.json``        — best/last metrics + full history,
          written by ``close()`` or ``__del__``.

    Args:
        log_path: Path to the per-round JSONL log file.
        log_dir:  Directory for ``summary.json``. Defaults to ``log_path.parent``.
        **kwargs: All keyword arguments forwarded to ``FedAvg``.
    """

    def __init__(
        self,
        log_path: Union[str, Path],
        log_dir: Union[str, Path, None] = None,
        num_shares: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.log_path    = Path(log_path)
        self.log_dir     = Path(log_dir) if log_dir is not None else self.log_path.parent
        self._num_shares = num_shares     # used by _simulate_secagg_time
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Append mode: multiple runs in the same session accumulate cleanly.
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        self._history:              list[dict[str, Any]] = []
        self._round_start_time:     float = 0.0
        self._current_fit_metrics:  dict[str, Any] = {}
        self._current_server_eval:  dict[str, Any] = {}

        print(f"[LoggingFedAvg] Logging to: {self.log_path}")

    # ------------------------------------------------------------------
    # FedAvg overrides
    # ------------------------------------------------------------------

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Record wall-clock time at the start of each round."""
        self._round_start_time    = time.time()
        self._current_server_eval = {}          # reset server eval for this round
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model updates and capture weighted client training metrics.

        Also measures SecAgg+ overhead: simulates the mask-generation and
        mask-summation operations on the actual parameter arrays.  This gives
        honest wall-clock timing without requiring Flower's SecAggPlusWorkflow
        (which is incompatible with ``run_simulation`` in Flower 1.29).
        """
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        secagg_time = _simulate_secagg_time(results, self._num_shares)

        fit_metrics = _weighted_average(
            results,
            ["train_loss", "train_ce_loss", "train_pidl_loss", "train_accuracy"],
        )
        self._current_fit_metrics = {
            "server_round":        server_round,
            "num_clients_fit":     len(results),
            "num_failures_fit":    len(failures),
            "secagg_overhead_sec": round(secagg_time, 4),
            **fit_metrics,
        }
        return aggregated_params, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate the global model on the server's test set.

        Called by the Flower workflow after ``aggregate_fit``. Delegates to the
        ``evaluate_fn`` set in the constructor and stores the result so that
        ``aggregate_evaluate`` can include it in the combined round record.
        """
        result = super().evaluate(server_round, parameters)

        if result is not None:
            loss, metrics = result
            self._current_server_eval = {
                "server_loss":        round(loss, 6),
                "server_accuracy":    round(float(metrics.get("accuracy", 0.0)), 6),
                "server_num_samples": int(metrics.get("num_samples", 0)),
            }
            acc_pct = self._current_server_eval["server_accuracy"] * 100
            print(
                f"[Server Eval] Round {server_round:>3} | "
                f"Loss: {loss:.4f}  Acc: {acc_pct:5.2f}%  "
                f"N={self._current_server_eval['server_num_samples']}"
            )

        return result

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate client eval results and write the complete round record."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        eval_metrics = _weighted_average(results, ["val_loss", "val_accuracy"])

        elapsed = time.time() - self._round_start_time

        round_record: dict[str, Any] = {
            "round":           server_round,
            "elapsed_seconds": round(elapsed, 2),
            **self._current_fit_metrics,
            **self._current_server_eval,   # empty dict if no evaluate_fn set
            "num_clients_eval":  len(results),
            "num_failures_eval": len(failures),
            **eval_metrics,
        }

        self._history.append(round_record)
        self._write_record(round_record)
        self._print_summary(round_record)

        return aggregated_loss, aggregated_metrics

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _write_record(self, record: dict[str, Any]) -> None:
        """Append one JSON record to the JSONL log file."""
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def _print_summary(self, r: dict[str, Any]) -> None:
        """Print a one-line human-readable round summary."""
        rnd  = r.get("round", "?")
        ta   = r.get("train_accuracy", 0) * 100
        tl   = r.get("train_loss", 0)
        pidl = r.get("train_pidl_loss", 0)
        va   = r.get("val_accuracy", 0) * 100
        vl   = r.get("val_loss", 0)
        sa   = r.get("server_accuracy", None)
        t    = r.get("elapsed_seconds", 0)

        server_part = ""
        if sa is not None:
            server_part = f"  Server Acc: {sa*100:5.2f}% |"

        print(
            f"Round {rnd:>3} | "
            f"Train Acc: {ta:5.2f}%  Loss: {tl:.4f}  PIDL: {pidl:.6f} | "
            f"Client Val Acc: {va:5.2f}%  Loss: {vl:.4f} |"
            f"{server_part} "
            f"Elapsed: {t:.1f}s"
        )

    def _save_summary(self) -> None:
        """Write summary.json with best/last metrics and full history."""
        if not self._history:
            return

        # Best server accuracy round (or last if no server eval)
        best_server = max(
            self._history,
            key=lambda r: r.get("server_accuracy", r.get("val_accuracy", 0)),
        )
        last_round = self._history[-1]

        summary = {
            "total_rounds": len(self._history),
            "best_round":   best_server,
            "last_round":   last_round,
            "history":      self._history,
        }

        summary_path = self.log_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[LoggingFedAvg] Summary saved to: {summary_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict[str, Any]]:
        """Return the full list of per-round metric records."""
        return self._history

    def close(self) -> None:
        """Flush the JSONL log, save ``summary.json``, and close the file handle."""
        if not self._log_file.closed:
            self._log_file.flush()
            self._save_summary()
            self._log_file.close()

    def __del__(self) -> None:
        self.close()
