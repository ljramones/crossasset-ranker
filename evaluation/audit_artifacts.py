"""Helpers for standardized audit-ready OOF artifact frames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import compute_signal_turnover


STANDARD_AUDIT_COLUMNS = [
    "date",
    "split_id",
    "model_name",
    "asset",
    "asset_return",
    "benchmark_return",
    "raw_signal",
    "prediction_probability",
    "target",
    "executed_position",
    "strategy_gross_return",
    "strategy_net_return",
    "turnover",
    "transaction_cost",
    "regime_id",
    "regime_prob_0",
    "regime_prob_1",
    "regime_prob_2",
]


def _coerce_series(values: pd.Series | Any, *, index: pd.Index, name: str) -> pd.Series:
    """Return a float Series aligned to the provided index."""

    if isinstance(values, pd.Series):
        return values.astype(float).rename(name)
    return pd.Series(values, index=index, name=name, dtype=float)


def build_standard_audit_artifact_frame(
    *,
    frame: pd.DataFrame,
    label: str,
    prediction: pd.Series,
    probability: pd.Series | None,
    split_id: int,
    transaction_cost_bps: float,
    asset_name: str | None = None,
    target_column: str = "target_direction",
    return_column: str = "forward_simple_return_1d",
    benchmark_column: str = "benchmark_return_1d",
) -> pd.DataFrame:
    """Build a standardized audit-ready artifact frame for one test split.

    The returned frame intentionally retains the legacy columns already used by
    the codebase while adding unambiguous audit columns for future artifact-only
    matched-null analysis.
    """

    signal = _coerce_series(prediction, index=frame.index, name="raw_signal")
    probability_series = (
        _coerce_series(probability, index=frame.index, name="prediction_probability")
        if probability is not None
        else pd.Series(np.nan, index=frame.index, name="prediction_probability")
    )
    asset_returns = frame[return_column].astype(float).rename("asset_return")
    benchmark_returns = frame[benchmark_column].astype(float).rename("benchmark_return")
    executed_position = signal.shift(1).fillna(0.0).astype(float).rename("executed_position")
    turnover = compute_signal_turnover(executed_position).rename("turnover")
    transaction_cost = (turnover * (transaction_cost_bps / 10_000.0)).rename("transaction_cost")
    strategy_gross_return = (executed_position * asset_returns).rename("strategy_gross_return")
    strategy_net_return = (strategy_gross_return - transaction_cost).rename("strategy_net_return")

    artifact = pd.DataFrame(
        {
            "date": frame.index if "date" not in frame.columns else frame["date"].values,
            "split_id": split_id,
            "model_name": label,
            "asset": asset_name if asset_name is not None else np.nan,
            "asset_return": asset_returns.values,
            "benchmark_return": benchmark_returns.values,
            "raw_signal": signal.values,
            "prediction_probability": probability_series.values,
            "target": frame[target_column].values if target_column in frame.columns else np.nan,
            "executed_position": executed_position.values,
            "strategy_gross_return": strategy_gross_return.values,
            "strategy_net_return": strategy_net_return.values,
            "turnover": turnover.values,
            "transaction_cost": transaction_cost.values,
            # Legacy / backward-compatible names
            "target_direction": frame[target_column].values if target_column in frame.columns else np.nan,
            "forward_simple_return_1d": asset_returns.values,
            "benchmark_return_1d": benchmark_returns.values,
            "probability": probability_series.values,
            "prediction": signal.values,
            "model": label,
        }
    )

    for column in frame.columns:
        if column in {"regime_id", "best_regime_id", "best_regime_prob", "trade_allowed_aggressive"} or column.startswith("regime_prob_"):
            artifact[column] = frame[column].values

    return artifact
