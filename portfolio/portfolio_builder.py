"""Selective portfolio construction from multi-asset regime-stacking signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import compute_return_stream_metrics


def load_asset_signal_panel(
    run_records: list[dict[str, Any]],
    *,
    champion_model: str,
) -> dict[float, pd.DataFrame]:
    """Load saved OOF signal files into one panel per transaction-cost scenario."""

    panels: dict[float, list[pd.DataFrame]] = {}
    filename = "regime_stacking_oof.csv" if champion_model == "regime_stacking_ensemble" else "regime_stacking_oof_interactions.csv"
    for record in run_records:
        path = Path(record["results_dir"]) / "ensembles" / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["ticker"] = record["ticker"]
        frame["cost_bps"] = float(record["cost_bps"])
        panels.setdefault(float(record["cost_bps"]), []).append(frame)

    merged: dict[float, pd.DataFrame] = {}
    for cost_bps, frames in panels.items():
        merged[cost_bps] = pd.concat(frames, ignore_index=True)
    return merged


def build_selective_portfolios(
    signal_panel: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    annualization_factor: int,
    transaction_cost_bps: float,
    volatility_target: float,
) -> list[dict[str, Any]]:
    """Build equal-weight and volatility-targeted selective portfolios."""

    eligible_assets = eligibility.loc[eligibility["eligible"], "ticker"].tolist()
    if not eligible_assets:
        return []

    returns_wide = signal_panel.pivot(index="date", columns="ticker", values="forward_simple_return_1d").sort_index()
    signal_wide = signal_panel.pivot(index="date", columns="ticker", values="prediction").sort_index().fillna(0.0)
    benchmark_returns = signal_panel.drop_duplicates("date").set_index("date")["benchmark_return_1d"].sort_index()

    eligible_returns = returns_wide.reindex(columns=eligible_assets).fillna(0.0)
    eligible_signals = signal_wide.reindex(columns=eligible_assets).fillna(0.0)
    all_returns = returns_wide.fillna(0.0)

    results = [
        _evaluate_weight_scheme(
            label="selective_equal_weight",
            returns=eligible_returns,
            signals=eligible_signals,
            benchmark_returns=benchmark_returns,
            annualization_factor=annualization_factor,
            transaction_cost_bps=transaction_cost_bps,
            weight_builder=_equal_weight_targets,
            selected_assets=eligible_assets,
        ),
        _evaluate_weight_scheme(
            label="selective_vol_target",
            returns=eligible_returns,
            signals=eligible_signals,
            benchmark_returns=benchmark_returns,
            annualization_factor=annualization_factor,
            transaction_cost_bps=transaction_cost_bps,
            weight_builder=lambda returns, signals: _vol_target_targets(
                returns=returns,
                signals=signals,
                annualization_factor=annualization_factor,
                volatility_target=volatility_target,
            ),
            selected_assets=eligible_assets,
        ),
        _evaluate_static_benchmark(
            label="equal_weight_all_assets",
            returns=all_returns,
            benchmark_returns=benchmark_returns,
            annualization_factor=annualization_factor,
            transaction_cost_bps=transaction_cost_bps,
        ),
    ]
    return results


def _evaluate_weight_scheme(
    *,
    label: str,
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark_returns: pd.Series,
    annualization_factor: int,
    transaction_cost_bps: float,
    weight_builder,
    selected_assets: list[str],
) -> dict[str, Any]:
    """Evaluate one dynamic portfolio weighting scheme."""

    target_weights = weight_builder(returns=returns, signals=signals).fillna(0.0)
    executed_weights = target_weights.shift(1).fillna(0.0)
    gross_returns = (executed_weights * returns).sum(axis=1)
    turnover_series = target_weights.diff().abs().sum(axis=1).fillna(target_weights.abs().sum(axis=1))
    net_returns = gross_returns - turnover_series * (transaction_cost_bps / 10_000.0)
    metrics = compute_return_stream_metrics(
        net_returns=net_returns.rename("net_return"),
        benchmark_returns=benchmark_returns.reindex(net_returns.index).fillna(0.0),
        annualization_factor=annualization_factor,
        gross_returns=gross_returns.rename("gross_return"),
        trade_frequency=float(executed_weights.abs().sum(axis=1).gt(0.0).mean()),
        turnover=float(turnover_series.mean()),
    )
    return {
        "model": label,
        "selected_assets": ",".join(selected_assets),
        "selected_count": len(selected_assets),
        **metrics,
    }


def _evaluate_static_benchmark(
    *,
    label: str,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    annualization_factor: int,
    transaction_cost_bps: float,
) -> dict[str, Any]:
    """Evaluate a naive always-invested equal-weight benchmark."""

    target_weights = pd.DataFrame(
        np.repeat(1.0 / returns.shape[1], repeats=len(returns) * returns.shape[1]).reshape(len(returns), returns.shape[1]),
        index=returns.index,
        columns=returns.columns,
    )
    executed_weights = target_weights.shift(1).fillna(0.0)
    gross_returns = (executed_weights * returns).sum(axis=1)
    turnover_series = target_weights.diff().abs().sum(axis=1).fillna(target_weights.abs().sum(axis=1))
    net_returns = gross_returns - turnover_series * (transaction_cost_bps / 10_000.0)
    metrics = compute_return_stream_metrics(
        net_returns=net_returns.rename("net_return"),
        benchmark_returns=benchmark_returns.reindex(net_returns.index).fillna(0.0),
        annualization_factor=annualization_factor,
        gross_returns=gross_returns.rename("gross_return"),
        trade_frequency=float(executed_weights.abs().sum(axis=1).gt(0.0).mean()),
        turnover=float(turnover_series.mean()),
    )
    return {
        "model": label,
        "selected_assets": ",".join(returns.columns.tolist()),
        "selected_count": returns.shape[1],
        **metrics,
    }


def _equal_weight_targets(*, returns: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Build equal-weight target allocations across active signals."""

    active_counts = signals.sum(axis=1).replace(0.0, np.nan)
    return signals.div(active_counts, axis=0).fillna(0.0)


def _vol_target_targets(
    *,
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    annualization_factor: int,
    volatility_target: float,
) -> pd.DataFrame:
    """Build inverse-vol weighted targets with a rolling portfolio volatility scaler."""

    asset_vol = returns.rolling(20).std().shift(1) * np.sqrt(annualization_factor)
    inv_vol = (1.0 / asset_vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    base_weights = (signals * inv_vol).div((signals * inv_vol).sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    base_executed = base_weights.shift(1).fillna(0.0)
    base_returns = (base_executed * returns).sum(axis=1)
    realized_portfolio_vol = base_returns.rolling(20).std().shift(1) * np.sqrt(annualization_factor)
    leverage = (float(volatility_target) / realized_portfolio_vol.replace(0.0, np.nan)).clip(lower=0.0, upper=2.0).fillna(1.0)
    return base_weights.mul(leverage, axis=0).fillna(0.0)
