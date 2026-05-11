"""Utilities for fold-local regime-risk overlay experiments.

This module is intentionally model-agnostic. It does not fit HMMs or import
regime-detection logic. It operates only on supplied return streams, supplied
regime labels, and supplied regime probabilities so later experiments can test
overlay hypotheses without hardcoding regime IDs or mixing in training logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import compute_signal_turnover
from evaluation.null_baselines import evaluate_position_strategy


def _compute_position_return_streams(
    *,
    asset_returns,
    positions,
    transaction_cost_bps: float,
) -> dict[str, pd.Series]:
    """Return gross/net streams for an already-executed position path."""

    returns = pd.Series(asset_returns, dtype=float)
    position = pd.Series(positions, index=returns.index, dtype=float)
    turnover = compute_signal_turnover(position).rename("turnover")
    transaction_cost = (turnover * (transaction_cost_bps / 10_000.0)).rename("transaction_cost")
    gross = (position * returns).rename("strategy_gross_return")
    net = (gross - transaction_cost).rename("strategy_net_return")
    return {
        "gross_returns": gross,
        "net_returns": net,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
    }


def build_vol_target_positions(
    asset_returns,
    *,
    target_vol: float = 0.10,
    realized_vol_window: int = 20,
    annualization: int = 252,
    min_position: float = 0.0,
    max_position: float = 1.0,
    split_ids=None,
) -> pd.Series:
    """Build a volatility-targeted baseline position series using past returns only.

    The position at time ``t`` is based on realized volatility estimated from
    returns through ``t-1``. When ``split_ids`` are supplied, the rolling
    volatility estimate resets inside each split to avoid cross-fold leakage.
    """

    returns = pd.Series(asset_returns, copy=True, dtype=float)
    if realized_vol_window <= 0:
        raise ValueError("realized_vol_window must be positive.")
    if annualization <= 0:
        raise ValueError("annualization must be positive.")
    if min_position > max_position:
        raise ValueError("min_position cannot exceed max_position.")

    annualizer = float(np.sqrt(annualization))

    def _per_group(group: pd.Series) -> pd.Series:
        realized = group.shift(1).rolling(realized_vol_window, min_periods=realized_vol_window).std()
        realized = realized * annualizer
        raw = target_vol / realized.replace(0.0, np.nan)
        clipped = raw.clip(lower=min_position, upper=max_position)
        return clipped.fillna(min_position)

    if split_ids is None:
        return _per_group(returns).rename("base_position")

    split_series = pd.Series(split_ids, index=returns.index, name="split_id")
    if len(split_series) != len(returns):
        raise ValueError("split_ids must be the same length as asset_returns.")
    positions = returns.groupby(split_series, sort=False).apply(_per_group)
    positions.index = returns.index
    return positions.rename("base_position")


def characterize_regimes(
    train_returns,
    regime_labels,
    *,
    annualization: int = 252,
) -> pd.DataFrame:
    """Compute fold-local risk statistics for each supplied regime label."""

    returns = pd.Series(train_returns, dtype=float).rename("asset_return")
    regimes = pd.Series(regime_labels, index=returns.index, name="regime_id")
    if len(returns) != len(regimes):
        raise ValueError("train_returns and regime_labels must have matching lengths.")

    rows: list[dict[str, float]] = []
    for regime_id in pd.unique(regimes):
        regime_returns = returns.loc[regimes.eq(regime_id)].dropna()
        if regime_returns.empty:
            continue
        downside = regime_returns[regime_returns < 0.0]
        equity = (1.0 + regime_returns).cumprod()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        rows.append(
            {
                "regime_id": float(regime_id),
                "count": float(len(regime_returns)),
                "average_return": float(regime_returns.mean()),
                "realized_volatility": float(regime_returns.std() * np.sqrt(annualization)),
                "downside_volatility": float(downside.std() * np.sqrt(annualization)) if len(downside) >= 2 else 0.0,
                "drawdown_tendency": float(abs(drawdown.min())) if not drawdown.empty else 0.0,
                "negative_day_fraction": float((regime_returns < 0.0).mean()),
                "large_down_day_fraction": float((regime_returns <= -0.01).mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "regime_id",
                "count",
                "average_return",
                "realized_volatility",
                "downside_volatility",
                "drawdown_tendency",
                "negative_day_fraction",
                "large_down_day_fraction",
            ]
        )
    return pd.DataFrame(rows).sort_values("regime_id").reset_index(drop=True)


def compute_regime_danger_scores(regime_summary: pd.DataFrame) -> pd.DataFrame:
    """Add a fold-local danger score without assuming regime label semantics."""

    if regime_summary.empty:
        return regime_summary.assign(danger_score=pd.Series(dtype=float))

    summary = regime_summary.copy()
    summary["danger_score"] = (
        summary["realized_volatility"].rank(method="average", ascending=True)
        + summary["downside_volatility"].rank(method="average", ascending=True)
        + summary["drawdown_tendency"].rank(method="average", ascending=True)
        + summary["negative_day_fraction"].rank(method="average", ascending=True)
        + summary["large_down_day_fraction"].rank(method="average", ascending=True)
        - summary["average_return"].rank(method="average", ascending=True)
    )
    return summary


def identify_dangerous_regime(regime_summary: pd.DataFrame) -> int | float:
    """Return the fold-local regime ID with the highest danger score."""

    scored = compute_regime_danger_scores(regime_summary)
    if scored.empty:
        raise ValueError("Cannot identify a dangerous regime from an empty summary.")
    row = scored.sort_values(["danger_score", "regime_id"], ascending=[False, True]).iloc[0]
    regime_id = row["regime_id"]
    return int(regime_id) if float(regime_id).is_integer() else float(regime_id)


def extract_dangerous_regime_probability(
    regime_probabilities: pd.DataFrame | pd.Series,
    dangerous_regime_id: int | float,
) -> pd.Series:
    """Return the probability series for the chosen dangerous regime."""

    if isinstance(regime_probabilities, pd.Series):
        return regime_probabilities.astype(float).rename("dangerous_regime_probability")

    candidates = [
        f"regime_prob_{int(dangerous_regime_id)}" if float(dangerous_regime_id).is_integer() else f"regime_prob_{dangerous_regime_id}",
        f"regime_{int(dangerous_regime_id)}_prob" if float(dangerous_regime_id).is_integer() else f"regime_{dangerous_regime_id}_prob",
        str(dangerous_regime_id),
    ]
    for column in candidates:
        if column in regime_probabilities.columns:
            return regime_probabilities[column].astype(float).rename("dangerous_regime_probability")
    raise KeyError(f"Could not resolve dangerous regime probability column for regime {dangerous_regime_id!r}.")


def apply_hard_regime_overlay(
    base_positions,
    dangerous_regime_probability,
    *,
    threshold: float,
    risk_multiplier: float,
) -> pd.Series:
    """Apply a hard-veto overlay to a supplied baseline position series."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    if risk_multiplier < 0.0:
        raise ValueError("risk_multiplier must be non-negative.")

    base = pd.Series(base_positions, dtype=float)
    probability = pd.Series(dangerous_regime_probability, index=base.index, dtype=float)
    overlay = base.where(probability <= threshold, other=base * risk_multiplier)
    return overlay.rename("overlay_position")


def build_hard_overlay_parameter_grid(
    *,
    thresholds: list[float] | tuple[float, ...],
    risk_multipliers: list[float] | tuple[float, ...],
) -> list[dict[str, float]]:
    """Return a simple hard-overlay parameter grid for train/validation search."""

    grid: list[dict[str, float]] = []
    for threshold in thresholds:
        for risk_multiplier in risk_multipliers:
            grid.append(
                {
                    "threshold": float(threshold),
                    "risk_multiplier": float(risk_multiplier),
                }
            )
    return grid


def apply_soft_regime_overlay(
    base_positions,
    dangerous_regime_probability,
    *,
    cut_strength: float,
    min_position: float = 0.0,
    max_position: float = 1.0,
) -> pd.Series:
    """Apply a soft probability-weighted risk cut to the baseline position."""

    if cut_strength < 0.0:
        raise ValueError("cut_strength must be non-negative.")
    if min_position > max_position:
        raise ValueError("min_position cannot exceed max_position.")

    base = pd.Series(base_positions, dtype=float)
    probability = pd.Series(dangerous_regime_probability, index=base.index, dtype=float).clip(lower=0.0, upper=1.0)
    scaled = base * (1.0 - cut_strength * probability)
    return scaled.clip(lower=min_position, upper=max_position).rename("overlay_position")


def summarize_overlay_position_change(base_positions, overlay_positions) -> dict[str, float]:
    """Summarize exposure and turnover effects of an overlay versus baseline."""

    base = pd.Series(base_positions, dtype=float)
    overlay = pd.Series(overlay_positions, index=base.index, dtype=float)
    base_turnover = compute_signal_turnover(base)
    overlay_turnover = compute_signal_turnover(overlay)
    cut_mask = overlay < base
    return {
        "average_base_position": float(base.mean()),
        "average_overlay_position": float(overlay.mean()),
        "average_position_delta": float((overlay - base).mean()),
        "fraction_cut_days": float(cut_mask.mean()),
        "base_daily_turnover": float(base_turnover.mean()),
        "overlay_daily_turnover": float(overlay_turnover.mean()),
        "turnover_delta": float(overlay_turnover.mean() - base_turnover.mean()),
    }


def evaluate_overlay_strategy(
    *,
    asset_returns,
    benchmark_returns,
    positions,
    transaction_cost_bps: float = 0.0,
    annualization: int = 252,
) -> dict[str, float]:
    """Evaluate executed overlay positions against an asset and benchmark return stream."""

    return evaluate_position_strategy(
        positions=positions,
        returns=asset_returns,
        benchmark_returns=benchmark_returns,
        annualization_factor=annualization,
        transaction_cost_bps=transaction_cost_bps,
    )


def evaluate_overlay_vs_baseline(
    *,
    asset_returns,
    benchmark_returns,
    base_positions,
    overlay_positions,
    transaction_cost_bps: float = 0.0,
    annualization: int = 252,
) -> dict[str, Any]:
    """Compare overlay performance directly against its volatility-target baseline.

    Returns:
    - `base_metrics`: base strategy vs market benchmark
    - `overlay_metrics`: overlay strategy vs market benchmark
    - `overlay_vs_base_metrics`: overlay active metrics where the base net-return
      stream is treated as the benchmark
    - `position_change_summary`: exposure/turnover deltas from the overlay
    """

    base_metrics = evaluate_overlay_strategy(
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        positions=base_positions,
        transaction_cost_bps=transaction_cost_bps,
        annualization=annualization,
    )
    overlay_metrics = evaluate_overlay_strategy(
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        positions=overlay_positions,
        transaction_cost_bps=transaction_cost_bps,
        annualization=annualization,
    )
    base_streams = _compute_position_return_streams(
        asset_returns=asset_returns,
        positions=base_positions,
        transaction_cost_bps=transaction_cost_bps,
    )
    overlay_vs_base_metrics = evaluate_position_strategy(
        positions=overlay_positions,
        returns=asset_returns,
        benchmark_returns=base_streams["net_returns"],
        annualization_factor=annualization,
        transaction_cost_bps=transaction_cost_bps,
    )
    return {
        "base_metrics": base_metrics,
        "overlay_metrics": overlay_metrics,
        "overlay_vs_base_metrics": overlay_vs_base_metrics,
        "position_change_summary": summarize_overlay_position_change(base_positions, overlay_positions),
    }
