"""Utilities for simple volatility-quantile overlay baselines."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from evaluation.null_baselines import (
    as_position_series,
    block_bootstrap_same_exposure_random,
    evaluate_position_strategy,
    monte_carlo_null_metrics,
    same_average_exposure_random,
    same_exposure_and_turnover_random,
    same_turnover_random,
    summarize_null_distribution,
)


def compute_trailing_realized_volatility(
    asset_returns,
    *,
    window: int = 20,
    annualization: int = 252,
    split_ids=None,
) -> pd.Series:
    """Compute trailing annualized realized volatility using past data only."""

    returns = pd.Series(asset_returns, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive.")
    if annualization <= 0:
        raise ValueError("annualization must be positive.")

    annualizer = float(np.sqrt(annualization))

    def _per_group(group: pd.Series) -> pd.Series:
        return group.shift(1).rolling(window, min_periods=window).std() * annualizer

    if split_ids is None:
        return _per_group(returns).rename("trailing_realized_volatility")

    split_series = pd.Series(split_ids, index=returns.index, name="split_id")
    if len(split_series) != len(returns):
        raise ValueError("split_ids must be the same length as asset_returns.")
    realized = returns.groupby(split_series, sort=False).apply(_per_group)
    realized.index = returns.index
    return realized.rename("trailing_realized_volatility")


def derive_train_volatility_cutoffs(
    train_realized_volatility,
    *,
    quantiles: tuple[float, ...] = (0.8,),
) -> dict[str, float]:
    """Derive train-only realized-volatility cutoffs."""

    realized = pd.Series(train_realized_volatility, dtype=float).dropna()
    if realized.empty:
        raise ValueError("Cannot derive volatility cutoffs from an empty train series.")
    ordered = tuple(float(q) for q in quantiles)
    if not ordered:
        raise ValueError("At least one quantile is required.")
    if any(not 0.0 < q < 1.0 for q in ordered):
        raise ValueError("Quantiles must lie strictly inside (0, 1).")
    if list(ordered) != sorted(ordered):
        raise ValueError("Quantiles must be sorted ascending.")
    return {f"q_{q:.4f}": float(realized.quantile(q)) for q in ordered}


def assign_volatility_states(
    realized_volatility,
    *,
    cutoffs: dict[str, float],
) -> pd.Series:
    """Map realized volatility into ordinal state labels from train-only cutoffs."""

    realized = pd.Series(realized_volatility, dtype=float)
    thresholds = np.array(sorted(float(value) for value in cutoffs.values()), dtype=float)
    if len(thresholds) == 0:
        raise ValueError("At least one cutoff is required.")
    states = np.searchsorted(thresholds, realized.to_numpy(dtype=float), side="right")
    return pd.Series(states, index=realized.index, name="vol_state", dtype=int)


def identify_high_vol_state_ids(
    state_labels,
    *,
    min_state: int | None = None,
) -> list[int]:
    """Return the high-volatility state ids to cut."""

    states = pd.Series(state_labels, dtype=int)
    if states.empty:
        raise ValueError("Cannot identify high-vol states from an empty label series.")
    state_max = int(states.max())
    if min_state is None:
        return [state_max]
    return [int(value) for value in sorted(states.unique()) if int(value) >= int(min_state)]


def build_vol_state_from_quantile(
    realized_volatility,
    *,
    quantile: float,
) -> tuple[pd.Series, dict[str, float]]:
    """Convenience helper for a one-quantile two-state vol overlay."""

    cutoffs = derive_train_volatility_cutoffs(realized_volatility, quantiles=(quantile,))
    labels = assign_volatility_states(realized_volatility, cutoffs=cutoffs)
    return labels, cutoffs


def apply_volatility_quantile_overlay(
    base_positions,
    vol_state_labels,
    *,
    high_state_ids: list[int] | tuple[int, ...],
    risk_multiplier: float,
) -> pd.Series:
    """Reduce exposure when the current vol state is in the chosen high-vol set."""

    if risk_multiplier < 0.0:
        raise ValueError("risk_multiplier must be non-negative.")
    base = pd.Series(base_positions, dtype=float)
    states = pd.Series(vol_state_labels, index=base.index, dtype=int)
    high_state_set = {int(value) for value in high_state_ids}
    if not high_state_set:
        return base.rename("overlay_position")
    overlay = base.where(~states.isin(high_state_set), other=base * risk_multiplier)
    return overlay.rename("overlay_position")


def same_vol_state_exposure_random(
    positions,
    vol_state_labels,
    *,
    seed: int | None = None,
    index=None,
    name: str = "position",
) -> pd.Series:
    """Shuffle positions within each volatility-state bucket."""

    series = as_position_series(positions, index=index, name=name)
    states = as_position_series(vol_state_labels, index=series.index, name="vol_state").astype(int)
    if len(series) != len(states):
        raise ValueError("Positions and vol_state_labels must have the same length.")

    rng = np.random.default_rng(seed)
    randomized = series.copy()
    for state_id in pd.unique(states):
        mask = states.eq(state_id)
        values = randomized.loc[mask].to_numpy(copy=True)
        rng.shuffle(values)
        randomized.loc[mask] = values
    return randomized.astype(float)


def run_volatility_matched_null_suite(
    *,
    positions,
    returns,
    benchmark_returns,
    vol_state_labels,
    n_runs: int = 100,
    seed: int = 42,
    annualization_factor: int = 252,
    transaction_cost_bps: float = 2.0,
    decision_metric: str = "information_ratio",
    include_block_bootstrap: bool = True,
) -> dict[str, Any]:
    """Evaluate canonical metrics plus matched nulls including vol-state exposure."""

    canonical_metrics = evaluate_position_strategy(
        positions=positions,
        returns=returns,
        benchmark_returns=benchmark_returns,
        annualization_factor=annualization_factor,
        transaction_cost_bps=transaction_cost_bps,
    )

    suite: dict[str, dict[str, Any]] = {}
    generators: list[tuple[str, Callable[..., pd.Series], dict[str, Any]]] = [
        ("same_average_exposure_random", same_average_exposure_random, {}),
        ("same_turnover_random", same_turnover_random, {}),
        ("same_exposure_and_turnover_random", same_exposure_and_turnover_random, {}),
        ("same_vol_state_exposure_random", same_vol_state_exposure_random, {"vol_state_labels": vol_state_labels}),
    ]
    if include_block_bootstrap:
        generators.append(("block_bootstrap_same_exposure_random", block_bootstrap_same_exposure_random, {}))

    for label, generator, generator_kwargs in generators:
        null_metrics = monte_carlo_null_metrics(
            generator=generator,
            positions=positions,
            returns=returns,
            benchmark_returns=benchmark_returns,
            n_runs=n_runs,
            seed=seed,
            annualization_factor=annualization_factor,
            transaction_cost_bps=transaction_cost_bps,
            generator_kwargs=generator_kwargs,
        )
        summary = summarize_null_distribution(
            canonical_metrics=canonical_metrics,
            null_metrics=null_metrics,
            decision_metric=decision_metric,
        )
        suite[label] = {
            "summary": summary,
            "null_metrics": null_metrics,
        }

    return {
        "canonical_metrics": canonical_metrics,
        "decision_metric": decision_metric,
        "null_summaries": suite,
    }
