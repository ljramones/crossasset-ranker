"""Matched null-baseline utilities for exposure and regime-aware validation.

These helpers are intentionally lightweight and deterministic under a seed.
They operate on executed position paths rather than model objects so they can be
reused across existing strategies, overlays, and future clean-report workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from evaluation.metrics import compute_return_stream_metrics, compute_signal_turnover


@dataclass(slots=True)
class NullDistributionSummary:
    """Summary statistics comparing a canonical strategy to null draws."""

    decision_metric: str
    canonical_value: float
    mean_null_value: float
    percentile_95_null_value: float
    p_value: float
    n_runs: int


def as_position_series(positions, index=None, name: str = "position") -> pd.Series:
    """Convert arbitrary position-like input into a float Series."""

    if isinstance(positions, pd.Series):
        series = positions.astype(float).copy()
        if index is not None:
            if len(series) != len(index):
                raise ValueError("Provided index length does not match position series length.")
            series.index = index
        series.name = name
        return series

    values = np.asarray(list(positions), dtype=float)
    if index is not None and len(values) != len(index):
        raise ValueError("Provided index length does not match positions length.")
    return pd.Series(values, index=index, name=name, dtype=float)


def same_average_exposure_random(
    positions,
    *,
    seed: int | None = None,
    index=None,
    name: str = "position",
) -> pd.Series:
    """Shuffle the full position vector while preserving its exact value distribution."""

    series = as_position_series(positions, index=index, name=name)
    rng = np.random.default_rng(seed)
    shuffled = series.to_numpy(copy=True)
    rng.shuffle(shuffled)
    return pd.Series(shuffled, index=series.index, name=series.name, dtype=float)


def same_turnover_random(
    positions,
    *,
    seed: int | None = None,
    index=None,
    name: str = "position",
) -> pd.Series:
    """Break return alignment by circularly shifting the existing position path.

    This preserves the path structure exactly up to rotation, including holding
    runs and most turnover characteristics, while removing date alignment.
    """

    series = as_position_series(positions, index=index, name=name)
    if len(series) <= 1:
        return series.copy()
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, len(series)))
    shifted = np.roll(series.to_numpy(copy=True), shift)
    return pd.Series(shifted, index=series.index, name=series.name, dtype=float)


def same_exposure_and_turnover_random(
    positions,
    *,
    seed: int | None = None,
    index=None,
    name: str = "position",
) -> pd.Series:
    """Alias for the current path-structure-matched null generator."""

    return same_turnover_random(positions, seed=seed, index=index, name=name)


def same_regime_exposure_random(
    positions,
    regime_labels,
    *,
    seed: int | None = None,
    index=None,
    name: str = "position",
) -> pd.Series:
    """Shuffle positions within each regime bucket while preserving per-regime exposure."""

    series = as_position_series(positions, index=index, name=name)
    regimes = as_position_series(regime_labels, index=series.index, name="regime_id")
    if len(series) != len(regimes):
        raise ValueError("Positions and regime_labels must have the same length.")

    rng = np.random.default_rng(seed)
    randomized = series.copy()
    for regime_id in pd.unique(regimes):
        mask = regimes.eq(regime_id)
        values = randomized.loc[mask].to_numpy(copy=True)
        rng.shuffle(values)
        randomized.loc[mask] = values
    return randomized.astype(float)


def block_bootstrap_same_exposure_random(
    positions,
    *,
    block_size: int = 5,
    seed: int | None = None,
    index=None,
    name: str = "position",
) -> pd.Series:
    """Resample contiguous position blocks with replacement to break date alignment.

    This preserves local run structure better than a full shuffle while keeping
    the null deterministic under a seed. The output length always matches the
    input length, though exact exposure is only approximately preserved.
    """

    series = as_position_series(positions, index=index, name=name)
    if len(series) <= 1:
        return series.copy()
    block = max(int(block_size), 1)
    rng = np.random.default_rng(seed)
    values = series.to_numpy(copy=True)
    starts = list(range(0, len(values), block))
    resampled_chunks: list[np.ndarray] = []
    while sum(len(chunk) for chunk in resampled_chunks) < len(values):
        start = starts[int(rng.integers(0, len(starts)))]
        stop = min(start + block, len(values))
        resampled_chunks.append(values[start:stop])
    sampled = np.concatenate(resampled_chunks)[: len(values)]
    return pd.Series(sampled, index=series.index, name=series.name, dtype=float)


def evaluate_position_strategy(
    *,
    positions,
    returns,
    benchmark_returns,
    annualization_factor: int = 252,
    transaction_cost_bps: float = 2.0,
    index=None,
) -> dict[str, float]:
    """Evaluate a strategy from an executed position path and return stream."""

    position = as_position_series(positions, index=index, name="position")
    strategy_returns = as_position_series(returns, index=position.index, name="strategy_return")
    benchmark = as_position_series(benchmark_returns, index=position.index, name="benchmark_return")
    if not (len(position) == len(strategy_returns) == len(benchmark)):
        raise ValueError("positions, returns, and benchmark_returns must have matching lengths.")

    gross_returns = (position * strategy_returns).rename("gross_return")
    turnover_series = compute_signal_turnover(position)
    cost_series = turnover_series * (transaction_cost_bps / 10_000.0)
    net_returns = (gross_returns - cost_series).rename("net_return")
    prediction = position.gt(0.0).astype(float).rename("prediction")

    metrics = compute_return_stream_metrics(
        net_returns=net_returns,
        benchmark_returns=benchmark,
        annualization_factor=annualization_factor,
        gross_returns=gross_returns,
        trade_frequency=float(position.ne(0.0).mean()),
        turnover=float(turnover_series.mean()),
        position=position,
        prediction=prediction,
    )
    metrics["annualized_cost_drag"] = float(cost_series.mean() * annualization_factor)
    return metrics


def monte_carlo_null_metrics(
    *,
    generator: Callable[..., pd.Series],
    positions,
    returns,
    benchmark_returns,
    n_runs: int = 100,
    seed: int = 42,
    annualization_factor: int = 252,
    transaction_cost_bps: float = 2.0,
    generator_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run one null generator repeatedly and return one metrics row per draw."""

    base_position = as_position_series(positions)
    generator_kwargs = dict(generator_kwargs or {})
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for run_id in range(n_runs):
        child_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        sampled_position = generator(
            base_position,
            seed=child_seed,
            index=base_position.index,
            **generator_kwargs,
        )
        metrics = evaluate_position_strategy(
            positions=sampled_position,
            returns=returns,
            benchmark_returns=benchmark_returns,
            annualization_factor=annualization_factor,
            transaction_cost_bps=transaction_cost_bps,
            index=base_position.index,
        )
        rows.append({"run_id": float(run_id), **metrics})
    return pd.DataFrame(rows)


def summarize_null_distribution(
    *,
    canonical_metrics: dict[str, float],
    null_metrics: pd.DataFrame,
    decision_metric: str = "information_ratio",
) -> NullDistributionSummary:
    """Compare canonical metrics to a null distribution on one decision metric."""

    if decision_metric not in canonical_metrics:
        raise KeyError(f"Canonical metrics are missing decision metric {decision_metric!r}.")
    if decision_metric not in null_metrics.columns:
        raise KeyError(f"Null metrics are missing decision metric {decision_metric!r}.")

    values = null_metrics[decision_metric].astype(float).to_numpy()
    canonical_value = float(canonical_metrics[decision_metric])
    if len(values) == 0:
        return NullDistributionSummary(
            decision_metric=decision_metric,
            canonical_value=canonical_value,
            mean_null_value=0.0,
            percentile_95_null_value=0.0,
            p_value=1.0,
            n_runs=0,
        )

    return NullDistributionSummary(
        decision_metric=decision_metric,
        canonical_value=canonical_value,
        mean_null_value=float(np.mean(values)),
        percentile_95_null_value=float(np.percentile(values, 95)),
        p_value=float(np.mean(values >= canonical_value)),
        n_runs=int(len(values)),
    )


def run_matched_null_suite(
    *,
    positions,
    returns,
    benchmark_returns,
    regime_labels=None,
    n_runs: int = 100,
    seed: int = 42,
    annualization_factor: int = 252,
    transaction_cost_bps: float = 2.0,
    decision_metric: str = "information_ratio",
    include_block_bootstrap: bool = True,
) -> dict[str, Any]:
    """Evaluate canonical metrics plus a small suite of matched null summaries."""

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
    ]
    if regime_labels is not None:
        generators.append(
            (
                "same_regime_exposure_random",
                same_regime_exposure_random,
                {"regime_labels": regime_labels},
            )
        )
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
