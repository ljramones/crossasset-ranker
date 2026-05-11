"""Tests for matched null-baseline utilities."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from evaluation.metrics import compute_signal_turnover
from evaluation.null_baselines import (
    as_position_series,
    block_bootstrap_same_exposure_random,
    evaluate_position_strategy,
    monte_carlo_null_metrics,
    run_matched_null_suite,
    same_average_exposure_random,
    same_exposure_and_turnover_random,
    same_regime_exposure_random,
    same_turnover_random,
    summarize_null_distribution,
)


def _run_signature(series: pd.Series) -> list[tuple[float, int]]:
    values = series.tolist()
    if not values:
        return []
    runs: list[tuple[float, int]] = []
    current_value = values[0]
    current_length = 1
    for value in values[1:]:
        if value == current_value:
            current_length += 1
        else:
            runs.append((float(current_value), current_length))
            current_value = value
            current_length = 1
    runs.append((float(current_value), current_length))
    return runs


def test_as_position_series_respects_index_and_name() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")

    series = as_position_series([0, 1, -1], index=index, name="alloc")

    assert list(series.index) == list(index)
    assert series.name == "alloc"
    assert series.dtype == float


def test_same_average_exposure_random_is_deterministic_and_preserves_distribution() -> None:
    position = pd.Series([0.0, 1.0, 0.0, -1.0, 1.0, 0.0], name="position")

    shuffled_a = same_average_exposure_random(position, seed=7)
    shuffled_b = same_average_exposure_random(position, seed=7)

    assert shuffled_a.equals(shuffled_b)
    assert Counter(shuffled_a.tolist()) == Counter(position.tolist())


def test_same_turnover_random_preserves_rotated_path_structure() -> None:
    position = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0], name="position")

    shifted = same_turnover_random(position, seed=3)

    assert Counter(shifted.tolist()) == Counter(position.tolist())
    assert Counter(_run_signature(shifted)) == Counter(_run_signature(position))
    assert len(compute_signal_turnover(shifted)) == len(compute_signal_turnover(position))


def test_same_regime_exposure_random_preserves_per_regime_position_multiset() -> None:
    position = pd.Series([0.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0], name="position")
    regimes = pd.Series([0, 0, 0, 1, 1, 1, 2, 2], name="regime_id")

    randomized = same_regime_exposure_random(position, regimes, seed=11)

    for regime_id in sorted(regimes.unique()):
        original_values = position.loc[regimes == regime_id].tolist()
        randomized_values = randomized.loc[regimes == regime_id].tolist()
        assert Counter(original_values) == Counter(randomized_values)


def test_same_exposure_and_turnover_random_is_callable_alias() -> None:
    position = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, -1.0], name="position")

    randomized = same_exposure_and_turnover_random(position, seed=5)

    assert len(randomized) == len(position)
    assert Counter(randomized.tolist()) == Counter(position.tolist())


def test_block_bootstrap_same_exposure_random_is_deterministic_and_length_preserving() -> None:
    position = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0], name="position")

    sample_a = block_bootstrap_same_exposure_random(position, block_size=3, seed=17)
    sample_b = block_bootstrap_same_exposure_random(position, block_size=3, seed=17)

    assert sample_a.equals(sample_b)
    assert len(sample_a) == len(position)
    assert set(sample_a.unique()).issubset(set(position.unique()))


def test_evaluate_position_strategy_returns_active_and_turnover_metrics() -> None:
    position = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0], name="position")
    returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.01], name="returns")
    benchmark = pd.Series([0.0, 0.005, 0.0, 0.0, 0.002], name="benchmark")

    metrics = evaluate_position_strategy(
        positions=position,
        returns=returns,
        benchmark_returns=benchmark,
        annualization_factor=252,
        transaction_cost_bps=2.0,
    )

    assert {
        "information_ratio",
        "annualized_active_return",
        "tracking_error",
        "fraction_in_market",
        "daily_turnover",
        "annualized_turnover",
        "cost_drag",
        "annualized_cost_drag",
    } <= set(metrics)


def test_monte_carlo_null_metrics_is_deterministic_under_seed() -> None:
    position = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], name="position")
    returns = pd.Series([0.01, -0.005, 0.004, 0.003, -0.002, 0.005], name="returns")
    benchmark = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name="benchmark")

    sample_a = monte_carlo_null_metrics(
        generator=same_average_exposure_random,
        positions=position,
        returns=returns,
        benchmark_returns=benchmark,
        n_runs=5,
        seed=123,
    )
    sample_b = monte_carlo_null_metrics(
        generator=same_average_exposure_random,
        positions=position,
        returns=returns,
        benchmark_returns=benchmark,
        n_runs=5,
        seed=123,
    )

    pd.testing.assert_frame_equal(sample_a, sample_b)


def test_summarize_null_distribution_returns_p_value_summary() -> None:
    canonical_metrics = {"information_ratio": 0.25}
    null_metrics = pd.DataFrame({"information_ratio": [-0.1, 0.0, 0.1, 0.3, 0.5]})

    summary = summarize_null_distribution(
        canonical_metrics=canonical_metrics,
        null_metrics=null_metrics,
        decision_metric="information_ratio",
    )

    assert summary.decision_metric == "information_ratio"
    assert summary.canonical_value == 0.25
    assert summary.n_runs == 5
    assert summary.p_value == pytest.approx(0.4)


def test_run_matched_null_suite_returns_expected_summary_keys() -> None:
    position = pd.Series([0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0], name="position")
    returns = pd.Series([0.01, -0.005, 0.004, 0.003, -0.002, 0.005, 0.002, -0.001], name="returns")
    benchmark = pd.Series([0.0] * len(position), name="benchmark")
    regimes = pd.Series([0, 0, 1, 1, 2, 2, 0, 1], name="regime_id")

    suite = run_matched_null_suite(
        positions=position,
        returns=returns,
        benchmark_returns=benchmark,
        regime_labels=regimes,
        n_runs=5,
        seed=23,
    )

    assert "canonical_metrics" in suite
    assert "null_summaries" in suite
    assert "same_average_exposure_random" in suite["null_summaries"]
    assert "same_turnover_random" in suite["null_summaries"]
    assert "same_exposure_and_turnover_random" in suite["null_summaries"]
    assert "same_regime_exposure_random" in suite["null_summaries"]
    assert "block_bootstrap_same_exposure_random" in suite["null_summaries"]
    assert suite["null_summaries"]["same_average_exposure_random"]["summary"].n_runs == 5
