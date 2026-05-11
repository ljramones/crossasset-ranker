"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest
import pandas as pd

from evaluation.metrics import (
    active_return_series,
    average_holding_period,
    compute_classification_metrics,
    compute_signal_turnover,
    compute_trading_metrics,
    position_flip_count,
    round_trip_count,
)


def test_compute_classification_metrics_returns_expected_keys() -> None:
    metrics = compute_classification_metrics(
        y_true=pd.Series([0, 1, 1, 0]),
        y_score=pd.Series([0.1, 0.9, 0.8, 0.2]),
        y_pred=pd.Series([0, 1, 1, 0]),
    )

    assert set(metrics) == {"directional_accuracy", "auc_roc"}


def test_compute_trading_metrics_returns_expected_keys() -> None:
    metrics = compute_trading_metrics(
        returns=pd.Series([0.01, -0.005, 0.004, 0.002]),
        benchmark_returns=pd.Series([0.0, 0.0, 0.0, 0.0]),
        signal=pd.Series([0, 1, 1, 0]),
        annualization_factor=252,
        transaction_cost_bps=2.0,
    )

    assert {
        "sharpe",
        "net_sharpe",
        "sharpe_cost_delta",
        "sortino",
        "net_sortino",
        "sortino_cost_delta",
        "calmar",
        "information_ratio",
        "benchmark_sharpe",
        "excess_net_sharpe",
        "max_drawdown",
        "annualized_active_return",
        "active_volatility",
        "tracking_error",
        "active_max_drawdown",
        "active_calmar",
        "correlation_to_benchmark",
        "beta_to_benchmark",
        "alpha_after_beta",
        "fraction_in_market",
        "average_short_exposure",
        "average_abs_position",
        "position_flip_count",
        "daily_turnover",
        "annualized_turnover",
        "average_holding_period_days",
        "round_trip_count",
        "cost_drag",
        "cost_per_unit_active_return",
        "gross_total_return",
        "net_total_return",
        "total_return",
    } <= set(metrics)


def test_compute_signal_turnover_charges_only_on_position_flips() -> None:
    position = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])

    turnover = compute_signal_turnover(position)

    assert turnover.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_active_return_series_is_strategy_minus_benchmark() -> None:
    active = active_return_series(
        strategy_returns=pd.Series([0.02, -0.01, 0.03]),
        benchmark_returns=pd.Series([0.01, 0.00, 0.02]),
    )

    assert active.tolist() == pytest.approx([0.01, -0.01, 0.01])


def test_compute_trading_metrics_always_flat_has_zero_exposure_and_turnover() -> None:
    metrics = compute_trading_metrics(
        returns=pd.Series([0.01, -0.02, 0.015, 0.005]),
        benchmark_returns=pd.Series([0.0, 0.0, 0.0, 0.0]),
        signal=pd.Series([0, 0, 0, 0]),
        annualization_factor=252,
        transaction_cost_bps=2.0,
    )

    assert metrics["trade_frequency"] == 0.0
    assert metrics["fraction_in_market"] == 0.0
    assert metrics["average_long_exposure"] == 0.0
    assert metrics["average_short_exposure"] == 0.0
    assert metrics["average_abs_position"] == 0.0
    assert metrics["position_flip_count"] == 0.0
    assert metrics["daily_turnover"] == 0.0
    assert metrics["round_trip_count"] == 0.0


def test_buy_and_hold_like_signal_has_no_flips_after_initial_entry() -> None:
    signal = pd.Series([1] * 20)
    metrics = compute_trading_metrics(
        returns=pd.Series([0.001] * 20),
        benchmark_returns=pd.Series([0.0] * 20),
        signal=signal,
        annualization_factor=252,
        transaction_cost_bps=2.0,
    )

    assert metrics["position_flip_count"] == 0.0
    assert metrics["fraction_positive_predictions"] == 1.0
    assert metrics["fraction_in_market"] > 0.9
    assert metrics["daily_turnover"] > 0.0


def test_alternating_long_flat_has_high_turnover_and_round_trips() -> None:
    signal = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    metrics = compute_trading_metrics(
        returns=pd.Series([0.0] * len(signal)),
        benchmark_returns=pd.Series([0.0] * len(signal)),
        signal=signal,
        annualization_factor=252,
        transaction_cost_bps=2.0,
    )

    assert metrics["position_flip_count"] >= 5.0
    assert metrics["daily_turnover"] > 0.5
    assert metrics["round_trip_count"] >= 3.0
    assert metrics["average_holding_period_days"] <= 1.0


def test_long_short_turnover_helper_handles_two_unit_flips() -> None:
    position = pd.Series([0.0, 1.0, -1.0, 1.0, -1.0])

    turnover = compute_signal_turnover(position)

    assert turnover.tolist() == [0.0, 1.0, 2.0, 2.0, 2.0]


def test_position_helpers_capture_holding_periods_and_round_trips() -> None:
    position = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])

    assert position_flip_count(position) == 3
    assert average_holding_period(position) == 1.5
    assert round_trip_count(position) == 2
