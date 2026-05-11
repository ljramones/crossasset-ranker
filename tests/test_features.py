"""Tests for stationary feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.engineering import build_feature_set


def test_build_feature_set_creates_expected_columns() -> None:
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    base = np.linspace(100.0, 120.0, num=len(dates))
    frame = pd.DataFrame(
        {
            "Open": base * 0.99,
            "High": base * 1.01,
            "Low": base * 0.98,
            "Close": base,
            "Adj Close": base,
            "Volume": np.linspace(1_000_000, 1_200_000, num=len(dates)),
            "BenchmarkClose": base,
            "benchmark_return_1d": np.zeros(len(dates)),
        },
        index=dates,
    )

    feature_set = build_feature_set(frame)

    assert "return_1d" in feature_set.feature_columns
    assert "target_direction" in feature_set.frame.columns
    assert not feature_set.stationarity_summary.empty


def test_build_feature_set_adds_advanced_columns_when_enabled() -> None:
    dates = pd.date_range("2020-01-01", periods=180, freq="D")
    trend = np.linspace(100.0, 130.0, num=len(dates))
    oscillation = 2.5 * np.sin(np.linspace(0.0, 18.0, num=len(dates)))
    base = trend + oscillation
    benchmark = trend * 0.995 + 1.5 * np.cos(np.linspace(0.0, 14.0, num=len(dates)))
    frame = pd.DataFrame(
        {
            "Open": base * 0.997,
            "High": base * 1.012,
            "Low": base * 0.988,
            "Close": base,
            "Adj Close": base,
            "Volume": 1_000_000 + 50_000 * np.sin(np.linspace(0.0, 10.0, num=len(dates))),
            "BenchmarkClose": benchmark,
            "benchmark_return_1d": pd.Series(benchmark, index=dates).pct_change().fillna(0.0).values,
        },
        index=dates,
    )

    feature_set = build_feature_set(frame, advanced_features=True)

    expected_columns = {
        "rsi_zscore",
        "macd_histogram_zscore",
        "bollinger_band_width_zscore",
        "relative_strength_vs_benchmark",
        "overnight_gap_zscore",
        "volume_trend_strength",
        "autocorrelation_zscore",
        "volatility_regime",
        "price_acceleration_2nd",
    }

    assert expected_columns.issubset(set(feature_set.feature_columns))
    assert expected_columns.issubset(set(feature_set.stationarity_summary["feature"]))


def test_build_feature_set_adds_vix_columns_when_enabled() -> None:
    dates = pd.date_range("2020-01-01", periods=180, freq="D")
    trend = np.linspace(100.0, 130.0, num=len(dates))
    oscillation = 2.5 * np.sin(np.linspace(0.0, 18.0, num=len(dates)))
    base = trend + oscillation
    benchmark = trend * 0.995 + 1.5 * np.cos(np.linspace(0.0, 14.0, num=len(dates)))
    vix = 18.0 + 2.0 * np.sin(np.linspace(0.0, 12.0, num=len(dates))) + np.linspace(-1.0, 1.0, num=len(dates))
    frame = pd.DataFrame(
        {
            "Open": base * 0.997,
            "High": base * 1.012,
            "Low": base * 0.988,
            "Close": base,
            "Adj Close": base,
            "Volume": 1_000_000 + 50_000 * np.sin(np.linspace(0.0, 10.0, num=len(dates))),
            "BenchmarkClose": benchmark,
            "benchmark_return_1d": pd.Series(benchmark, index=dates).pct_change().fillna(0.0).values,
            "VIXClose": vix,
        },
        index=dates,
    )

    feature_set = build_feature_set(frame, advanced_features=True, vix_features=True)

    expected_columns = {
        "vix_zscore",
        "implied_vs_realized_vol",
        "vix_momentum_5d",
        "vix_extreme",
        "vix_vol_interaction",
        "vix_return_1d",
        "vix_return_5d",
    }

    assert expected_columns.issubset(set(feature_set.feature_columns))
    assert expected_columns.issubset(set(feature_set.stationarity_summary["feature"]))
