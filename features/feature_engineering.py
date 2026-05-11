"""Stationary feature engineering and target creation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureSet:
    """Container for feature matrices, target, and stationarity diagnostics."""

    frame: pd.DataFrame
    feature_columns: list[str]
    stationarity_summary: pd.DataFrame


def build_feature_set(
    market_data: pd.DataFrame,
    target_horizon: int = 1,
    short_window: int = 5,
    medium_window: int = 20,
    adf_significance: float = 0.05,
    dropna: bool = True,
    advanced_features: bool = False,
    vix_features: bool = False,
) -> FeatureSet:
    """Create stationary features and a binary next-day target.

    All features are constructed from contemporaneous or lagged information only.
    The optional advanced block adds higher-order momentum, regime, and relative-value
    signals without introducing lookahead bias.
    """

    frame = market_data.copy()
    close = frame["Adj Close"].astype(float)
    volume = frame["Volume"].astype(float)
    benchmark_close = frame["BenchmarkClose"].astype(float)
    simple_return_1d = close.pct_change()
    benchmark_simple_return_1d = benchmark_close.pct_change()

    base_feature_columns, vol_20 = _add_base_features(
        frame=frame,
        close=close,
        volume=volume,
        short_window=short_window,
        medium_window=medium_window,
    )

    advanced_feature_columns: list[str] = []
    if advanced_features:
        advanced_feature_columns = _add_advanced_features(
            frame=frame,
            close=close,
            volume=volume,
            simple_return_1d=simple_return_1d,
            benchmark_simple_return_1d=benchmark_simple_return_1d,
            vol_20=vol_20,
            short_window=short_window,
            medium_window=medium_window,
        )

    vix_feature_columns: list[str] = []
    if vix_features:
        vix_feature_columns = build_vix_features(
            frame=frame,
            short_window=short_window,
            medium_window=medium_window,
        )

    next_return = frame["return_1d"].shift(-target_horizon)
    frame["forward_simple_return_1d"] = simple_return_1d.shift(-target_horizon)
    forward_vol = vol_20.shift(-target_horizon)
    frame["target_return_risk_adjusted"] = next_return / forward_vol.replace(0.0, np.nan)
    frame["target_direction"] = (frame["target_return_risk_adjusted"] > 0.0).astype(int)

    feature_columns = base_feature_columns + advanced_feature_columns + vix_feature_columns
    if dropna:
        frame = frame.dropna(
            subset=feature_columns + ["target_direction", "target_return_risk_adjusted", "forward_simple_return_1d"]
        )

    stationarity = run_stationarity_checks(frame[feature_columns], significance_level=adf_significance)
    return FeatureSet(frame=frame, feature_columns=feature_columns, stationarity_summary=stationarity)


def _add_base_features(
    frame: pd.DataFrame,
    close: pd.Series,
    volume: pd.Series,
    short_window: int,
    medium_window: int,
) -> tuple[list[str], pd.Series]:
    """Add the original stationary baseline features."""

    frame["return_1d"] = np.log(close).diff(1)
    frame["return_5d"] = np.log(close).diff(short_window)
    frame["return_20d"] = np.log(close).diff(medium_window)

    vol_5 = frame["return_1d"].rolling(short_window).std()
    vol_20 = frame["return_1d"].rolling(medium_window).std()
    frame["vol_ratio"] = vol_5 / vol_20.replace(0.0, np.nan)
    frame["momentum_norm"] = frame["return_5d"] / vol_20.replace(0.0, np.nan)

    volume_mean = volume.rolling(medium_window).mean()
    volume_std = volume.rolling(medium_window).std()
    frame["volume_zscore"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)

    intraday_range = (frame["High"] - frame["Low"]) / close.replace(0.0, np.nan)
    frame["range_norm"] = intraday_range / vol_20.replace(0.0, np.nan)

    sma_short = close.rolling(short_window).mean()
    sma_medium = close.rolling(medium_window).mean()
    frame["sma_ratio"] = sma_short / sma_medium.replace(0.0, np.nan) - 1.0

    frame["realized_vol_20"] = vol_20
    frame["close_to_open_gap"] = np.log(frame["Open"] / close.shift(1))
    frame["price_acceleration"] = frame["return_5d"] - frame["return_20d"] / 4.0
    frame["downside_vol_ratio"] = (
        frame["return_1d"].clip(upper=0.0).rolling(short_window).std()
        / frame["return_1d"].clip(upper=0.0).rolling(medium_window).std().replace(0.0, np.nan)
    )

    feature_columns = [
        "return_1d",
        "return_5d",
        "return_20d",
        "vol_ratio",
        "momentum_norm",
        "volume_zscore",
        "range_norm",
        "sma_ratio",
        "realized_vol_20",
        "close_to_open_gap",
        "price_acceleration",
        "downside_vol_ratio",
    ]
    return feature_columns, vol_20


def _add_advanced_features(
    frame: pd.DataFrame,
    close: pd.Series,
    volume: pd.Series,
    simple_return_1d: pd.Series,
    benchmark_simple_return_1d: pd.Series,
    vol_20: pd.Series,
    short_window: int,
    medium_window: int,
) -> list[str]:
    """Add advanced stationary features for richer signal discovery."""

    long_window = max(60, medium_window * 3)
    epsilon = 1e-6

    rsi = _compute_rsi(close=close, period=14)
    frame["rsi_zscore"] = _rolling_zscore(rsi, window=long_window)

    macd_histogram = _compute_macd_histogram(close=close)
    frame["macd_histogram_zscore"] = _rolling_zscore(macd_histogram, window=long_window)

    bollinger_width = _compute_bollinger_band_width(close=close, period=20)
    frame["bollinger_band_width_zscore"] = _rolling_zscore(bollinger_width, window=long_window)

    relative_strength_raw = (1.0 + simple_return_1d) / (1.0 + benchmark_simple_return_1d.replace(0.0, np.nan)) - 1.0
    frame["relative_strength_vs_benchmark"] = _rolling_zscore(relative_strength_raw.replace([np.inf, -np.inf], np.nan), window=long_window)

    overnight_gap_raw = frame["close_to_open_gap"] / vol_20.replace(0.0, np.nan)
    frame["overnight_gap_zscore"] = _rolling_zscore(overnight_gap_raw, window=long_window)

    volume_ratio = volume / volume.rolling(medium_window).mean().replace(0.0, np.nan) - 1.0
    volume_momentum = np.log(volume.replace(0.0, np.nan)).diff(short_window)
    frame["volume_trend_strength"] = _rolling_zscore(volume_ratio + volume_momentum, window=long_window)

    rolling_autocorrelation = frame["return_1d"].rolling(window=medium_window).apply(
        lambda values: pd.Series(values).autocorr(lag=min(5, max(len(values) - 1, 1))),
        raw=False,
    )
    frame["autocorrelation_zscore"] = _rolling_zscore(rolling_autocorrelation, window=long_window)

    vol_10 = frame["return_1d"].rolling(window=max(10, short_window * 2)).std()
    vol_60 = frame["return_1d"].rolling(window=long_window).std()
    frame["volatility_regime"] = vol_10 / vol_60.replace(0.0, np.nan)

    price_acceleration_2nd = frame["return_1d"].diff().diff()
    frame["price_acceleration_2nd"] = price_acceleration_2nd / vol_20.replace(0.0, np.nan)

    benchmark_log_return_1d = np.log(frame["BenchmarkClose"]).diff()
    benchmark_vol_20 = benchmark_log_return_1d.rolling(medium_window).std()
    relative_strength_spy = frame["return_1d"] - benchmark_log_return_1d
    frame["asset_return_vs_spy"] = _rolling_zscore(relative_strength_spy, window=long_window)
    frame["relative_vol_ratio"] = vol_20 / benchmark_vol_20.replace(0.0, np.nan)

    if "VIXClose" in frame.columns:
        implied_vol = frame["VIXClose"] / 100.0 / np.sqrt(252.0)
        frame["vix_relative"] = implied_vol / vol_20.replace(0.0, np.nan)
        optional_features = ["vix_relative"]
    else:
        optional_features = []

    advanced_features = [
        "rsi_zscore",
        "macd_histogram_zscore",
        "bollinger_band_width_zscore",
        "relative_strength_vs_benchmark",
        "overnight_gap_zscore",
        "volume_trend_strength",
        "autocorrelation_zscore",
        "volatility_regime",
        "price_acceleration_2nd",
        "asset_return_vs_spy",
        "relative_vol_ratio",
        *optional_features,
    ]
    return advanced_features


def build_vix_features(
    frame: pd.DataFrame,
    short_window: int,
    medium_window: int,
) -> list[str]:
    """Add stationary volatility-index features derived from the VIX time series."""

    if "VIXClose" not in frame.columns:
        return []

    long_window = max(60, medium_window * 3)
    vix = frame["VIXClose"].astype(float)
    vix_return_1d = np.log(vix).diff()
    vix_return_5d = np.log(vix).diff(short_window)
    realized_vol = frame["realized_vol_20"].replace(0.0, np.nan)

    frame["vix_zscore"] = _rolling_zscore(vix, window=medium_window)
    frame["implied_vs_realized_vol"] = (vix / 100.0) / (realized_vol * np.sqrt(252.0))
    frame["vix_momentum_5d"] = _rolling_zscore(vix_return_5d, window=long_window)
    frame["vix_extreme"] = _rolling_zscore(vix, window=long_window)
    frame["vix_vol_interaction"] = frame["vix_zscore"] * frame["vol_ratio"]
    frame["vix_return_1d"] = vix_return_1d
    frame["vix_return_5d"] = vix_return_5d

    return [
        "vix_zscore",
        "implied_vs_realized_vol",
        "vix_momentum_5d",
        "vix_extreme",
        "vix_vol_interaction",
        "vix_return_1d",
        "vix_return_5d",
    ]


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute a simple RSI series from adjusted close prices."""

    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd_histogram(close: pd.Series) -> pd.Series:
    """Compute the MACD histogram from exponential moving averages."""

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line


def _compute_bollinger_band_width(close: pd.Series, period: int) -> pd.Series:
    """Compute Bollinger Band width as a stationary dispersion proxy."""

    rolling_mean = close.rolling(period).mean()
    rolling_std = close.rolling(period).std()
    upper = rolling_mean + 2.0 * rolling_std
    lower = rolling_mean - 2.0 * rolling_std
    return (upper - lower) / rolling_mean.replace(0.0, np.nan)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Normalize a feature using its own rolling mean and standard deviation."""

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def run_stationarity_checks(features: pd.DataFrame, significance_level: float = 0.05) -> pd.DataFrame:
    """Compute Augmented Dickey-Fuller results for each engineered feature."""

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:  # pragma: no cover - exercised when optional dependency missing.
        raise ImportError(
            "statsmodels is required for ADF stationarity checks. Install dependencies from requirements.txt."
        ) from exc

    rows: list[dict[str, float | bool | str]] = []
    for column in features.columns:
        series = features[column].dropna()
        if len(series) < 10 or series.nunique() <= 1:
            rows.append(
                {
                    "feature": column,
                    "adf_statistic": float("nan"),
                    "p_value": float("nan"),
                    "is_stationary": False,
                }
            )
            continue

        statistic, p_value, *_ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "feature": column,
                "adf_statistic": float(statistic),
                "p_value": float(p_value),
                "is_stationary": bool(p_value < significance_level),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value", na_position="last").reset_index(drop=True)
