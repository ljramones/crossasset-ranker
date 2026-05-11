"""Tests for the active-track market-data cache module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import market_cache
from data.market_cache import (
    MarketCacheConfig,
    build_asset_cache_frame,
    ensure_universe_cache,
    load_cached_ohlcv,
    normalize_ticker_for_filename,
    write_market_cache,
)


def _synthetic_ohlcv(
    *,
    ticker: str,
    start: str = "2020-01-01",
    rows: int = 30,
    base_price: float = 100.0,
    skip_weekends: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows, freq="D")
    if skip_weekends:
        dates = pd.date_range(start, periods=rows, freq="B")
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    returns = rng.normal(0.0, 0.01, size=len(dates))
    closes = base_price * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes * 0.999,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Adj Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, size=len(dates)),
        }
    )


def test_normalize_ticker_for_filename_handles_common_tickers() -> None:
    assert normalize_ticker_for_filename("SPY") == "spy"
    assert normalize_ticker_for_filename("QQQ") == "qqq"
    assert normalize_ticker_for_filename("BTC-USD") == "btc-usd"
    assert normalize_ticker_for_filename("^VIX") == "vix"
    assert normalize_ticker_for_filename("  spy  ") == "spy"


def test_normalize_ticker_for_filename_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        normalize_ticker_for_filename("")
    with pytest.raises(ValueError):
        normalize_ticker_for_filename("^")


def test_write_and_load_market_cache_roundtrips_dates(tmp_path: Path) -> None:
    frame = _synthetic_ohlcv(ticker="SPY", rows=12)
    cache_dir = tmp_path / "multi_asset_cache"

    written = write_market_cache(frame, ticker="SPY", cache_dir=cache_dir)

    assert written.exists()
    assert written.name == "spy_daily.csv"

    loaded = load_cached_ohlcv("SPY", cache_dir=cache_dir)

    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)
    assert pd.api.types.is_datetime64_any_dtype(loaded["Date"])
    assert loaded["Date"].iloc[0] == pd.Timestamp("2020-01-01")
    np.testing.assert_allclose(loaded["Adj Close"].to_numpy(), frame["Adj Close"].to_numpy())


def test_write_market_cache_refuses_empty_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_market_cache(pd.DataFrame(columns=["Date", "Close"]), ticker="SPY", cache_dir=tmp_path)


def test_load_or_fetch_ohlcv_uses_cache_and_writes_metadata_on_fresh_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    config = MarketCacheConfig(cache_dir=cache_dir, start_date="2020-01-01")

    fetch_calls: list[str] = []

    def fake_fetch(ticker: str, *, start_date: str, end_date=None, auto_adjust: bool = False):
        fetch_calls.append(ticker)
        return _synthetic_ohlcv(ticker=ticker, rows=8)

    monkeypatch.setattr(market_cache, "fetch_yfinance_ohlcv", fake_fetch)

    first = market_cache.load_or_fetch_ohlcv("SPY", config=config)
    assert fetch_calls == ["SPY"]
    assert (cache_dir / "spy_daily.csv").exists()
    meta_path = cache_dir / "spy_daily.meta.json"
    assert meta_path.exists()
    metadata = json.loads(meta_path.read_text())
    assert metadata["ticker"] == "SPY"
    assert metadata["row_count"] == len(first)
    assert metadata["first_date"] == "2020-01-01"

    second = market_cache.load_or_fetch_ohlcv("SPY", config=config)
    assert fetch_calls == ["SPY"], "Second call should hit cache, not refetch."
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_build_asset_cache_frame_joins_benchmark_and_ffills_vix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    config = MarketCacheConfig(cache_dir=cache_dir, start_date="2020-01-01")

    asset_frame = _synthetic_ohlcv(ticker="QQQ", rows=15)
    benchmark_frame = _synthetic_ohlcv(ticker="SPY", rows=15, base_price=400.0)
    vix_frame = _synthetic_ohlcv(ticker="^VIX", rows=15, base_price=18.0)
    vix_frame = vix_frame.drop(index=[3, 7]).reset_index(drop=True)

    table = {"QQQ": asset_frame, "SPY": benchmark_frame, "^VIX": vix_frame}

    def fake_fetch(ticker: str, *, start_date: str, end_date=None, auto_adjust: bool = False):
        return table[ticker]

    monkeypatch.setattr(market_cache, "fetch_yfinance_ohlcv", fake_fetch)

    enriched = build_asset_cache_frame(
        "QQQ",
        benchmark_ticker="SPY",
        vix_ticker="^VIX",
        config=config,
    )

    for column in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "BenchmarkClose", "VIXClose"]:
        assert column in enriched.columns

    np.testing.assert_allclose(enriched["BenchmarkClose"].to_numpy(), benchmark_frame["Close"].to_numpy())
    assert enriched["VIXClose"].isna().sum() == 0, "VIXClose should be forward-filled across gaps."
    assert enriched["VIXClose"].iloc[3] == pytest.approx(enriched["VIXClose"].iloc[2])
    assert enriched["Date"].is_monotonic_increasing


def test_build_asset_cache_frame_does_not_backfill_vix_into_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    config = MarketCacheConfig(cache_dir=cache_dir, start_date="2020-01-01")

    asset_frame = _synthetic_ohlcv(ticker="QQQ", rows=10)
    benchmark_frame = _synthetic_ohlcv(ticker="SPY", rows=10, base_price=400.0)
    vix_frame = _synthetic_ohlcv(ticker="^VIX", rows=10, base_price=18.0)
    vix_frame = vix_frame.iloc[3:].reset_index(drop=True)

    table = {"QQQ": asset_frame, "SPY": benchmark_frame, "^VIX": vix_frame}

    def fake_fetch(ticker: str, *, start_date: str, end_date=None, auto_adjust: bool = False):
        return table[ticker]

    monkeypatch.setattr(market_cache, "fetch_yfinance_ohlcv", fake_fetch)

    enriched = build_asset_cache_frame(
        "QQQ",
        benchmark_ticker="SPY",
        vix_ticker="^VIX",
        config=config,
    )

    assert enriched["VIXClose"].iloc[:3].isna().all(), (
        "Rows preceding the first VIX observation must remain NaN — never backfilled."
    )
    assert enriched["VIXClose"].iloc[3:].notna().all()


def test_ensure_universe_cache_returns_normalized_path_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    config = MarketCacheConfig(
        cache_dir=cache_dir,
        start_date="2020-01-01",
        benchmark_ticker="SPY",
        vix_ticker="^VIX",
    )

    def fake_fetch(ticker: str, *, start_date: str, end_date=None, auto_adjust: bool = False):
        return _synthetic_ohlcv(ticker=ticker, rows=6)

    monkeypatch.setattr(market_cache, "fetch_yfinance_ohlcv", fake_fetch)

    paths = ensure_universe_cache(["SPY", "QQQ", "BTC-USD"], config=config)

    assert paths["SPY"].name == "spy_daily.csv"
    assert paths["QQQ"].name == "qqq_daily.csv"
    assert paths["BTC-USD"].name == "btc-usd_daily.csv"
    assert "^VIX" in paths and paths["^VIX"].name == "vix_daily.csv"
    for path in paths.values():
        assert path.exists()
