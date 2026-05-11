"""Active-track market-data cache.

This module is the *only* sanctioned data-fetch entry point for the post-reset
research track (``experiments/`` + ``scripts/``). It deliberately replaces the
missing ``data/market_data.py`` rather than recreating it, so the frozen
legacy CLI path stays dormant.

Responsibilities:

* Fetch raw OHLCV from yfinance and persist deterministic per-asset CSV caches.
* Join benchmark Close and VIX Close onto an asset frame for downstream
  feature engineering, with strictly past-only forward-fill on VIX gaps.
* Write a JSON metadata sidecar next to every fresh fetch so reruns can be
  audited for adjusted-close drift.

Constraints:

* No imports from the legacy ``data.market_data`` module.
* No imports from ``main``, ``utils.experiment``, or ``audit.integrity_audit``.
* All joins are strictly date-aligned. Forward-fill only — never bfill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

CROSS_ASSET_UNIVERSE: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
    "BTC-USD",
)


@dataclass(frozen=True)
class MarketCacheConfig:
    """Configuration for cache layout and fetch defaults."""

    cache_dir: Path = Path("data/multi_asset_cache")
    source: str = "yfinance"
    start_date: str = "2010-01-01"
    benchmark_ticker: str = "SPY"
    vix_ticker: str = "^VIX"
    fetch_date: str | None = None


def normalize_ticker_for_filename(ticker: str) -> str:
    """Return a deterministic filename slug for a yfinance ticker.

    Examples:
        SPY      -> spy
        QQQ      -> qqq
        BTC-USD  -> btc-usd
        ^VIX     -> vix
    """

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")
    cleaned = ticker.strip().lstrip("^").lower()
    if not cleaned:
        raise ValueError(f"ticker {ticker!r} normalizes to an empty filename.")
    return cleaned


def _cache_csv_path(ticker: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{normalize_ticker_for_filename(ticker)}_daily.csv"


def _cache_meta_path(ticker: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{normalize_ticker_for_filename(ticker)}_daily.meta.json"


def _flatten_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse a possible MultiIndex column header into flat OHLCV names."""

    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    flattened_columns: list[str] = []
    for column in frame.columns:
        parts = [str(part) for part in column if str(part) not in ("", ticker)]
        flattened_columns.append(parts[0] if parts else str(column[0]))
    flat = frame.copy()
    flat.columns = flattened_columns
    return flat


def fetch_yfinance_ohlcv(
    ticker: str,
    *,
    start_date: str,
    end_date: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Fetch raw OHLCV from yfinance and return a flat, chronological frame.

    The returned frame has ``Date`` as a column (not the index) and exposes the
    standard ``OHLCV_COLUMNS`` whenever yfinance provides them.
    """

    import yfinance as yf  # Imported lazily so tests can monkeypatch the symbol.

    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        actions=False,
        group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"yfinance returned no rows for ticker={ticker!r} start={start_date!r} "
            f"end={end_date!r}. Refusing to write an empty cache."
        )

    flat = _flatten_yfinance_columns(raw, ticker)
    flat = flat.reset_index()
    if "Date" not in flat.columns:
        if "Datetime" in flat.columns:
            flat = flat.rename(columns={"Datetime": "Date"})
        elif "index" in flat.columns:
            flat = flat.rename(columns={"index": "Date"})
        else:
            raise RuntimeError(
                f"yfinance frame for {ticker!r} is missing a Date/Datetime column; "
                f"got columns={list(flat.columns)!r}."
            )

    flat["Date"] = pd.to_datetime(flat["Date"]).dt.tz_localize(None).dt.normalize()
    available = [column for column in OHLCV_COLUMNS if column in flat.columns]
    flat = flat[available]
    flat = flat.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    if flat.empty:
        raise RuntimeError(
            f"Post-cleanup OHLCV frame for {ticker!r} is empty. Refusing to write cache."
        )
    return flat


def write_market_cache(
    frame: pd.DataFrame,
    *,
    ticker: str,
    cache_dir: Path,
) -> Path:
    """Write a per-asset OHLCV frame to ``cache_dir`` and return its path."""

    if frame is None or frame.empty:
        raise ValueError(f"Refusing to write an empty cache for ticker={ticker!r}.")
    if "Date" not in frame.columns:
        raise ValueError(
            f"Cache frame for {ticker!r} must include a Date column; got {list(frame.columns)!r}."
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = _cache_csv_path(ticker, cache_dir)
    out_frame = frame.copy()
    out_frame["Date"] = pd.to_datetime(out_frame["Date"]).dt.tz_localize(None).dt.normalize()
    out_frame.to_csv(output, index=False, date_format="%Y-%m-%d")
    return output


def load_cached_ohlcv(
    ticker: str,
    *,
    cache_dir: Path = Path("data/multi_asset_cache"),
) -> pd.DataFrame:
    """Load a previously written per-asset cache CSV."""

    path = _cache_csv_path(ticker, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"No cache file found for ticker={ticker!r} at {path}.")
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None).dt.normalize()
    return frame.sort_values("Date").reset_index(drop=True)


def _write_cache_metadata(
    *,
    ticker: str,
    cache_dir: Path,
    frame: pd.DataFrame,
    config: MarketCacheConfig,
    end_date: str | None,
) -> Path:
    """Persist a JSON metadata sidecar describing the most recent fetch."""

    meta_path = _cache_meta_path(ticker, cache_dir)
    fetch_date = config.fetch_date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        import yfinance as yf

        yfinance_version = getattr(yf, "__version__", "unknown")
    except Exception:  # pragma: no cover - defensive
        yfinance_version = "unknown"
    metadata = {
        "ticker": ticker,
        "normalized_filename": normalize_ticker_for_filename(ticker),
        "source": config.source,
        "fetch_date_utc": fetch_date,
        "configured_start_date": config.start_date,
        "requested_start_date": config.start_date,
        "requested_end_date": end_date,
        "row_count": int(len(frame)),
        "first_date": str(pd.to_datetime(frame["Date"].iloc[0]).date()) if len(frame) else None,
        "last_date": str(pd.to_datetime(frame["Date"].iloc[-1]).date()) if len(frame) else None,
        "columns": list(frame.columns),
        "yfinance_version": yfinance_version,
        "drift_warning": (
            "yfinance Adj Close values can be revised retroactively. Compare "
            "first_date/last_date and row_count against prior runs before "
            "treating downstream artifacts as comparable."
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


def load_or_fetch_ohlcv(
    ticker: str,
    *,
    config: MarketCacheConfig,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a per-asset OHLCV frame, fetching from yfinance only when needed."""

    csv_path = _cache_csv_path(ticker, config.cache_dir)
    if csv_path.exists() and not force_refresh:
        return load_cached_ohlcv(ticker, cache_dir=config.cache_dir)

    frame = fetch_yfinance_ohlcv(
        ticker,
        start_date=config.start_date,
        end_date=end_date,
        auto_adjust=False,
    )
    write_market_cache(frame, ticker=ticker, cache_dir=config.cache_dir)
    _write_cache_metadata(
        ticker=ticker,
        cache_dir=config.cache_dir,
        frame=frame,
        config=config,
        end_date=end_date,
    )
    return frame


def build_asset_cache_frame(
    ticker: str,
    *,
    benchmark_ticker: str | None = None,
    vix_ticker: str | None = None,
    config: MarketCacheConfig,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return an asset frame enriched with BenchmarkClose and VIXClose columns.

    Joins are strictly by ``Date``. ``VIXClose`` is forward-filled (past-only)
    after the join to bridge holiday/non-overlap gaps. Raw OHLCV columns from
    the asset side are preserved.
    """

    benchmark = benchmark_ticker or config.benchmark_ticker
    vix = vix_ticker or config.vix_ticker

    asset_frame = load_or_fetch_ohlcv(
        ticker,
        config=config,
        end_date=end_date,
        force_refresh=force_refresh,
    )
    benchmark_frame = load_or_fetch_ohlcv(
        benchmark,
        config=config,
        end_date=end_date,
        force_refresh=force_refresh,
    )
    vix_frame = load_or_fetch_ohlcv(
        vix,
        config=config,
        end_date=end_date,
        force_refresh=force_refresh,
    )

    benchmark_join = benchmark_frame[["Date", "Close"]].rename(columns={"Close": "BenchmarkClose"})
    vix_join = vix_frame[["Date", "Close"]].rename(columns={"Close": "VIXClose"})

    merged = asset_frame.merge(benchmark_join, on="Date", how="left")
    merged = merged.merge(vix_join, on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["VIXClose"] = merged["VIXClose"].ffill()
    return merged


def ensure_universe_cache(
    tickers: Iterable[str],
    *,
    config: MarketCacheConfig,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Path]:
    """Make sure every ticker in ``tickers`` has a cache CSV on disk.

    Always also ensures the benchmark and VIX tickers from ``config`` are
    present so downstream feature builds can proceed without an extra fetch.
    """

    paths: dict[str, Path] = {}
    universe = list(dict.fromkeys([*tickers, config.benchmark_ticker, config.vix_ticker]))
    for ticker in universe:
        load_or_fetch_ohlcv(
            ticker,
            config=config,
            end_date=end_date,
            force_refresh=force_refresh,
        )
        paths[ticker] = _cache_csv_path(ticker, config.cache_dir)
    return paths
