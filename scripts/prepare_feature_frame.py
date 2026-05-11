"""Active-track CLI: prepare a feature frame CSV for downstream experiments.

This script is the post-reset replacement for the broken legacy preparation
path. It deliberately avoids importing ``main``, ``utils.experiment``,
``audit.integrity_audit``, or ``data.market_data`` so it cannot accidentally
revive the frozen model-zoo CLI. Use the new ``data.market_cache`` for all
fetch/cache work.

Output contract (when sources permit):

    date, Open, High, Low, Close, Adj Close, Volume,
    BenchmarkClose, VIXClose,
    return_1d, return_5d, return_20d, vol_ratio, momentum_norm,
    volume_zscore, range_norm, sma_ratio, realized_vol_20,
    benchmark_return_1d, forward_simple_return_1d,
    target_drawdown_event_<H>d_<T>pct ...
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from time import gmtime, strftime
from typing import Iterable

import pandas as pd

from data.market_cache import (
    CROSS_ASSET_UNIVERSE,
    MarketCacheConfig,
    build_asset_cache_frame,
)
from evaluation.drawdown_labels import (
    append_drawdown_label_grid,
    get_drawdown_label_columns,
)
from features.feature_engineering import build_feature_set


REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "BenchmarkClose",
    "return_1d",
    "return_5d",
    "return_20d",
    "vol_ratio",
    "momentum_norm",
    "volume_zscore",
    "range_norm",
    "sma_ratio",
    "realized_vol_20",
    "benchmark_return_1d",
    "forward_simple_return_1d",
)


def prepare_single_asset_feature_frame(
    *,
    ticker: str,
    benchmark_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    cache_dir: Path = Path("data/multi_asset_cache"),
    include_regime_features: bool = False,
    include_drawdown_labels: bool = True,
    horizons: tuple[int, ...] = (10, 20),
    thresholds: tuple[float, ...] = (-0.02, -0.03, -0.05),
    advanced_features: bool = True,
    vix_features: bool = True,
    output_csv: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build a prepared feature frame for one asset and optionally write it."""

    config = MarketCacheConfig(
        cache_dir=Path(cache_dir),
        start_date=start_date,
        benchmark_ticker=benchmark_ticker,
        vix_ticker=vix_ticker,
    )

    raw_frame = build_asset_cache_frame(
        ticker,
        benchmark_ticker=benchmark_ticker,
        vix_ticker=vix_ticker,
        config=config,
        end_date=end_date,
        force_refresh=force_refresh,
    )

    market_frame = raw_frame.copy()
    # Restrict to the benchmark's native calendar before feature engineering.
    # Cross-asset assets like BTC-USD trade on weekends when SPY/^VIX do not;
    # keeping those rows would force benchmark-derived features to be NaN and
    # cascade through long rolling windows. Matching to the benchmark's calendar
    # is also what the cross-asset panel inner-join would do downstream — doing
    # it here keeps the prepared frame self-consistent for any consumer.
    benchmark_observed = market_frame["BenchmarkClose"].notna()
    market_frame = market_frame.loc[benchmark_observed].reset_index(drop=True)
    market_frame["benchmark_return_1d"] = market_frame["BenchmarkClose"].astype(float).pct_change()
    market_frame_indexed = market_frame.set_index("Date")

    feature_set = build_feature_set(
        market_frame_indexed,
        advanced_features=advanced_features,
        vix_features=vix_features,
    )
    prepared = feature_set.frame.copy()
    prepared["benchmark_return_1d"] = market_frame_indexed["benchmark_return_1d"].reindex(prepared.index)
    prepared = prepared.reset_index().rename(columns={"Date": "date", "index": "date"})

    if include_regime_features:
        if "regime_id" in prepared.columns:
            from features.regime_features import add_regime_features

            add_regime_features(prepared)
        else:
            warnings.warn(
                "include_regime_features=True but no regime_id column is present. "
                "Regime fitting must be done fold-safely inside the experiment that "
                "consumes this frame; skipping in the prepare step.",
                stacklevel=2,
            )

    if include_drawdown_labels:
        prepared = append_drawdown_label_grid(
            prepared,
            price_column="Adj Close",
            horizons=horizons,
            thresholds=thresholds,
        )

    if output_csv is not None:
        write_prepared_feature_frame(prepared, Path(output_csv))

    return prepared


def validate_prepared_feature_frame(frame: pd.DataFrame) -> dict:
    """Return a structured summary of frame health for CLI reporting."""

    required = list(REQUIRED_OUTPUT_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    target_columns = [column for column in frame.columns if str(column).startswith("target_")]
    drawdown_targets = get_drawdown_label_columns(frame)
    excluded = {
        "date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "BenchmarkClose",
        "VIXClose",
        "forward_simple_return_1d",
        "benchmark_return_1d",
        "target_return_risk_adjusted",
        "target_direction",
    }
    excluded.update({c for c in frame.columns if str(c).startswith("future_max_drawdown_")})
    excluded.update({c for c in frame.columns if str(c).startswith("target_drawdown_event_")})
    feature_candidates = [c for c in frame.columns if c not in excluded]

    notes: list[str] = []
    if "VIXClose" not in frame.columns:
        notes.append("VIXClose missing — VIX-derived features will be absent.")
    if not drawdown_targets:
        notes.append("No drawdown event labels present.")
    if frame.empty:
        notes.append("Frame is empty after feature engineering (likely too few rows).")

    return {
        "is_valid": not missing and not frame.empty,
        "row_count": int(len(frame)),
        "date_start": (
            str(pd.to_datetime(frame["date"].iloc[0]).date()) if "date" in frame.columns and len(frame) else None
        ),
        "date_end": (
            str(pd.to_datetime(frame["date"].iloc[-1]).date()) if "date" in frame.columns and len(frame) else None
        ),
        "missing_required_columns": missing,
        "target_columns": target_columns,
        "drawdown_label_columns": drawdown_targets,
        "feature_candidate_count": len(feature_candidates),
        "notes": notes,
    }


def write_prepared_feature_frame(frame: pd.DataFrame, output_csv: Path) -> Path:
    """Write the prepared feature frame to ``output_csv`` and return the path."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_frame = frame.copy()
    if "date" in out_frame.columns:
        out_frame["date"] = pd.to_datetime(out_frame["date"]).dt.tz_localize(None).dt.normalize()
        out_frame.to_csv(output_csv, index=False, date_format="%Y-%m-%d")
    else:
        out_frame.to_csv(output_csv, index=False)
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print intended actions without fetching or writing.")
    mode.add_argument("--execute", action="store_true", help="Fetch data and write the prepared CSV.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--benchmark", dest="benchmark_ticker", default="SPY")
    parser.add_argument("--vix", dest="vix_ticker", default="^VIX")
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--cache-dir", default="data/multi_asset_cache")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--include-drawdown-labels",
        action="store_true",
        help="Append target_drawdown_event_<H>d_<T>pct columns.",
    )
    parser.add_argument(
        "--no-advanced-features",
        action="store_true",
        help="Skip the advanced stationary feature block (default: include).",
    )
    parser.add_argument(
        "--no-vix-features",
        action="store_true",
        help="Skip VIX-derived features even when VIXClose is present.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[10, 20],
        help="Drawdown horizons in trading days.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[-0.02, -0.03, -0.05],
        help="Drawdown thresholds (negative floats).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch from yfinance even when a cache file already exists.",
    )
    parser.add_argument(
        "--ensure-universe",
        nargs="*",
        default=None,
        help=(
            "Optional list of additional tickers whose caches should be ensured "
            "(does not produce prepared CSVs). Pass without arguments to use the "
            f"default cross-asset universe: {', '.join(CROSS_ASSET_UNIVERSE)}."
        ),
    )
    return parser


def _print_dry_run_summary(args: argparse.Namespace) -> None:
    print("prepare_feature_frame DRY RUN — no data will be fetched and no files will be written.")
    print(f"  ticker:                 {args.ticker}")
    print(f"  benchmark:              {args.benchmark_ticker}")
    print(f"  vix:                    {args.vix_ticker}")
    print(f"  start_date:             {args.start_date}")
    print(f"  end_date:               {args.end_date}")
    print(f"  cache_dir:              {args.cache_dir}")
    print(f"  output_csv:             {args.output_csv}")
    print(f"  include_drawdown_labels:{args.include_drawdown_labels}")
    print(f"  horizons:               {args.horizons}")
    print(f"  thresholds:             {args.thresholds}")
    print(f"  advanced_features:      {not args.no_advanced_features}")
    print(f"  vix_features:           {not args.no_vix_features}")
    print(f"  force_refresh:          {args.force_refresh}")
    if args.ensure_universe is not None:
        universe = list(args.ensure_universe) or list(CROSS_ASSET_UNIVERSE)
        print(f"  ensure_universe:        {universe}")


def _run_execute(args: argparse.Namespace) -> int:
    if not args.output_csv:
        print("--execute requires --output-csv to be set.", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir)
    horizons: Iterable[int] = tuple(args.horizons)
    thresholds: Iterable[float] = tuple(args.thresholds)

    if args.ensure_universe is not None:
        from data.market_cache import ensure_universe_cache

        universe = list(args.ensure_universe) or list(CROSS_ASSET_UNIVERSE)
        config = MarketCacheConfig(
            cache_dir=cache_dir,
            start_date=args.start_date,
            benchmark_ticker=args.benchmark_ticker,
            vix_ticker=args.vix_ticker,
        )
        paths = ensure_universe_cache(
            universe,
            config=config,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
        )
        for ticker, path in paths.items():
            print(f"  cache: {ticker:<10} -> {path}")

    frame = prepare_single_asset_feature_frame(
        ticker=args.ticker,
        benchmark_ticker=args.benchmark_ticker,
        vix_ticker=args.vix_ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=cache_dir,
        include_drawdown_labels=args.include_drawdown_labels,
        horizons=tuple(horizons),
        thresholds=tuple(thresholds),
        advanced_features=not args.no_advanced_features,
        vix_features=not args.no_vix_features,
        output_csv=Path(args.output_csv),
        force_refresh=args.force_refresh,
    )

    summary = validate_prepared_feature_frame(frame)
    print("Prepared feature frame written.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["is_valid"]:
        print("WARNING: prepared frame failed validation.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    timestamp = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())

    if args.dry_run:
        print(f"[{timestamp}] prepare_feature_frame")
        _print_dry_run_summary(args)
        return 0

    print(f"[{timestamp}] prepare_feature_frame --execute")
    return _run_execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
