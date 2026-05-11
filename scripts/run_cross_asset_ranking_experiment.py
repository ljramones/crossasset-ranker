"""Standalone CLI for the cross-asset ranking feasibility prototype.

This wrapper does *not* download data by default. Pass --prepare-missing to
allow the underlying ``prepare_single_asset_feature_frame`` call to fetch
missing raw OHLCV via yfinance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import gmtime, strftime

import pandas as pd

from experiments.cross_asset_ranking_experiment import (
    CrossAssetRankingConfig,
    FEATURE_NORMALIZATION_CHOICES,
    KNOWN_MODELS,
    load_prepared_asset_frames,
    run_cross_asset_ranking_experiment,
)


DEFAULT_ASSETS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    parser.add_argument("--cache-dir", default="data/multi_asset_cache")
    parser.add_argument("--output-dir", default="results/cross_asset_ranking_feasibility")
    parser.add_argument("--forward-horizon", type=int, default=20)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=756)
    parser.add_argument("--val-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--step-size", type=int, default=252)
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--random-null-runs", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-purpose", choices=["plumbing", "diagnostic", "decision_grade"], default="plumbing")
    parser.add_argument("--decision-grade", action="store_true")
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=1,
        help="Rebalance allocations every N trading days (default 1 = daily). Weights are held between rebalance dates.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Restrict to a subset of model names. Default: all of "
            "momentum_baseline, linear_regression, hist_gradient_boosting."
        ),
    )
    parser.add_argument(
        "--feature-normalization",
        choices=list(FEATURE_NORMALIZATION_CHOICES),
        default="none",
        help=(
            "Per-asset train-only feature z-scoring. With 'per_asset_train_zscore' "
            "the model sees standardized features (per asset, using only that "
            "split's train rows for mean/std); allocation returns still use raw "
            "return_1d. Default 'none'."
        ),
    )
    parser.add_argument(
        "--prepare-missing",
        action="store_true",
        help="Allow prepare_single_asset_feature_frame to fetch missing raw caches via yfinance.",
    )
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--vix", default="^VIX")
    return parser


def _config_from_args(args: argparse.Namespace) -> CrossAssetRankingConfig:
    defaults = CrossAssetRankingConfig()
    if args.models:
        valid = set(KNOWN_MODELS)
        unknown = sorted(set(args.models) - valid)
        if unknown:
            raise SystemExit(f"Unknown --models entries: {unknown}. Valid: {sorted(valid)}")
        model_names = tuple(args.models)
    else:
        model_names = defaults.model_names
    return CrossAssetRankingConfig(
        assets=tuple(args.assets),
        forward_horizon=int(args.forward_horizon),
        vol_window=int(args.vol_window),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        step_size=int(args.step_size),
        transaction_cost_bps=float(args.transaction_cost_bps),
        top_k_values=tuple(int(k) for k in args.top_k),
        model_names=model_names,
        random_null_runs=int(args.random_null_runs),
        random_state=int(args.random_state),
        run_purpose=str(args.run_purpose),
        decision_grade=bool(args.decision_grade or args.run_purpose == "decision_grade"),
        rebalance_every=int(args.rebalance_every),
        feature_normalization=str(args.feature_normalization),
    )


def _print_dry_run(args: argparse.Namespace, timestamp: str) -> None:
    print(f"[{timestamp}] cross_asset_ranking DRY RUN — no data load, no model fit, no output writes.")
    print(f"  assets:               {args.assets}")
    print(f"  cache_dir:            {args.cache_dir}")
    print(f"  output_dir:           {args.output_dir}")
    print(f"  forward_horizon:      {args.forward_horizon}")
    print(f"  vol_window:           {args.vol_window}")
    print(f"  train_size:           {args.train_size}")
    print(f"  val_size:             {args.val_size}")
    print(f"  test_size:            {args.test_size}")
    print(f"  step_size:            {args.step_size}")
    print(f"  transaction_cost_bps: {args.transaction_cost_bps}")
    print(f"  top_k:                {args.top_k}")
    print(f"  random_null_runs:     {args.random_null_runs}")
    print(f"  random_state:         {args.random_state}")
    print(f"  run_purpose:          {args.run_purpose}")
    print(f"  decision_grade:       {args.decision_grade}")
    print(f"  prepare_missing:      {args.prepare_missing}")
    print(f"  rebalance_every:      {args.rebalance_every}")
    print(f"  models:               {args.models or 'default (all 3)'}")
    print(f"  feature_normalization:{args.feature_normalization}")


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows_"
    frame = frame.head(max_rows).copy()
    columns = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_output_bundle(
    *,
    output_dir: Path,
    timestamp: str,
    result: dict,
    config: CrossAssetRankingConfig,
    data_downloaded: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": output_dir / f"cross_asset_ranking_summary_{timestamp}.csv",
        "fold_details": output_dir / f"cross_asset_ranking_fold_details_{timestamp}.csv",
        "scored_panel": output_dir / f"cross_asset_ranking_scored_panel_{timestamp}.csv",
        "allocations": output_dir / f"cross_asset_ranking_allocations_{timestamp}.csv",
        "portfolio_returns": output_dir / f"cross_asset_ranking_portfolio_returns_{timestamp}.csv",
        "random_nulls": output_dir / f"cross_asset_ranking_random_nulls_{timestamp}.csv",
        "null_pvalues": output_dir / f"cross_asset_ranking_null_pvalues_{timestamp}.csv",
        "report": output_dir / f"cross_asset_ranking_report_{timestamp}.md",
        "metadata": output_dir / f"cross_asset_ranking_metadata_{timestamp}.json",
    }
    for key in ("summary", "fold_details", "scored_panel", "allocations", "portfolio_returns", "random_nulls", "null_pvalues"):
        frame = result.get(key)
        if frame is not None and not frame.empty:
            frame.to_csv(paths[key], index=False)
        else:
            pd.DataFrame().to_csv(paths[key], index=False)

    metadata = result.get("metadata", {})
    metadata = {
        **metadata,
        "generated_at_utc": timestamp,
        "data_downloaded": bool(data_downloaded),
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    summary = result.get("summary", pd.DataFrame())
    null_pvalues = result.get("null_pvalues", pd.DataFrame())
    fold_details = result.get("fold_details", pd.DataFrame())

    report_lines = [
        "# Cross-Asset Ranking Feasibility Report",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Run purpose: `{config.run_purpose}`",
        f"- Decision grade: `{config.decision_grade}`",
        "- This is a feasibility prototype. **Not** a validated alpha model and not production-ready.",
        "",
        "## Scope and safety",
        "",
        "- main.py used: `False`",
        "- prepare_experiment(...) used: `False`",
        "- Old model-zoo path used: `False`",
        "- Optuna used: `False`",
        "- Deep / sequence models used: `False`",
        "- Stacking ensembles used: `False`",
        "",
        "## Universe",
        "",
        f"`{', '.join(config.assets)}`",
        "",
        "## Target",
        "",
        f"`forward_{config.forward_horizon}d_risk_adjusted_return = forward_{config.forward_horizon}d_return / trailing_{config.vol_window}d_realized_vol`",
        "",
        "## Models",
        "",
        f"`{', '.join(config.model_names)}`",
        "",
        "## Top-k policies",
        "",
        f"`{', '.join(str(k) for k in config.top_k_values)}`",
        "",
        "## Summary (mean across folds)",
        "",
        _markdown_table(summary),
        "",
        "## Fold details",
        "",
        _markdown_table(fold_details),
        "",
        "## Random top-k null p-values",
        "",
        _markdown_table(null_pvalues),
        "",
        "## Stop/go interpretation (plumbing-only)",
        "",
        "- This is a plumbing run. Do not read economic conclusions from a single split or a tiny null grid.",
        "- For a decision-grade run, raise `--random-null-runs`, set `--step-size` to give multiple folds, and re-run with `--run-purpose decision_grade`.",
    ]
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")
    return paths


def _run_execute(args: argparse.Namespace, timestamp: str) -> int:
    config = _config_from_args(args)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    print(f"[{timestamp}] cross_asset_ranking --execute")
    print(f"  loading prepared frames for: {list(config.assets)}")
    frames = load_prepared_asset_frames(
        assets=config.assets,
        cache_dir=cache_dir,
        benchmark_ticker=args.benchmark,
        vix_ticker=args.vix,
        start_date=args.start_date,
        end_date=args.end_date,
        prepare_missing=bool(args.prepare_missing),
    )
    for asset, frame in frames.items():
        if frame.empty:
            print(f"    {asset:<10} rows=     0 dates=<empty>")
        else:
            print(
                f"    {asset:<10} rows={len(frame):>6} "
                f"dates={frame['date'].iloc[0]} -> {frame['date'].iloc[-1]}"
            )

    result = run_cross_asset_ranking_experiment(asset_frames=frames, config=config)
    paths = write_output_bundle(
        output_dir=output_dir,
        timestamp=timestamp,
        result=result,
        config=config,
        data_downloaded=bool(args.prepare_missing),
    )

    print()
    print("Output files:")
    for key, path in paths.items():
        print(f"  {key:<18} {path}")
    print()
    print("Summary (mean across folds):")
    print(result["summary"].to_string(index=False))
    if not result.get("null_pvalues", pd.DataFrame()).empty:
        print()
        print("Random top-k null p-values:")
        print(result["null_pvalues"].to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    timestamp = strftime("%Y%m%dT%H%M%SZ", gmtime())

    if args.dry_run:
        _print_dry_run(args, timestamp)
        return 0
    return _run_execute(args, timestamp)


if __name__ == "__main__":
    raise SystemExit(main())
