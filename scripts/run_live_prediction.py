"""Live prediction script for cross-asset ranking deployment.

Orchestrates existing pipeline functions to produce predictions for any
as-of date using a 1008-day rolling training window (train + val of the
campaign walk-forward merged). Supports live mode (predict today) and
historical replay (build a forward-walk track record).

Profiles intentionally exclude Ridge variants — see
``docs/RIDGE_BASELINE_RESULTS.md`` for the empirical justification. Any
future model substitution must respect LambdaRank's per-date grouping
(pairwise loss constrained within each date's universe); pooled regression
models on per-asset-normalized features fail the cross-date sign-coherence
problem documented in that patch.

No new feature engineering or model code lives here — every transformation
is imported from ``experiments.cross_asset_ranking_experiment`` or
``evaluation.cross_asset_ranking``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import gmtime, strftime
from typing import Any

import numpy as np
import pandas as pd

from evaluation.cross_asset_ranking import (
    build_top_k_allocations,
    normalize_features_per_asset_train_only,
    select_cross_asset_feature_columns,
)
from experiments.cross_asset_ranking_experiment import (
    CrossAssetRankingConfig,
    DEFAULT_ASSET_COLUMN,
    DEFAULT_DATE_COLUMN,
    _build_panel,
    _score_with_lambdarank,
    load_prepared_asset_frames,
    target_column_for_horizon,
)


# 18-asset universe from the campaign source runs (must match exactly for
# any forward-walk validation against the experiment runner's allocations).
DEPLOYABLE_UNIVERSE: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "DBA", "UUP",
    "VNQ", "BTC-USD",
)

TRAINING_WINDOW_DAYS: int = 1008  # train (756) + val (252) merged for live use
VIX_ZSCORE_WARMUP_DAYS: int = 252


@dataclass(frozen=True)
class Profile:
    name: str
    forward_horizon: int
    top_k: int
    include_cross_sectional_features: bool
    include_regime_interactions: bool
    feature_normalization: str = "per_asset_train_zscore"
    model_name: str = "lambdarank"
    description: str = ""

    def signature(self) -> str:
        """Stable 12-char hash of deployment-relevant fields.

        Includes the universe and training window so any change to those
        produces a different signature (and a clear cohort break in the log).
        """
        payload = {
            "name": self.name,
            "forward_horizon": int(self.forward_horizon),
            "top_k": int(self.top_k),
            "include_cross_sectional_features": bool(self.include_cross_sectional_features),
            "include_regime_interactions": bool(self.include_regime_interactions),
            "feature_normalization": self.feature_normalization,
            "model_name": self.model_name,
            "universe": list(DEPLOYABLE_UNIVERSE),
            "training_window_days": TRAINING_WINDOW_DAYS,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


PROFILES: dict[str, Profile] = {
    "v3_5d_top2": Profile(
        name="v3_5d_top2",
        forward_horizon=5,
        top_k=2,
        include_cross_sectional_features=True,
        include_regime_interactions=True,
        description="5d horizon, v1 xs features + regime interactions, top-2 (economic high-water mark)",
    ),
    "20d_top1": Profile(
        name="20d_top1",
        forward_horizon=20,
        top_k=1,
        include_cross_sectional_features=True,
        include_regime_interactions=False,
        description="20d horizon, v1 xs features, top-1 (cost-efficient alternative)",
    ),
    "20d_top2": Profile(
        name="20d_top2",
        forward_horizon=20,
        top_k=2,
        include_cross_sectional_features=True,
        include_regime_interactions=False,
        description="20d horizon, v1 xs features, top-2 (concentration hedge of 20d_top1)",
    ),
    "v1_5d_top2": Profile(
        name="v1_5d_top2",
        forward_horizon=5,
        top_k=2,
        include_cross_sectional_features=True,
        include_regime_interactions=False,
        description="5d horizon, v1 xs features, top-2 (rank-quality reference)",
    ),
}


def _profile_to_config(profile: Profile) -> CrossAssetRankingConfig:
    """Build a CrossAssetRankingConfig matching the profile's source-run flags.

    The walk-forward split sizes are kept at campaign defaults so that
    ``_build_panel`` and friends see consistent values, but live prediction
    does not use ``generate_walk_forward_splits`` — training window slicing
    is done explicitly per as-of date.
    """
    return CrossAssetRankingConfig(
        assets=DEPLOYABLE_UNIVERSE,
        forward_horizon=profile.forward_horizon,
        vol_window=20,
        train_size=756,
        val_size=252,
        test_size=252,
        step_size=252,
        transaction_cost_bps=2.0,
        top_k_values=(profile.top_k,),
        model_names=(profile.model_name,),
        random_null_runs=0,
        random_state=42,
        run_purpose="diagnostic",
        decision_grade=False,
        annualization_factor=252,
        rebalance_every=profile.forward_horizon,
        feature_normalization=profile.feature_normalization,
        include_cross_sectional_features=profile.include_cross_sectional_features,
        include_regime_interactions=profile.include_regime_interactions,
        regime_architecture="none",
        regime_min_train_days=120,
    )


# ----------------------------------------------------------------------
# Prediction record + I/O
# ----------------------------------------------------------------------

@dataclass
class PredictionRecord:
    as_of_date: str
    profile_name: str
    profile_signature: str
    forward_horizon: int
    top_k: int
    universe: list[str]
    scores: dict[str, float | None]
    top_k_picks: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    training_window_start: str
    training_window_end: str
    training_n_samples: int
    training_n_dates: int
    cache_last_bar_date: str
    fetched_fresh_data: bool
    nan_feature_diagnostics: dict[str, int]
    feature_values_at_live_date: dict[str, dict[str, float | None]]
    generated_at_utc: str


def _save_record(record: PredictionRecord, *, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    json_path = output_dir / f"{record.as_of_date}_{record.profile_name}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    log_path = output_dir / "predictions_log.jsonl"
    with log_path.open("a") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), sort_keys=False) + "\n")
    return json_path, log_path


def _format_stdout(record: PredictionRecord) -> str:
    lines: list[str] = []
    lines.append(f"=== Live Prediction for {record.as_of_date} ===")
    lines.append(f"Profile: {record.profile_name}  (signature: {record.profile_signature})")
    lines.append(f"Horizon: {record.forward_horizon} trading days")
    picks_str = ", ".join(
        f"{p['asset']} (score: {p['score']:.4f}, weight: {p['weight']:.3f})"
        for p in record.top_k_picks
    )
    lines.append(f"Top-{record.top_k} Pick(s): {picks_str}")
    next_rebal = (
        pd.to_datetime(record.as_of_date)
        + pd.tseries.offsets.BDay(record.forward_horizon)
    ).strftime("%Y-%m-%d")
    lines.append(f"Next rebalance date (business-day offset, approximate): {next_rebal}")
    lines.append("")
    lines.append("Full ranking:")
    scored = sorted(
        record.scores.items(),
        key=lambda kv: (-(kv[1] if kv[1] is not None else -1e18), kv[0]),
    )
    for i, (asset, score) in enumerate(scored, 1):
        score_str = "      n/a" if score is None else f"{score:8.4f}"
        lines.append(f" {i:2d}. {asset:8s}  (score: {score_str})")
    lines.append("")
    lines.append("Top features (LambdaRank gain importance, top-10):")
    for f in record.feature_importance:
        lines.append(f"  - {f['feature']:42s}  gain={f.get('gain', 0.0):.1f}")
    lines.append("")
    lines.append(
        f"Training window: {record.training_window_start} to {record.training_window_end} "
        f"({record.training_n_samples} samples across {record.training_n_dates} dates)"
    )
    lines.append(
        f"Cache freshness: last bar {record.cache_last_bar_date} "
        f"(fetched_fresh_data={record.fetched_fresh_data})"
    )
    if record.nan_feature_diagnostics:
        lines.append(f"NaN feature diagnostics: {record.nan_feature_diagnostics}")
    else:
        lines.append("NaN feature diagnostics: clean (no NaN feature values at live date)")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def _load_frames(
    *,
    cache_dir: Path,
    end_date: pd.Timestamp,
    start_date: pd.Timestamp | None = None,
    fetch_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load per-asset prepared feature frames through ``end_date``.

    Pulls enough history for the 1008-day training window plus VIX z-score
    warmup plus a generous buffer for cross-sectional rank computation.
    With ``fetch_missing=True`` the underlying caches will be created via
    yfinance for any asset that is not already cached. Already-cached
    assets are read as-is (this script does NOT force-refresh the cache —
    run ``scripts/prepare_feature_frame.py`` if a hard refresh is needed).
    """
    if start_date is None:
        # ~6 years of trading history at 252 days/year + VIX warmup + buffer.
        # Using calendar days approximation generous enough to ensure coverage.
        history_calendar_days = int((TRAINING_WINDOW_DAYS + VIX_ZSCORE_WARMUP_DAYS + 60) * 365 / 252)
        start_date = end_date - pd.Timedelta(days=history_calendar_days)
    return load_prepared_asset_frames(
        assets=DEPLOYABLE_UNIVERSE,
        cache_dir=cache_dir,
        benchmark_ticker="SPY",
        vix_ticker="^VIX",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        prepare_missing=fetch_missing,
    )


def _truncate_frames(
    frames: dict[str, pd.DataFrame],
    *,
    as_of_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Strict no-look-ahead: drop rows with date > as_of_date in every frame."""
    out: dict[str, pd.DataFrame] = {}
    for asset, frame in frames.items():
        dates = pd.to_datetime(frame["date"])
        out[asset] = frame[dates <= as_of_date].copy()
    return out


def _common_trading_dates(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Intersection of date sets across all asset frames (matches the
    strict-inner-join calendar used by ``build_cross_asset_panel``)."""
    sets = [set(pd.to_datetime(f["date"])) for f in frames.values()]
    common = sorted(set.intersection(*sets)) if sets else []
    return pd.DatetimeIndex(common)


# ----------------------------------------------------------------------
# Core prediction workflow
# ----------------------------------------------------------------------

def _select_training_window(
    panel_dates: pd.DatetimeIndex,
    *,
    live_date: pd.Timestamp,
    window_days: int = TRAINING_WINDOW_DAYS,
) -> pd.DatetimeIndex:
    """Most-recent ``window_days`` panel dates STRICTLY BEFORE ``live_date``.

    The live date itself is held out (its forward target is unknown and
    not used for fitting). Training rows whose forward target is NaN are
    dropped inside ``_fit_lambdarank_on_panel`` regardless of selection here.
    """
    prior = panel_dates[panel_dates < live_date]
    if len(prior) < window_days:
        raise RuntimeError(
            f"Insufficient training history: panel has {len(prior)} dates before "
            f"{live_date.date()}, need {window_days}. Extend the data load window."
        )
    return prior[-window_days:]


def _predict_for_as_of(
    *,
    profile: Profile,
    as_of_date: pd.Timestamp,
    frames_truncated: dict[str, pd.DataFrame],
    cache_last_bar_date: pd.Timestamp,
    fetched_fresh_data: bool,
) -> PredictionRecord:
    """Single-date prediction. ``frames_truncated`` must already be sliced
    to <= as_of_date (no future leakage)."""
    config = _profile_to_config(profile)
    panel = _build_panel(frames_truncated, config=config)
    panel[DEFAULT_DATE_COLUMN] = pd.to_datetime(panel[DEFAULT_DATE_COLUMN])
    if panel.empty:
        raise RuntimeError(
            f"Empty panel after build_cross_asset_panel for as-of {as_of_date.date()}."
        )

    target_column = target_column_for_horizon(profile.forward_horizon)
    feature_columns = select_cross_asset_feature_columns(panel, target_col=target_column)
    if not feature_columns:
        raise RuntimeError("Feature column inference returned an empty list.")

    panel_dates = pd.DatetimeIndex(sorted(panel[DEFAULT_DATE_COLUMN].unique()))
    if as_of_date not in panel_dates:
        # Fall back to the latest available trading day at or before as_of.
        available = panel_dates[panel_dates <= as_of_date]
        if len(available) == 0:
            raise RuntimeError(
                f"No panel date at or before {as_of_date.date()}; cache likely too short."
            )
        live_date = available[-1]
    else:
        live_date = as_of_date

    train_dates = _select_training_window(panel_dates, live_date=live_date)

    normalized = normalize_features_per_asset_train_only(
        panel,
        train_dates=train_dates,
        feature_columns=feature_columns,
        date_col=DEFAULT_DATE_COLUMN,
        asset_col=DEFAULT_ASSET_COLUMN,
    )
    normalized[DEFAULT_DATE_COLUMN] = pd.to_datetime(normalized[DEFAULT_DATE_COLUMN])

    train_panel = normalized[normalized[DEFAULT_DATE_COLUMN].isin(train_dates)].copy()
    live_panel = normalized[normalized[DEFAULT_DATE_COLUMN] == live_date].copy()
    if live_panel.empty:
        raise RuntimeError(f"Live panel empty at {live_date.date()}.")
    # Reset indexes so the position-aligned series returned by
    # _score_with_lambdarank lines up with live_panel rows.
    live_panel = live_panel.reset_index(drop=True)
    train_panel = train_panel.reset_index(drop=True)

    scores, importance_rows = _score_with_lambdarank(
        train_panel=train_panel,
        test_panel=live_panel,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    if scores.isna().all():
        raise RuntimeError(
            "LambdaRank produced all-NaN scores — likely insufficient training "
            "rows after target-NaN filtering."
        )

    scored = live_panel[[DEFAULT_DATE_COLUMN, DEFAULT_ASSET_COLUMN]].copy()
    scored["score"] = scores.values
    alloc = build_top_k_allocations(
        scored,
        score_col="score",
        k=profile.top_k,
        date_col=DEFAULT_DATE_COLUMN,
        asset_col=DEFAULT_ASSET_COLUMN,
    )

    asset_scores: dict[str, float | None] = {
        str(row[DEFAULT_ASSET_COLUMN]): (float(row["score"]) if pd.notna(row["score"]) else None)
        for _, row in scored.iterrows()
    }

    picks_df = alloc[alloc["weight"] > 0].sort_values("rank")
    top_k_picks = [
        {
            "asset": str(row[DEFAULT_ASSET_COLUMN]),
            "score": float(row["score"]) if pd.notna(row["score"]) else None,
            "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
            "weight": float(row["weight"]),
        }
        for _, row in picks_df.iterrows()
    ]

    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        importance_top = (
            importance_df.sort_values("gain", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )
    else:
        importance_top = []

    live_feature_block = live_panel[feature_columns].astype(float)
    nan_diag = {
        col: int(live_feature_block[col].isna().sum())
        for col in feature_columns
        if live_feature_block[col].isna().any()
    }

    feature_values: dict[str, dict[str, float | None]] = {}
    for _, row in live_panel.iterrows():
        asset = str(row[DEFAULT_ASSET_COLUMN])
        feature_values[asset] = {
            col: (float(row[col]) if pd.notna(row[col]) else None)
            for col in feature_columns
        }

    return PredictionRecord(
        as_of_date=live_date.strftime("%Y-%m-%d"),
        profile_name=profile.name,
        profile_signature=profile.signature(),
        forward_horizon=profile.forward_horizon,
        top_k=profile.top_k,
        universe=list(DEPLOYABLE_UNIVERSE),
        scores=asset_scores,
        top_k_picks=top_k_picks,
        feature_importance=importance_top,
        training_window_start=train_dates[0].strftime("%Y-%m-%d"),
        training_window_end=train_dates[-1].strftime("%Y-%m-%d"),
        training_n_samples=int(train_panel[target_column].notna().sum()),
        training_n_dates=int(len(train_dates)),
        cache_last_bar_date=cache_last_bar_date.strftime("%Y-%m-%d"),
        fetched_fresh_data=bool(fetched_fresh_data),
        nan_feature_diagnostics=nan_diag,
        feature_values_at_live_date=feature_values,
        generated_at_utc=strftime("%Y%m%dT%H%M%SZ", gmtime()),
    )


# ----------------------------------------------------------------------
# Replay mode
# ----------------------------------------------------------------------

def _replay_step_dates(
    panel_dates: pd.DatetimeIndex,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    step: int,
) -> pd.DatetimeIndex:
    """Pick every ``step``-th trading day in [start, end] inclusive."""
    window = panel_dates[(panel_dates >= start_date) & (panel_dates <= end_date)]
    if len(window) == 0:
        return window
    indices = list(range(0, len(window), step))
    return window[indices]


def _run_replay(
    *,
    profile: Profile,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    output_dir: Path,
    fetch_missing: bool,
) -> list[PredictionRecord]:
    frames_full = _load_frames(
        cache_dir=cache_dir,
        end_date=end_date,
        fetch_missing=fetch_missing,
    )
    cache_last_bar = max(pd.to_datetime(f["date"]).max() for f in frames_full.values())

    common = _common_trading_dates(frames_full)
    step_dates = _replay_step_dates(
        common, start_date=start_date, end_date=end_date, step=profile.forward_horizon,
    )
    if len(step_dates) == 0:
        raise RuntimeError(
            f"No trading days in cache between {start_date.date()} and {end_date.date()}."
        )
    print(
        f"Replay: {len(step_dates)} prediction dates from {step_dates[0].date()} "
        f"to {step_dates[-1].date()} (step={profile.forward_horizon} trading days)"
    )

    records: list[PredictionRecord] = []
    for i, as_of in enumerate(step_dates, 1):
        frames_d = _truncate_frames(frames_full, as_of_date=as_of)
        record = _predict_for_as_of(
            profile=profile,
            as_of_date=as_of,
            frames_truncated=frames_d,
            cache_last_bar_date=cache_last_bar,
            fetched_fresh_data=fetch_missing,
        )
        json_path, _ = _save_record(record, output_dir=output_dir)
        picks_str = ", ".join(p["asset"] for p in record.top_k_picks)
        print(
            f"  [{i:>3}/{len(step_dates)}] {as_of.date()} → top-{profile.top_k}: {picks_str} "
            f"(→ {json_path.name})"
        )
        records.append(record)
    return records


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--dry-run", action="store_true")
    mode_grp.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--profile",
        required=True,
        choices=sorted(PROFILES.keys()),
        help=(
            "Named deployment profile. LambdaRank only; Ridge profiles are "
            "excluded — see docs/RIDGE_BASELINE_RESULTS.md for the empirical "
            "justification (Ridge Spearman -0.030 at 5d, -0.055 at 20d, both "
            "below the trivial baseline)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("live", "replay"),
        default="live",
        help="'live' produces one prediction for --as-of-date; 'replay' produces a sequence over [--start-date, --end-date].",
    )
    parser.add_argument(
        "--as-of-date",
        default="today",
        help="As-of date (YYYY-MM-DD or 'today') for live mode. Ignored in replay mode.",
    )
    parser.add_argument("--start-date", default=None, help="Replay window start (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Replay window end (YYYY-MM-DD).")
    parser.add_argument(
        "--fetch-fresh-data",
        action="store_true",
        help=(
            "Allow underlying loader to fetch any MISSING asset cache via "
            "yfinance. Note: does NOT force-refresh already-cached data — "
            "to update existing caches, run scripts/prepare_feature_frame.py."
        ),
    )
    parser.add_argument("--cache-dir", default="data/multi_asset_cache")
    parser.add_argument("--output-dir", default="predictions")
    return parser


def _resolve_as_of(spec: str) -> pd.Timestamp:
    if str(spec).lower() == "today":
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(spec).normalize()


def _print_dry_run(profile: Profile, args: argparse.Namespace, timestamp: str) -> None:
    print(f"[{timestamp}] live_prediction DRY RUN — no data load, no fit, no writes.")
    print(f"  profile:              {profile.name}")
    print(f"  description:          {profile.description}")
    print(f"  signature:            {profile.signature()}")
    print(f"  forward_horizon:      {profile.forward_horizon}")
    print(f"  top_k:                {profile.top_k}")
    print(f"  include_xs_features:  {profile.include_cross_sectional_features}")
    print(f"  include_regime_int:   {profile.include_regime_interactions}")
    print(f"  feature_normalization:{profile.feature_normalization}")
    print(f"  model_name:           {profile.model_name}")
    print(f"  training_window_days: {TRAINING_WINDOW_DAYS}")
    print(f"  mode:                 {args.mode}")
    print(f"  as_of_date:           {args.as_of_date}")
    if args.mode == "replay":
        print(f"  start_date:           {args.start_date}")
        print(f"  end_date:             {args.end_date}")
    print(f"  fetch_fresh_data:     {args.fetch_fresh_data}")
    print(f"  cache_dir:            {args.cache_dir}")
    print(f"  output_dir:           {args.output_dir}")
    print(f"  universe ({len(DEPLOYABLE_UNIVERSE)}):     {list(DEPLOYABLE_UNIVERSE)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    timestamp = strftime("%Y%m%dT%H%M%SZ", gmtime())
    if args.dry_run:
        _print_dry_run(profile, args, timestamp)
        return 0

    if args.mode == "replay":
        if not args.start_date or not args.end_date:
            parser.error("--mode replay requires both --start-date and --end-date.")
        start = pd.to_datetime(args.start_date).normalize()
        end = pd.to_datetime(args.end_date).normalize()
        if start > end:
            parser.error("--start-date must be <= --end-date.")
        _run_replay(
            profile=profile,
            start_date=start,
            end_date=end,
            cache_dir=cache_dir,
            output_dir=output_dir,
            fetch_missing=args.fetch_fresh_data,
        )
        return 0

    # Live / single-date mode.
    as_of = _resolve_as_of(args.as_of_date)
    frames = _load_frames(
        cache_dir=cache_dir,
        end_date=as_of,
        fetch_missing=args.fetch_fresh_data,
    )
    cache_last_bar = max(pd.to_datetime(f["date"]).max() for f in frames.values())
    frames_d = _truncate_frames(frames, as_of_date=as_of)
    common = _common_trading_dates(frames_d)
    if len(common) == 0:
        raise SystemExit(
            f"No trading day available at or before {as_of.date()} in cache. "
            "Run scripts/prepare_feature_frame.py first."
        )
    effective_as_of = common[-1]
    if effective_as_of != as_of:
        print(
            f"Note: requested as-of {as_of.date()} not a trading day in cache; "
            f"using latest available {effective_as_of.date()}."
        )
    record = _predict_for_as_of(
        profile=profile,
        as_of_date=effective_as_of,
        frames_truncated=frames_d,
        cache_last_bar_date=cache_last_bar,
        fetched_fresh_data=args.fetch_fresh_data,
    )
    print(_format_stdout(record))
    json_path, log_path = _save_record(record, output_dir=output_dir)
    print()
    print(f"Saved record: {json_path}")
    print(f"Appended to:  {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
