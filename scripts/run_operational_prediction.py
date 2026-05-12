"""Operational wrapper: hard cache refresh + live prediction in one command.

Closes two operational gaps from the live prediction script:

1. ``scripts/run_live_prediction.py``'s ``--fetch-fresh-data`` only fills
   *missing* asset caches via yfinance; it does not force a hard refresh of
   already-cached data. This wrapper invokes
   ``scripts/prepare_feature_frame.py --force-refresh`` per asset to
   genuinely update each cache to the latest available bar.

2. Per-asset refresh requires 18+1 separate invocations. This wrapper does
   them in sequence with per-asset retry on transient failures, then
   computes the common-date intersection and hands off to the live
   prediction script with ``--as-of-date`` set to that date.

Pure subprocess orchestration over the two existing CLIs. No new feature
engineering, no new model code, no new pipeline functions.

**VIX is always refreshed.** The task spec suggested fetching VIX only for
``v3_5d_top2`` (the regime-interaction profile). Empirically the per-asset
``vix_*`` features (``vix_relative``, ``vix_extreme``, ``vix_momentum_5d``,
etc.) are computed inside ``prepare_single_asset_feature_frame`` for all
profiles whenever ``VIXClose`` is present in the asset frame — they appear
in every campaign run's ``feature_count: 39``. Skipping VIX for non-regime
profiles would cause those features to silently fall to NaN. The wrapper
deviates from the spec on this single point and always refreshes VIX.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import gmtime, strftime
from typing import Sequence

import pandas as pd


# Universe — must match scripts/run_live_prediction.DEPLOYABLE_UNIVERSE.
ASSET_UNIVERSE: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "DBA", "UUP",
    "VNQ", "BTC-USD",
)
VIX_TICKER: str = "^VIX"
BENCHMARK_TICKER: str = "SPY"

# Profiles the wrapper knows how to forward to scripts/run_live_prediction.py.
# Kept in sync with PROFILES in run_live_prediction.py.
KNOWN_PROFILES: tuple[str, ...] = (
    "v3_5d_top2", "20d_top1", "20d_top2", "v1_5d_top2",
)

DEFAULT_CACHE_DIR: Path = Path("data/multi_asset_cache")
DEFAULT_OUTPUT_DIR: Path = Path("predictions")
SCRATCH_DIR_NAME: str = "_operational_refresh_scratch"

RETRY_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 15.0)  # 2 retries (3 total attempts)
ABORT_THRESHOLD: int = 2  # >= this many failures → confirm or abort


def _normalize_for_filename(ticker: str) -> str:
    return ticker.strip().lstrip("^").lower()


def _cache_csv_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_normalize_for_filename(ticker)}_daily.csv"


# ----------------------------------------------------------------------
# Refresh logic
# ----------------------------------------------------------------------

@dataclass
class AssetRefreshResult:
    ticker: str
    status: str  # "ok" | "failed"
    latest_bar_date: str | None
    attempts: int
    error_reason: str | None = None


def _read_latest_bar_date(cache_path: Path) -> str | None:
    """Return the last date in the cache CSV, or None on missing / unparseable."""
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return None
    try:
        with cache_path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 2048))
            chunk = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return None
    last_field = lines[-1].split(",", 1)[0].strip()
    try:
        pd.to_datetime(last_field)
    except (ValueError, TypeError):
        return None
    return last_field


def _build_refresh_command(
    ticker: str,
    *,
    cache_dir: Path,
    scratch_csv: Path,
) -> list[str]:
    """Build the prepare_feature_frame.py subprocess command for one ticker.

    VIX is special-cased: it cannot be a primary ``--ticker`` because
    ``prepare_single_asset_feature_frame`` builds VIX-as-benchmark merges that
    produce an empty frame (validation fails with row_count=0). Its raw cache
    refresh is therefore routed through ``--ensure-universe ^VIX`` with SPY as
    the disposable main ticker. SPY refreshes twice when both SPY and VIX are
    in the universe (once as itself, once as VIX's host call); the redundant
    yfinance call is acceptable overhead.
    """
    if ticker == VIX_TICKER:
        return [
            sys.executable, "-m", "scripts.prepare_feature_frame",
            "--execute",
            "--ticker", "SPY",
            "--benchmark", BENCHMARK_TICKER,
            "--vix", VIX_TICKER,
            "--cache-dir", str(cache_dir),
            "--output-csv", str(scratch_csv),
            "--ensure-universe", VIX_TICKER,
            "--force-refresh",
        ]
    return [
        sys.executable, "-m", "scripts.prepare_feature_frame",
        "--execute",
        "--ticker", ticker,
        "--benchmark", BENCHMARK_TICKER,
        "--vix", VIX_TICKER,
        "--cache-dir", str(cache_dir),
        "--output-csv", str(scratch_csv),
        "--force-refresh",
    ]


def _refresh_one_asset(
    ticker: str,
    *,
    cache_dir: Path,
    scratch_csv: Path,
    backoffs: Sequence[float] = RETRY_BACKOFF_SECONDS,
) -> AssetRefreshResult:
    """Invoke prepare_feature_frame.py --force-refresh for one ticker.

    Retries on any non-zero exit (transient and permanent are not
    distinguishable from the subprocess return code alone). After exhausting
    retries, returns ``status='failed'`` with the last captured stderr.
    """
    scratch_csv.parent.mkdir(parents=True, exist_ok=True)
    last_error: str | None = None
    attempts = 0
    max_attempts = 1 + len(backoffs)
    cmd = _build_refresh_command(ticker, cache_dir=cache_dir, scratch_csv=scratch_csv)
    for attempt_idx in range(max_attempts):
        attempts = attempt_idx + 1
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            latest = _read_latest_bar_date(_cache_csv_path(ticker, cache_dir))
            return AssetRefreshResult(
                ticker=ticker,
                status="ok",
                latest_bar_date=latest,
                attempts=attempts,
            )
        err = (result.stderr or result.stdout or "").strip()
        last_error = err[-500:] if err else f"exit code {result.returncode}"
        if attempt_idx < len(backoffs):
            time.sleep(backoffs[attempt_idx])
    return AssetRefreshResult(
        ticker=ticker,
        status="failed",
        latest_bar_date=_read_latest_bar_date(_cache_csv_path(ticker, cache_dir)),
        attempts=attempts,
        error_reason=last_error,
    )


def _refresh_targets(profile: str | None) -> tuple[str, ...]:
    """The full list of tickers to refresh (always universe + VIX).

    The ``profile`` argument is accepted for forward-compatibility but does
    not gate VIX inclusion — see module docstring for rationale.
    """
    return ASSET_UNIVERSE + (VIX_TICKER,)


def _refresh_universe(
    *,
    targets: Sequence[str],
    cache_dir: Path,
    scratch_dir: Path,
) -> list[AssetRefreshResult]:
    results: list[AssetRefreshResult] = []
    for ticker in targets:
        scratch_csv = scratch_dir / f"{_normalize_for_filename(ticker)}.csv"
        print(f"  Refreshing {ticker:<10} ...", end="", flush=True)
        result = _refresh_one_asset(ticker, cache_dir=cache_dir, scratch_csv=scratch_csv)
        if result.status == "ok":
            print(f" ok    (latest bar {result.latest_bar_date}, attempts={result.attempts})")
        else:
            short = (result.error_reason or "")[:100].replace("\n", " ")
            print(f" FAIL  after {result.attempts} attempts — {short}")
        results.append(result)
    return results


def _print_refresh_summary(results: list[AssetRefreshResult]) -> None:
    print()
    print("Universe Refresh Summary")
    print("========================")
    print(f"{'Asset':<10} | {'Status':<8} | {'Latest Bar':<12} | Attempts")
    print(f"{'-'*10}-|-{'-'*8}-|-{'-'*12}-|---------")
    for r in results:
        latest = r.latest_bar_date or "(none)"
        print(f"{r.ticker:<10} | {r.status.upper():<8} | {latest:<12} | {r.attempts}")
    print(f"{'-'*10}-|-{'-'*8}-|-{'-'*12}-|---------")
    ok = sum(1 for r in results if r.status == "ok")
    print(f"{ok}/{len(results)} refreshed successfully")
    failures = [r for r in results if r.status != "ok"]
    if failures:
        print()
        print("Failures:")
        for r in failures:
            short = (r.error_reason or "")[:240].replace("\n", " ")
            print(f"  {r.ticker:<10}: {short}")


def _compute_common_intersection_date(
    cache_dir: Path,
    *,
    tickers: Sequence[str],
) -> str | None:
    """Latest date present in EVERY ticker's cache CSV."""
    date_sets: list[set[str]] = []
    for t in tickers:
        path = _cache_csv_path(t, cache_dir)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, usecols=["Date"])
        except (ValueError, FileNotFoundError, pd.errors.EmptyDataError):
            return None
        date_sets.append(set(df["Date"].astype(str)))
    if not date_sets:
        return None
    common = set.intersection(*date_sets)
    if not common:
        return None
    return max(common)


# ----------------------------------------------------------------------
# Live-prediction handoff
# ----------------------------------------------------------------------

def _run_live_prediction(
    *,
    profile: str,
    as_of_date: str,
    output_dir: Path,
    cache_dir: Path,
) -> int:
    """Invoke scripts/run_live_prediction.py as a subprocess, streaming output."""
    cmd = [
        sys.executable, "-m", "scripts.run_live_prediction",
        "--execute",
        "--profile", profile,
        "--mode", "live",
        "--as-of-date", as_of_date,
        "--cache-dir", str(cache_dir),
        "--output-dir", str(output_dir),
    ]
    print()
    print(f"Running live prediction: --profile {profile} --as-of-date {as_of_date}")
    print(f"  cmd: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd)
    return int(result.returncode)


# ----------------------------------------------------------------------
# Operational log
# ----------------------------------------------------------------------

@dataclass
class OperationalLogEntry:
    run_timestamp_utc: str
    profile: str | None
    mode: str  # "standard" | "refresh-only" | "skip-refresh"
    refresh_results: list[dict]
    common_intersection_date: str | None
    prediction_invoked: bool
    prediction_exit_code: int | None
    aborted_reason: str | None = None


def _append_operational_log(entry: OperationalLogEntry, *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "operational_log.jsonl"
    with log_path.open("a") as fh:
        fh.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
    return log_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(KNOWN_PROFILES),
        default=None,
        help="Live prediction profile to run after refresh. Required unless --refresh-only.",
    )
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--refresh-only",
        action="store_true",
        help="Phase 1+2 only: refresh caches and print summary. No prediction.",
    )
    mode_grp.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip Phase 1+2: use cache as-is, jump straight to prediction.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help=(
            f"Disable confirmation prompts. With this flag set, the wrapper aborts "
            f"the prediction step (and exits non-zero) if {ABORT_THRESHOLD}+ assets fail to refresh."
        ),
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def _confirm_proceed(n_failures: int) -> bool:
    prompt = (
        f"\nWARNING: {n_failures} assets failed to refresh. "
        f"Continue with prediction using stale data for those assets? [y/N] "
    )
    try:
        response = input(prompt).strip().lower()
    except EOFError:
        return False
    return response == "y"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    scratch_dir = cache_dir.parent / SCRATCH_DIR_NAME
    run_ts = strftime("%Y%m%dT%H%M%SZ", gmtime())

    if not args.refresh_only and not args.profile:
        parser.error("--profile is required unless --refresh-only is set.")

    mode = "refresh-only" if args.refresh_only else ("skip-refresh" if args.skip_refresh else "standard")
    print(f"[{run_ts}] operational_prediction mode={mode} profile={args.profile or '(none)'}")
    print(f"  cache_dir:  {cache_dir}")
    print(f"  output_dir: {output_dir}")
    print()

    refresh_results: list[AssetRefreshResult] = []
    aborted_reason: str | None = None
    prediction_invoked = False
    prediction_exit_code: int | None = None

    # ---- Phase 1 + 2: refresh ----
    if not args.skip_refresh:
        targets = _refresh_targets(args.profile)
        print(f"Refreshing {len(targets)} caches ({len(ASSET_UNIVERSE)} assets + VIX):")
        refresh_results = _refresh_universe(
            targets=targets, cache_dir=cache_dir, scratch_dir=scratch_dir,
        )
        _print_refresh_summary(refresh_results)

    # ---- Phase 3: common date + prediction ----
    targets_for_common = _refresh_targets(args.profile)
    common_date = _compute_common_intersection_date(cache_dir, tickers=targets_for_common)
    print()
    print(f"Common date intersection: {common_date or '(none — at least one cache missing)'}")

    if args.refresh_only:
        log_entry = OperationalLogEntry(
            run_timestamp_utc=run_ts,
            profile=args.profile,
            mode=mode,
            refresh_results=[asdict(r) for r in refresh_results],
            common_intersection_date=common_date,
            prediction_invoked=False,
            prediction_exit_code=None,
        )
        log_path = _append_operational_log(log_entry, output_dir=output_dir)
        print(f"Logged: {log_path}")
        print("Refresh-only mode: skipping prediction.")
        return 0

    n_failures = sum(1 for r in refresh_results if r.status != "ok")
    if n_failures >= ABORT_THRESHOLD:
        if args.no_interactive:
            aborted_reason = (
                f"{n_failures} assets failed to refresh and --no-interactive set; "
                f"aborting prediction."
            )
            print()
            print(f"ABORT: {aborted_reason}")
            log_entry = OperationalLogEntry(
                run_timestamp_utc=run_ts,
                profile=args.profile,
                mode=mode,
                refresh_results=[asdict(r) for r in refresh_results],
                common_intersection_date=common_date,
                prediction_invoked=False,
                prediction_exit_code=None,
                aborted_reason=aborted_reason,
            )
            log_path = _append_operational_log(log_entry, output_dir=output_dir)
            print(f"Logged: {log_path}")
            return 2
        proceed = _confirm_proceed(n_failures)
        if not proceed:
            aborted_reason = "user declined to proceed after refresh failures."
            print(f"ABORT: {aborted_reason}")
            log_entry = OperationalLogEntry(
                run_timestamp_utc=run_ts,
                profile=args.profile,
                mode=mode,
                refresh_results=[asdict(r) for r in refresh_results],
                common_intersection_date=common_date,
                prediction_invoked=False,
                prediction_exit_code=None,
                aborted_reason=aborted_reason,
            )
            log_path = _append_operational_log(log_entry, output_dir=output_dir)
            print(f"Logged: {log_path}")
            return 2

    if common_date is None:
        aborted_reason = "no common date across caches; cannot pick an as-of date."
        print(f"ABORT: {aborted_reason}")
        log_entry = OperationalLogEntry(
            run_timestamp_utc=run_ts,
            profile=args.profile,
            mode=mode,
            refresh_results=[asdict(r) for r in refresh_results],
            common_intersection_date=None,
            prediction_invoked=False,
            prediction_exit_code=None,
            aborted_reason=aborted_reason,
        )
        log_path = _append_operational_log(log_entry, output_dir=output_dir)
        print(f"Logged: {log_path}")
        return 3

    prediction_invoked = True
    prediction_exit_code = _run_live_prediction(
        profile=args.profile,
        as_of_date=common_date,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )

    # ---- Log + final status ----
    log_entry = OperationalLogEntry(
        run_timestamp_utc=run_ts,
        profile=args.profile,
        mode=mode,
        refresh_results=[asdict(r) for r in refresh_results],
        common_intersection_date=common_date,
        prediction_invoked=prediction_invoked,
        prediction_exit_code=prediction_exit_code,
    )
    log_path = _append_operational_log(log_entry, output_dir=output_dir)
    print()
    print(f"Logged: {log_path}")
    if prediction_exit_code == 0:
        print(f"SUCCESS: refresh + prediction completed for profile {args.profile} at {common_date}.")
    else:
        print(f"FAILURE: prediction exited {prediction_exit_code}.")
    return int(prediction_exit_code or 0)


if __name__ == "__main__":
    sys.exit(main())
