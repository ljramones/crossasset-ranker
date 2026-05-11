# Patch: Active-Track Data Rebuild — Market Cache + Prepared Feature Frame Pipeline

Date: 2026-05-10

## Why Path B was chosen

Path A (restore the old `data/`, `results/`, and `/private/tmp/regime_overlay_spy_feature_frame.csv` from the previous laptop) was unavailable — the old artifacts are not recoverable. The only viable option was Path B: rebuild a clean active-track data pipeline from scratch.

The legacy `data/market_data.py` was deliberately **not recreated**. Recreating it would silently revive the frozen model-zoo CLI (`main.py`, `utils/experiment.py`, `audit/integrity_audit.py`), which the first-principles reset explicitly retired. Instead, a new `data/market_cache.py` module was added that the active research track (`scripts/`, `experiments/`) can use exclusively.

## Files created

- `data/__init__.py` — package marker; documents that `data.market_data` is intentionally absent.
- `data/market_cache.py` — `MarketCacheConfig`, `normalize_ticker_for_filename`, `fetch_yfinance_ohlcv`, `write_market_cache`, `load_cached_ohlcv`, `load_or_fetch_ohlcv`, `build_asset_cache_frame`, `ensure_universe_cache`. Writes a JSON metadata sidecar (`<ticker>_daily.meta.json`) on every fresh fetch to make adjusted-close drift auditable.
- `scripts/prepare_feature_frame.py` — standalone CLI with `--dry-run` / `--execute` discipline. Builds prepared feature CSVs without touching `main.py`, `utils/experiment.py`, `audit/integrity_audit.py`, or `data.market_data`. Adds `benchmark_return_1d` (which `build_feature_set` does not produce on its own) so downstream `evaluation/audit_artifacts.py` and matched-null tooling find the column they expect.
- `tests/test_market_cache.py` — 8 tests, fully synthetic (yfinance is monkeypatched).
- `tests/test_prepare_feature_frame.py` — 7 tests, including a static check that the script does not import `data.market_data`, `utils.experiment`, `audit.integrity_audit`, or `main`.
- `docs/PATCH_DATA_REBUILD_ACTIVE_TRACK_REPORT.md` — this file.

## Tests run and results

```text
uv run python -m pytest tests/test_market_cache.py tests/test_prepare_feature_frame.py -x -q
... 15 passed in 15.17s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 127 passed in 3.21s
```

`tests/test_integrity_audit_matched_nulls.py` is the one collection error in the suite — it imports `audit.integrity_audit`, which still tries `from data.market_data import load_market_data`. That failure is **pre-existing** (the missing legacy module is exactly what the reset plan documents) and the spec forbids fixing it by recreating `data/market_data.py`. No other tests are affected.

## Dry-run command and result

```bash
uv run python -m scripts.prepare_feature_frame \
  --dry-run \
  --ticker SPY \
  --benchmark SPY \
  --vix '^VIX' \
  --start-date 2010-01-01 \
  --output-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --include-drawdown-labels \
  --horizons 10 20 \
  --thresholds -0.02 -0.03 -0.05
```

Result: dry-run printed the resolved configuration, fetched no data, wrote no files. No `data/multi_asset_cache/` write occurred during the dry-run.

## Execute command and result

```bash
uv run python -m scripts.prepare_feature_frame \
  --execute \
  --ticker SPY \
  --benchmark SPY \
  --vix '^VIX' \
  --start-date 2010-01-01 \
  --output-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --include-drawdown-labels \
  --horizons 10 20 \
  --thresholds -0.02 -0.03 -0.05
```

Result: yfinance fetch succeeded for SPY and ^VIX. Caches written to `data/multi_asset_cache/`. Prepared CSV written to `/private/tmp/regime_overlay_spy_feature_frame.csv`. Validation summary `is_valid: true`, no missing required columns.

## Prepared CSV

- Path: `/private/tmp/regime_overlay_spy_feature_frame.csv`
- Rows: **3315**
- Columns: **52** (raw OHLCV + benchmark/VIX closes + base/advanced/VIX features + drawdown helpers + 6 drawdown event labels)
- Date range: **2010-06-11 → 2026-05-07**
- 20d/3pct positive rate: **0.2915**

Note on the start date: the raw SPY cache covers 2010-01-04 → 2026-05-08, but `build_feature_set(advanced_features=True)` drops rows where its rolling-window features are still warming up (longest window is `max(60, 20*3) = 60` trading days, plus a few more days for second-order derivatives and autocorrelation). That truncation moves the prepared frame's first row from 2010-01-04 to 2010-06-11. The end date is one trading day shy of the cache because `target_horizon=1` shifts `forward_simple_return_1d` by one bar and the resulting NaN is dropped.

## Cache files created

```text
data/multi_asset_cache/
  spy_daily.csv       4112 rows  2010-01-04 -> 2026-05-08
  qqq_daily.csv       4112 rows  2010-01-04 -> 2026-05-08
  iwm_daily.csv       4112 rows  2010-01-04 -> 2026-05-08
  tlt_daily.csv       4112 rows  2010-01-04 -> 2026-05-08
  gld_daily.csv       4112 rows  2010-01-04 -> 2026-05-08
  vix_daily.csv       4112 rows  2010-01-04 -> 2026-05-08
  btc-usd_daily.csv   4254 rows  2014-09-17 -> 2026-05-10
```

Each CSV has a sibling `*_daily.meta.json` recording fetch date (UTC), yfinance version, requested span, observed span, row count, and a drift warning. SPY's metadata is reproduced below for reference:

```json
{
  "ticker": "SPY",
  "fetch_date_utc": "2026-05-10T21:57:37Z",
  "configured_start_date": "2010-01-01",
  "first_date": "2010-01-04",
  "last_date": "2026-05-08",
  "row_count": 4112,
  "source": "yfinance",
  "yfinance_version": "1.3.0"
}
```

The SPY raw cache span (2010-01-04 → 2026-05-08) is consistent with the previously documented span in `champions/current_champion_manifest.yaml` (2010-01-04 → 2026-05-06); the extra two trading days reflect the time elapsed since the manifest was frozen on 2026-05-08.

## Universe cache status

All six requested tickers — SPY, QQQ, IWM, TLT, GLD, BTC-USD — plus the SPY benchmark and ^VIX are cached and ready for the cross-asset ranking feasibility prototype. BTC-USD's start date of 2014-09-17 is the natural inception of the yfinance series.

## yfinance drift warnings

This is a fresh fetch on the new laptop. The following drift risks apply to every downstream artifact built on this cache:

- yfinance retroactively revises Adj Close after corporate actions, splits, and data-source corrections. **Run-to-run reproducibility of historical metrics is not guaranteed** even with seed=42 fixed.
- The previously documented SPY cache (per the champion manifest) ended 2026-05-06; the rebuild ends 2026-05-08. Even on the overlapping span, individual Adj Close values may differ from the prior cache. There is no way to verify magnitude of drift because the prior CSV was not recovered.
- The `*_daily.meta.json` sidecars exist precisely so future reruns can be diffed against the current snapshot to detect drift.

## Confirmation: legacy paths untouched

- `data/market_data.py` was **not** recreated (`ls data/market_data.py` would still fail).
- `main.py`, `utils/experiment.py`, `audit/integrity_audit.py` were not modified.
- No experiments were run, no models were trained, no Optuna study was started, no economic overlay was evaluated, and `prepare_experiment(...)` was not called.

## Active-script smoke checks

```text
uv run python -m scripts.run_drawdown_risk_classifier_experiment --dry-run
... Drawdown-risk classifier runner dry run only. No data will be loaded. No outputs will be written.

uv run python -m scripts.run_drawdown_risk_calibration_experiment --dry-run
... Drawdown-risk calibration runner dry run only. No data will be loaded. No outputs will be written.
```

The active classifier and calibration runners still dry-run cleanly with the new data layer underneath. They are now unblocked for `--execute` runs against `/private/tmp/regime_overlay_spy_feature_frame.csv`.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | `data/market_cache.py` exists | ✓ |
| 2 | `scripts/prepare_feature_frame.py` exists | ✓ |
| 3 | `data/market_data.py` is not recreated | ✓ |
| 4 | Tests exist and pass | ✓ (15/15 new, 127/127 collectible suite) |
| 5 | SPY prepared frame at `/private/tmp/regime_overlay_spy_feature_frame.csv` | ✓ (3315 rows) |
| 6 | Drawdown labels present in prepared frame | ✓ (6 event labels) |
| 7 | Universe cache for SPY/QQQ/IWM/TLT/GLD/BTC-USD | ✓ |
| 8 | Active classifier/calibration scripts dry-run | ✓ |
| 9 | No legacy experiment path was run | ✓ |
| 10 | No model training was run | ✓ |
| 11 | Documentation report created | ✓ (this file) |

## Recommended next task

**Cross-Asset Ranking Feasibility Prototype.**

Universe is already cached. The next prototype should:

- score forward 20-day risk-adjusted returns across SPY, QQQ, IWM, TLT, GLD, BTC-USD
- use simple models (regularized linear, HistGradientBoosting; LightGBM is already a dependency)
- evaluate top-1 / top-2 allocation policies with walk-forward splits
- benchmark against equal-weight, vol-targeted equal-weight, SPY buy-and-hold
- include matched-turnover random-selection nulls

Stop spending cycles on SPY-only target sensitivity until the cross-asset prototype either produces a real signal or definitively rules out this framework on this universe.
