# Operational Wrapper — Cache Refresh + Live Prediction in One Command

Created 2026-05-11 as the deployment-time entry point sitting above `scripts/run_live_prediction.py`. Closes two operational gaps from the live prediction script's design: hard cache refresh (the live script only fills missing caches, not stale ones) and per-asset orchestration (refreshing 19 caches would otherwise require 19 separate manual invocations).

## Scope

`scripts/run_operational_prediction.py` is pure subprocess orchestration over the two existing CLIs:

- `scripts/prepare_feature_frame.py --execute --force-refresh` — per-asset cache refresh
- `scripts/run_live_prediction.py --execute --profile X --as-of-date Y` — prediction handoff

No new feature engineering, no new model code, no new pipeline functions. The wrapper only adds: per-asset retry/error handling, common-date computation across caches, a confirmation prompt for partial-failure scenarios, and a structured operational log.

## Architecture

Per-asset refresh via subprocess (one Python process per ticker) → summary table → common-date intersection → subprocess handoff to the live prediction script with `--as-of-date` set to the intersection date.

```
                     ┌────────────────────────────────┐
                     │ run_operational_prediction.py  │
                     └───────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │ Phase 1: per-asset refresh (19 tickers)     │
              │   subprocess: prepare_feature_frame.py      │
              │                  --force-refresh            │
              │   retry: 2 retries, backoff [5s, 15s]       │
              │   per-asset AssetRefreshResult              │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │ Phase 2: refresh summary table              │
              │   compute common-date intersection of all   │
              │   19 cache CSVs                             │
              │   abort/confirm if ≥ 2 assets failed        │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │ Phase 3: prediction handoff                 │
              │   subprocess: run_live_prediction.py        │
              │                  --as-of-date <common date> │
              │   stream output to stdout                   │
              │   capture exit code                         │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │ Operational log entry                       │
              │   predictions/operational_log.jsonl         │
              └─────────────────────────────────────────────┘
```

### Per-asset refresh logic

Each ticker is refreshed by a fresh subprocess invocation. Subprocess command (built by `_build_refresh_command`):

- **Standard assets (18 of 19):** `prepare_feature_frame.py --execute --ticker <T> --benchmark SPY --vix ^VIX --cache-dir <dir> --output-csv <scratch> --force-refresh`
- **VIX (`^VIX`):** `prepare_feature_frame.py --execute --ticker SPY --benchmark SPY --vix ^VIX --cache-dir <dir> --output-csv <scratch> --ensure-universe ^VIX --force-refresh`

The wrapper writes prepared CSVs into `data/_operational_refresh_scratch/` (discarded — only the raw cache side effect under `data/multi_asset_cache/` matters). The scratch directory is created once and reused; nothing in the live prediction path reads from it.

#### Why VIX is special-cased

`prepare_single_asset_feature_frame` builds VIX-as-benchmark merges when `--ticker ^VIX` is the primary asset, which produces a 0-row prepared frame (validation fails with "Frame is empty after feature engineering"). The raw `vix_daily.csv` cache *does* get refreshed during that failed run, but the subprocess returns exit code 1 — which the wrapper would otherwise score as a failure. Routing VIX through `--ticker SPY --ensure-universe ^VIX` produces a valid prepared frame (SPY) and uses `ensure_universe_cache` to refresh `^VIX` as a side-asset, yielding exit code 0. SPY is refreshed twice when the universe includes both (once as itself, once as VIX's host call); the extra yfinance call is acceptable overhead.

### Deviation from spec: VIX is always refreshed

The task spec called for VIX to be fetched "only when the profile requires regime interactions (currently `v3_5d_top2`)." Empirically that does not match the codebase: per-asset `vix_*` features (`vix_relative`, `vix_extreme`, `vix_momentum_5d`, `vix_zscore`, `vix_vol_interaction`, etc.) are added inside `prepare_single_asset_feature_frame` whenever `VIXClose` is present in the asset frame, and they appear in every campaign run's `feature_count: 39` for all four LambdaRank profiles. Skipping VIX for non-regime profiles would silently produce NaN columns in those features at the live date. The wrapper always refreshes VIX and documents this in the script's module docstring.

### Retry behavior

`_refresh_one_asset` retries on any non-zero subprocess exit code. Backoffs are `(5.0s, 15.0s)` between attempts → 1 initial + 2 retries = 3 maximum attempts per asset. Transient vs permanent failures cannot be distinguished from a subprocess return code alone; the wrapper retries both. After exhausting retries, the function returns `AssetRefreshResult(status="failed", error_reason=<last 500 chars of stderr>)` and the wrapper continues to the next asset.

### Common-date intersection

After all refreshes, `_compute_common_intersection_date` reads the `Date` column from every cache CSV (including any that failed to refresh — uses the stale data) and returns `max(intersection_of_date_sets)`. This is what gets passed to the live prediction script via `--as-of-date`.

**Note on intersection drift:** even when every cache reports a `latest_bar_date` of today, the intersection can lag by 1–3 trading days. Example from a 2026-05-11 run: 18 equity tickers + VIX all had latest bar `2026-05-11`, but BTC-USD went `... 2026-05-10, 2026-05-12 (missing 05-11)`. yfinance occasionally omits a single day's BTC bar; the intersection accurately reflects this and caps at `2026-05-08` (the most recent date present in *every* set). The live prediction script then handles the intersection date with its own stricter post-`prepare_single_asset_feature_frame` calendar.

### Error handling matrix

| Scenario | Wrapper behavior |
| --- | --- |
| 0 assets failed | Proceed to prediction (or exit cleanly in `--refresh-only`) |
| 1 asset failed | Proceed to prediction silently (under threshold) |
| ≥ 2 assets failed, interactive (default) | Print prompt; on "y" proceed, on anything else abort with `aborted_reason` |
| ≥ 2 assets failed, `--no-interactive` | Abort prediction, exit code 2, log `aborted_reason` |
| No common date across caches (e.g. one cache fully empty) | Abort prediction, exit code 3, log `aborted_reason` |
| Prediction subprocess returned non-zero | Log the exit code, exit with the same code |
| Per-asset subprocess hung indefinitely | Not handled — relies on yfinance's own timeouts; if needed, add a per-subprocess timeout in a future change |

The threshold of 2 is the `ABORT_THRESHOLD` constant. Single-asset failures during a run are common (transient yfinance issues, one weekend BTC quirk, etc.) and don't warrant blocking the whole prediction. Two or more is the trip wire.

## CLI usage

```bash
# Standard operational run — refresh all caches, then predict
uv run python -m scripts.run_operational_prediction \
    --profile 20d_top1 \
    --output-dir predictions/

# Refresh only — useful before running multiple profiles or as a maintenance pass
uv run python -m scripts.run_operational_prediction \
    --refresh-only

# Predict only — skip the cache refresh (e.g. running multiple profiles back to back)
uv run python -m scripts.run_operational_prediction \
    --profile 20d_top1 \
    --skip-refresh \
    --output-dir predictions/

# Non-interactive for automation / cron — aborts on ≥2 refresh failures
uv run python -m scripts.run_operational_prediction \
    --profile 20d_top1 \
    --no-interactive \
    --output-dir predictions/
```

`--profile` is required unless `--refresh-only` is set. `--refresh-only` and `--skip-refresh` are mutually exclusive.

## Operational log

Each wrapper run appends one line to `predictions/operational_log.jsonl`:

```json
{
  "run_timestamp_utc": "20260512T000112Z",
  "profile": "20d_top1",
  "mode": "standard",
  "refresh_results": [
    {"ticker": "SPY", "status": "ok", "latest_bar_date": "2026-05-11", "attempts": 1, "error_reason": null},
    ...
  ],
  "common_intersection_date": "2026-05-08",
  "prediction_invoked": true,
  "prediction_exit_code": 0,
  "aborted_reason": null
}
```

`mode` is one of `standard`, `refresh-only`, `skip-refresh`. `aborted_reason` is `null` on success, populated when the wrapper exited early (≥2 refresh failures + `--no-interactive`, or user declined prompt, or no common date).

## Smoke-test evidence (2026-05-11)

All four required smoke tests pass.

### Test 1 — `--refresh-only` against live yfinance

```
19/19 refreshed successfully
Common date intersection: 2026-05-08
```

All 18 universe tickers reached `2026-05-11`; BTC-USD reached `2026-05-12` (yfinance returned 05-12 but skipped 05-11 — a known yfinance quirk that the intersection logic handles correctly). VIX reached `2026-05-11` via the special-cased ensure-universe path. Every asset refreshed on the first attempt; no retries triggered.

### Test 2 — full end-to-end (`--profile 20d_top1`)

Same refresh outcome as Test 1, followed by a successful `run_live_prediction.py` subprocess at `--as-of-date 2026-05-08`. Live prediction reported `Cache freshness: last bar 2026-05-08`, picked `BTC-USD` (the 20d top-1 high-water mark for the current regime), wrote `predictions/2026-04-16_20d_top1.json` and appended to `predictions/predictions_log.jsonl`. Wrapper logged `prediction_exit_code: 0` and printed `SUCCESS: refresh + prediction completed for profile 20d_top1 at 2026-05-08.`

### Test 3 — `--skip-refresh --profile 20d_top1`

Phases 1+2 skipped. Common-date intersection computed directly from the existing cache (`2026-05-08`). Prediction handoff identical to Test 2. Wrapper logged `mode: "skip-refresh"`, `refresh_results: []`. Useful when running multiple profiles back to back or rerunning a prediction on already-refreshed cache.

### Test 4 — per-asset failure path

Invoked `_refresh_one_asset("BOGUS-NONEXISTENT-TICKER-XYZ", ...)` directly with short backoffs to keep the test fast. Result:
```
status         = failed
attempts       = 3
latest_bar     = None
error_reason   = (truncated yfinance traceback indicating ticker not found)
```
Wrapper did not crash. The `_refresh_universe` loop continues past failures by design — verified by an additional mock-injected test simulating two simulated failures (SPY, QQQ): both got `status="failed"`, the remaining 17 tickers got `status="ok"`, and the resulting failure count (2) correctly meets `ABORT_THRESHOLD`, which is the trip wire for the `--no-interactive` abort path.

## Operational workflow

**When to run.** On the profile's rebalance cadence — every 5 trading days for `v3_5d_top2` / `v1_5d_top2`, every 20 trading days for `20d_top1` / `20d_top2`. Run after the prior bar's close (so the most recent close is included in the panel).

**What to do with the output.** Read the stdout summary for the top-k picks; the JSON record in `predictions/` is the durable audit trail. The wrapper's success/failure line at the end is the operational signal — anything other than `SUCCESS:` means the prediction did not produce a clean record.

**Integration with the forward walk.** `predictions/predictions_log.jsonl` accumulates one record per prediction across time. To build the forward-walk track record, tail that file; pair with realized return data once the forward horizon has elapsed. Realized-PnL comparison code is out of scope for this patch — build it as a separate analysis script when there is enough track-record to evaluate.

**Automation.** Use `--no-interactive` from cron / launchd. The wrapper exits 0 on success, 2 on refresh failure abort, 3 on no-common-date abort, or the live prediction's own exit code on prediction failure. Pair with a separate alert script that watches `predictions/operational_log.jsonl` for `prediction_invoked: false` or `prediction_exit_code != 0`.

## Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| One asset consistently fails refresh | yfinance has delisted/renamed the ticker | Check yfinance for the current symbol; if the asset's role in the universe is preserved by renaming, edit `ASSET_UNIVERSE` here AND in `scripts/run_live_prediction.py::DEPLOYABLE_UNIVERSE`. Both lists must move together. |
| `^VIX` shows as failed even though `vix_daily.csv` updated | The special-case path for VIX broke (e.g. someone added `--ticker ^VIX` invocation) | Re-check `_build_refresh_command`; VIX must use `--ticker SPY --ensure-universe ^VIX` |
| Common intersection date stuck several days behind | One asset's cache is missing a recent date that the others have | Inspect each cache's last 5 dates (see snippet below); if it's a yfinance gap, rerun the wrapper later when yfinance fills it in. If the gap is persistent, that asset's data is structurally compromised — investigate. |
| `--no-interactive` exits 2 unexpectedly | ≥ 2 caches failed. Re-run later (transient yfinance) or investigate the failures listed in the refresh summary | Inspect `predictions/operational_log.jsonl` for the `refresh_results` array and individual `error_reason` fields. |
| Live prediction subprocess exits non-zero | Look at the streamed live-prediction stdout (it printed before the wrapper's final SUCCESS/FAILURE line) | Most common cause: cache too short to support 1008-day training window. Run `--refresh-only` first; if cache is fresh, then the load-frames step is failing — inspect the prepared frame for that asset. |

Quick cache inspection snippet:
```python
import pandas as pd
from pathlib import Path
from scripts.run_operational_prediction import _cache_csv_path, ASSET_UNIVERSE, VIX_TICKER
for t in list(ASSET_UNIVERSE) + [VIX_TICKER]:
    df = pd.read_csv(_cache_csv_path(t, Path("data/multi_asset_cache")), usecols=["Date"])
    print(f"{t:<10} last5: {sorted(df['Date'].tolist())[-5:]}")
```

## Limitations

- **No timeout on per-asset subprocess.** If yfinance hangs indefinitely, the wrapper hangs with it. Acceptable for interactive use; for cron deployments consider wrapping the wrapper in `timeout(1)`.
- **All-or-mostly refresh.** Per-asset retries are sequential. 19 retries × 3 attempts × ~5s = up to ~5 minutes worst case. Typical clean run is ~30–60 seconds.
- **`--skip-refresh` trusts existing cache state.** If the cache is days stale, `--skip-refresh` will happily use it — the live prediction's `cache_last_bar_date` field is the only stale-data signal.
- **Universe is fixed.** Same constraint as the live prediction script: any universe change must be coordinated across both scripts.
- **VIX special-case is implicit.** Logic in `_build_refresh_command` checks `ticker == VIX_TICKER`. If `prepare_feature_frame.py`'s validation behavior changes (e.g. VIX-as-primary becomes valid), the special case becomes dead code but is harmless.
