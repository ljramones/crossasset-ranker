# Troubleshooting

Symptom → Diagnosis → Fix. Each entry is self-contained; you should not need to read the rest of the document to apply one.

When you encounter a new issue and resolve it, add an entry here in the same format. Future-you will thank you.

---

## 1. Cache common-intersection date older than expected

**Symptom.** Wrapper output shows `Common date intersection: <DATE>` several days behind today, even after a clean `19/19 refreshed successfully` summary.

**Diagnosis.** The common-date intersection is the latest date present in *every* one of the 19 cache CSVs. Two common causes:

- **BTC weekend gap.** BTC trades 24/7 but yfinance occasionally drops a single weekday (notably the day after a long weekend or holiday). Example seen on 2026-05-11: 18 equity tickers + VIX all had latest bar `2026-05-11`, but BTC-USD's cache went `... 2026-05-10, 2026-05-12` (missing 05-11). The intersection correctly caps at the last date all 19 caches share.
- **Single-asset refresh failure.** One asset's `--force-refresh` subprocess failed (network blip, rate limit) and the wrapper proceeded with stale cache for that asset. Its `latest_bar_date` field in `operational_log.jsonl` will lag the others.

**Fix.**

1. Check the per-asset status in the most recent `operational_log.jsonl` entry:
   ```bash
   tail -1 predictions/operational_log.jsonl | python -c "import json, sys; r=json.loads(sys.stdin.read()); [print(f'{x[\"ticker\"]:<10} {x[\"status\"]:<6} {x[\"latest_bar_date\"]}') for x in r['refresh_results']]"
   ```
2. If one asset is stale, retry it manually:
   ```bash
   uv run python -m scripts.prepare_feature_frame --execute --ticker IWM --benchmark SPY --vix ^VIX --cache-dir data/multi_asset_cache --output-csv /tmp/scratch.csv --force-refresh
   ```
3. If it's a BTC weekend gap, just rerun the wrapper later in the day or the next trading day — yfinance often backfills.
4. If the gap is *persistent* across multiple days for the same asset, that asset's data feed is structurally compromised. Stop trading the strategy and investigate before next rebalance.

---

## 2. `forkpty: Resource temporarily unavailable` (or "Resource temporarily unavailable" on subprocess spawn)

**Symptom.** Cannot open new terminal sessions; `uv run` fails to spawn; subprocess calls error immediately. Sometimes accompanied by repeated `pthread_create` or `Resource temporarily unavailable` errors.

**Diagnosis.** Per-user process limit (`ulimit -u`) reached. Most often caused by orphaned `python` / `claude` / `node` subprocesses from a previously killed long-running task. Each orphan consumes a slot in the process table.

**Fix.**

1. Confirm the limit is the issue:
   ```bash
   ps -u $USER | wc -l   # process count
   ulimit -u             # configured limit
   ```
   If the first number is within ~50 of the second, you're at or near the cap.
2. List orphaned processes (anything you don't recognize that's owned by you):
   ```bash
   ps -ef | grep -E "python|claude|node" | grep -v grep | head -30
   ```
3. Kill the orphans:
   ```bash
   kill -9 <PID_1> <PID_2> ...
   ```
   For a clean sweep of all your python processes (only do this if you can identify which ones are safe to kill):
   ```bash
   pkill -9 -u $USER -f "python -m scripts"
   ```
4. If you can't identify orphans cleanly, log out and back in (terminates all user processes), or reboot.

---

## 3. Process appears hung during a long-running script

**Symptom.** A wrapper or live prediction run has produced no stdout for several minutes. Unclear whether it's making progress or stuck.

**Diagnosis.** Could be legitimately compute-bound (LightGBM training on ~17,800 samples × 33 features is ~10–30 seconds; the wrapper's 19 yfinance calls take 30–90 seconds total). Could also be genuinely hung (rare — most often yfinance not respecting its own timeout). Characterize before killing.

**Fix.**

1. Find the PID:
   ```bash
   ps -ef | grep -E "run_operational_prediction|run_live_prediction|prepare_feature_frame" | grep -v grep
   ```
2. Characterize resource usage:
   ```bash
   ps -p <PID> -o %cpu,%mem,etime,state,command
   ```
   Interpretation:
   - **CPU ~100% on one core**: compute-bound (LightGBM fit or normalization). Let it run.
   - **CPU 0%, state `S` (sleeping)**: blocked on I/O. Usually yfinance — wait 60 more seconds, then kill.
   - **State `D` (uninterruptible sleep)**: blocked on disk I/O. Kernel-level wait. Don't kill — let the kernel resolve.
3. For a Python stack trace without disrupting the process:
   ```bash
   pip install py-spy --user   # one-time, if not installed
   sudo py-spy dump --pid <PID>
   ```
   The bottom of the stack is the active call. `yfinance.download(...)` or `urllib3` → network hang. `LGBMRanker.fit(...)` → compute, leave alone. `pandas.DataFrame.groupby(...)` → compute, leave alone.
4. If after 2–3 minutes of no progress and no CPU, kill the process:
   ```bash
   kill -9 <PID>
   ```
   Then re-run; transient yfinance hangs almost always clear on retry.

---

## 4. yfinance returns NaN or fails to fetch

**Symptom.** `prepare_feature_frame.py` exits non-zero for one or more assets, or the wrapper's refresh summary shows `FAIL`. Stderr typically contains a yfinance traceback (rate limit, "No data found, symbol may be delisted", connection reset).

**Diagnosis.** Most common: yfinance rate-limiting (too many recent requests from your IP), transient network blip, or scheduled market closure for the requested date range. Less common: ticker has been delisted or renamed.

**Fix.**

1. **Wait and retry.** The wrapper auto-retries with `[5s, 15s]` backoff. If all 3 attempts fail, wait 5–10 minutes (rate limits usually clear quickly) and rerun:
   ```bash
   uv run python -m scripts.run_operational_prediction --profile <profile> --output-dir predictions/
   ```
2. **Verify the ticker.** If a specific asset fails persistently, check Yahoo Finance's website directly — does the ticker exist? Has it changed? (E.g., a fund could be merged or renamed.) Run:
   ```bash
   uv run python -c "import yfinance as yf; print(yf.download('IWM', start='2026-04-01', end='2026-05-11').tail())"
   ```
3. **Ticker renamed.** If the asset's role is preserved by the rename, edit `ASSET_UNIVERSE` in `scripts/run_operational_prediction.py` AND `DEPLOYABLE_UNIVERSE` in `scripts/run_live_prediction.py` — they must move together. Then re-run. Note that this is a cohort break: the `profile.signature()` hash will change, and prior predictions in `predictions_log.jsonl` are tagged with the old signature. Document the change in a new patch file `docs/PATCH_UNIVERSE_RENAME_<TICKER>.md`.

---

## 5. Prediction script reports feature NaN at the live date

**Symptom.** stdout shows `NaN feature diagnostics: {'<feature>': <count>, ...}` instead of the usual `clean` line. Sometimes one feature, sometimes many.

**Diagnosis.** A feature couldn't be computed for the live date. Common causes:

- **Insufficient historical data.** Features like `realized_vol_20`, `xs_rank_vol_20d`, or `vix_zscore_252d` require N prior trading days. If the live date is too close to the cache's start, the rolling window can't fill.
- **Data gap in the cache.** One of the 18 assets is missing recent bars. The intersection-joined panel will still produce the date row, but the affected asset's features for that row will be NaN, propagating to the `xs_rank_*` features (which depend on the full universe).
- **First few days of a newly listed asset.** Rare. If the universe were extended with a new asset that has only 30 days of history, its rolling features would be NaN for the first 60+ days.

**Fix.**

1. Look at the JSON record's `feature_values_at_live_date` to see which (asset, feature) cells are null:
   ```bash
   uv run python -c "import json, sys; d=json.load(open(sys.argv[1])); [print(f'{a:<10} {f}: NaN') for a, fv in d['feature_values_at_live_date'].items() for f, v in fv.items() if v is None]" predictions/<latest>.json
   ```
2. If a specific asset is the source, inspect its cache:
   ```bash
   uv run python -c "import pandas as pd; df = pd.read_csv('data/multi_asset_cache/<ticker>_daily.csv'); print(df.tail(30))"
   ```
   Look for missing dates or NaN OHLCV rows in the recent past.
3. If the cache is short or has gaps, force-refresh:
   ```bash
   uv run python -m scripts.prepare_feature_frame --execute --ticker <T> --benchmark SPY --vix ^VIX --cache-dir data/multi_asset_cache --output-csv /tmp/scratch.csv --force-refresh
   ```
4. If after refresh the NaN persists, it's a data-quality issue. Do not trade on a prediction with NaN features for the picked asset(s). Either wait a day and rerun, or skip the rebalance and log the skip in `trade_log.csv` with a `notes` entry explaining why.

---

## 6. Operational log shows a profile signature mismatch from prior runs

**Symptom.** Comparing `profile_signature` field across `predictions_log.jsonl` entries for the same `profile_name`, the hash differs. Example: `20d_top1` records from May show signature `172c54297c8c`, but a recent run shows a different 12-char hash.

**Diagnosis.** The `profile.signature()` hash is computed over the profile's deployment-relevant fields plus the universe and training-window length. Any change to those values — including a code edit to `scripts/run_live_prediction.py::PROFILES` or `DEPLOYABLE_UNIVERSE` — produces a new signature. The mismatch is a *cohort break*, not necessarily a bug.

**Fix.**

1. Verify which signature is current:
   ```bash
   uv run python -c "from scripts.run_live_prediction import PROFILES; [print(f'{p.name}: {p.signature()}') for p in PROFILES.values()]"
   ```
2. Check git history for changes to the profile config:
   ```bash
   git log --oneline -10 scripts/run_live_prediction.py
   ```
3. **If the change was intentional** (e.g. you renamed an asset or adjusted top-k for some reason): document the cohort break in a patch file, and treat prior predictions as belonging to a different strategy. Don't aggregate cross-signature backtests as if they were one strategy.
4. **If the change was unintentional**: revert the profile config, rerun the prediction. The signature should match prior runs again.

---

## 7. Live mode pick differs from historical replay for the same date

**Symptom.** Running `--mode live --as-of-date 2026-04-16 --profile 20d_top1` picks ETF A. Running `--mode replay --start-date 2026-04-16 --end-date 2026-04-16 --profile 20d_top1` against the same cache picks ETF B.

**Diagnosis.** Most likely: cache state changed between the two runs. yfinance occasionally backfills historical bars or applies corporate action adjustments to prior bars; if those happened between your live run and your replay run, the training window's features changed slightly and the model produced different scores. This is **expected behavior**, not a bug.

Less likely: a code change between the two runs. If you edited the live prediction pipeline between runs, the cohort-break signature should differ (see entry #6).

**Fix.**

1. Compare the two JSON records' `cache_last_bar_date` and `training_window_start` / `training_window_end` fields:
   ```bash
   diff <(uv run python -c "import json; d=json.load(open('predictions/2026-04-16_20d_top1.json')); print({k: d[k] for k in ['cache_last_bar_date','training_window_start','training_window_end','profile_signature']})") <(uv run python -c "...")
   ```
2. If the cache windows differ, this is a yfinance-update artifact. The live-run record is the authoritative one (it was made in real-time on the cache state at that moment).
3. If the cache windows match but picks differ, signatures should also differ. If signatures match AND cache windows match AND picks differ, that's a bug — investigate via a small reproduction script and consult [PATCH_LIVE_PREDICTION_SCRIPT.md](PATCH_LIVE_PREDICTION_SCRIPT.md) for the validated pipeline behavior.

---

## 8. Trade log out of sync with predictions log

**Symptom.** `predictions/predictions_log.jsonl` shows a rebalance was predicted on date X (e.g. picked QQQ), but `predictions/trade_log.csv` has no matching trade. Or the trade log shows a position you don't recognize.

**Diagnosis.** Manual error in the workflow. Almost always one of:

- You ran the prediction but didn't place the trade (forgot, or got distracted).
- You placed the trade but didn't log it.
- You manually overrode the model's pick (the trade log shows what you actually held, not what was recommended).
- Two profiles running in parallel and the trade log wasn't tagged with which one.

**Fix.**

1. Reconcile manually. Pull the broker's executed trades for the affected window, the predictions log, and the trade log. Determine what actually happened.
2. Add a dated discrepancy note to the trade log:
   ```
   2026-05-11,note,,, ,"Discrepancy: predicted 2026-04-16 (QQQ top-1) but no trade placed; remained in prior SHY position. Resumed normal procedure 2026-05-11."
   ```
3. Tighten the rebalance procedure. Add a checkbox-style verification step at the end of step 5 in [OPERATIONAL_RUNBOOK.md](OPERATIONAL_RUNBOOK.md): *"trade log entry visible at `tail predictions/trade_log.csv`"*.
4. If running multiple profiles, add a `profile` column to `trade_log.csv` so reconciliation is unambiguous going forward.
5. Once-off discrepancies are tolerable. Repeated ones mean the workflow needs structural changes — consider a one-line shell alias that runs the wrapper and prompts you to confirm the trade-log entry before terminating.

---

## Adding new entries

When you hit something not covered here, after you fix it:

1. Add a new section in the same numbered Symptom / Diagnosis / Fix format.
2. Include real commands you ran, not pseudo-code.
3. Date the entry in the body if useful (e.g. "Example seen on YYYY-MM-DD" — like entry #1's BTC weekend gap).
4. Update the table-of-contents if this doc grows past ~10 entries.

The point of this document is institutional memory for a single-operator deployment. Every entry that saves five minutes the second time it happens has paid for itself.
