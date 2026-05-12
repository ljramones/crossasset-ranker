# Operational Runbook

Day-to-day operating procedure for the cross-asset ranking system. Assumes you've already read [GETTING_STARTED.md](GETTING_STARTED.md) and have made at least one successful test prediction.

## Operational rhythm by profile

The rebalance interval matches the profile's forward horizon: training, scoring, and trading all rest on the same horizon.

| Profile | Rebalance every | Calendar approximation | Yearly rebalances |
| --- | --- | --- | ---: |
| `v3_5d_top2` | 5 trading days | ~1 week (no holidays) | ~50 |
| `v1_5d_top2` | 5 trading days | ~1 week | ~50 |
| `20d_top1` | 20 trading days | ~1 calendar month (typical) | ~13 |
| `20d_top2` | 20 trading days | ~1 calendar month | ~13 |

Holidays, half-days, and the occasional yfinance gap can shift calendar timing by 1–2 days. The trading-day calendar is the source of truth — count trading days from the last prediction date, not calendar days.

If running multiple profiles in parallel (e.g. a small allocation to `v3_5d_top2` alongside a larger one to `20d_top1`), each profile has its own rebalance cadence and its own trade log lifetime. Don't average the two cadences into a single weekly chore.

## Standard rebalance procedure

The five steps below are the procedure for every rebalance. Don't skip any.

### 1. Determine whether today is a rebalance day

Open the last entry of `predictions/predictions_log.jsonl` (or the last `predictions/<date>_<profile>.json` for that profile) and read `as_of_date`. Count trading days from there:

```bash
tail predictions/predictions_log.jsonl | tail -1 | python -c "import json, sys, pandas as pd; r=json.loads(sys.stdin.read()); last=pd.to_datetime(r['as_of_date']); horizon=r['forward_horizon']; nxt=(last + pd.tseries.offsets.BDay(horizon)).date(); today=pd.Timestamp.today().normalize().date(); print(f'last: {last.date()}  horizon: {horizon}  next rebalance (business-day approx): {nxt}  today: {today}  → {\"REBALANCE\" if today >= nxt else \"hold\"}')"
```

`BDay` is a business-day offset, not a trading-day offset — it doesn't know US-market holidays. Treat the printed "next rebalance" as approximate; if today is within 1–2 days of the printed date and the market is open, run the prediction and let the wrapper pick the latest common trading day.

### 2. Refresh caches and run the prediction

One command:

```bash
uv run python -m scripts.run_operational_prediction \
    --profile 20d_top1 \
    --output-dir predictions/
```

Substitute the profile you're running. The wrapper refreshes all 19 caches (18 assets + VIX) via yfinance, then hands off to `run_live_prediction.py` with `--as-of-date` set to the common-date intersection.

Expected wall time: 30–90 seconds for a clean run, up to several minutes if any cache hits a retry path. The wrapper prints per-asset refresh progress and a summary table before the prediction starts.

If running unattended (cron, launchd), add `--no-interactive` so the wrapper aborts instead of prompting on partial refresh failures:

```bash
uv run python -m scripts.run_operational_prediction \
    --profile 20d_top1 \
    --no-interactive \
    --output-dir predictions/
```

### 3. Review the output

Read the stdout block in this order:

1. **Universe refresh summary** — confirm `19/19 refreshed successfully` or investigate any failures
2. **Common date intersection** — confirm the date is recent (within 1–3 trading days of today; weekend BTC quirks are normal)
3. **Top-k pick(s)** — the asset(s) to hold this rebalance period
4. **Full ranking** — scan for unusual concentration or sign patterns; flagged anomalies are not actionable by themselves but are worth noting in your trade-log `notes` column
5. **Top features (LambdaRank gain importance)** — should be dominated by `xs_rank_vol_20d`, `realized_vol_20`, `xs_rank_ret_60d`, `relative_vol_ratio`, etc. (see [TECHNICAL_DESCRIPTION.md](TECHNICAL_DESCRIPTION.md) for the expected pattern)
6. **NaN feature diagnostics** — should say "clean (no NaN feature values at live date)". Any NaN here means a feature couldn't be computed for the live date — investigate before trading
7. **Cache freshness** — last bar date

If any of (1), (2), (6), or (7) look wrong, stop and consult [TROUBLESHOOTING.md](TROUBLESHOOTING.md). Do not trade on a prediction whose inputs are suspect.

### 4. Execute the trade

Manual at your broker. For top-k > 1, weight equally across the picks. The script's `weight` field in the JSON record is the intended allocation (`1.0` for top-1, `0.5` each for top-2).

Two practical points:

- **Cost discipline.** The campaign's IR was modeled with 2 bps round-trip costs. Real-world slippage on liquid ETFs is comfortably under that; BTC slippage at retail-scale is higher (~10–30 bps depending on exchange). If trading via a high-cost route, the modest backtest edge erodes quickly.
- **Order type.** Market-on-open or VWAP for the rebalance day is fine. Limit orders at stale prices defeat the "act on today's signal" purpose.

### 5. Log the trade and confirm the operational log

Append to `predictions/trade_log.csv`:

```
date,action,ticker,price,position_size,notes
2026-05-11,buy,BTC-USD,80737.71,0.05,20d_top1 rebalance; sold prior SHY position
```

Suggested columns: `date`, `action` (`buy` / `sell`), `ticker`, executed `price`, `position_size` (as a fraction of total deployed capital), `notes`. If you change broker, add a `broker` column. Create the file with header on first use.

Confirm the wrapper logged the run:

```bash
tail -1 predictions/operational_log.jsonl
```

The line should contain `"prediction_invoked": true`, `"prediction_exit_code": 0`, `"aborted_reason": null`, and refresh-result entries for all 19 tickers.

## Calendar planning

Approximate next-rebalance dates for budgeting attention:

| Profile | If last rebalance was a Friday close | If last rebalance was mid-month |
| --- | --- | --- |
| 5d profile | The following Friday (5 trading days = 1 calendar week, assuming no holiday) | 5 trading days forward |
| 20d profile | ~4 calendar weeks out, falling on the same weekday | ~1 calendar month forward |

US market holidays that distort trading-day counts: New Year's Day, MLK Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas. In each affected month, the calendar approximation gains 1 extra calendar day per holiday.

For automation:

```bash
# Print the next 5 rebalance dates for a given profile based on the predictions log
uv run python -c "
import json, pandas as pd
from pathlib import Path
profile = '20d_top1'  # or your profile
horizon = 20 if profile.startswith('20d') else 5
lines = Path('predictions/predictions_log.jsonl').read_text().splitlines()
last_for_profile = max((json.loads(l) for l in lines if json.loads(l).get('profile_name') == profile), key=lambda r: r['as_of_date'])
last = pd.to_datetime(last_for_profile['as_of_date'])
schedule = [(last + pd.tseries.offsets.BDay(horizon * i)).date() for i in range(1, 6)]
print(f'Next 5 rebalance dates for {profile}: {schedule}')
"
```

Note that this prints *business-day* offsets, not holiday-corrected trading days. Always reconcile with the actual cache's most-recent common trading date when you run the prediction.

## Monthly review (non-rebalance months)

For 5d profiles, every rebalance is also a weekly check. For 20d profiles, do a separate monthly sweep even if no rebalance falls in the month:

1. **Cache health.** Run `--refresh-only` and confirm `19/19 refreshed successfully`:
   ```bash
   uv run python -m scripts.run_operational_prediction --refresh-only
   ```
2. **Predictions-log sanity.** Verify the predictions log isn't growing unexpectedly (each run adds ~30–40 KB; ~50 runs/year for 20d profiles is small).
3. **Brokerage vs predictions reconciliation.** Compare your broker holdings to the most recent prediction record's `top_k_picks`. They should match. Discrepancies mean either a missed rebalance or a manual override — investigate before the next rebalance.
4. **`operational_log.jsonl` audit.** Skim the last month of entries. Any `aborted_reason` non-null entries? Any `prediction_exit_code != 0`?

## Quarterly review

Every 3 months, do a deeper check.

1. **Realized return per held position.** For each rebalance period in the quarter, compute the actual return of the held asset(s) over the holding window. Compare against:
   - The benchmark equal-weight portfolio for the same window (1/18 in each asset)
   - The model's predicted score for the asset (sanity check: high-scored picks should outperform low-scored picks on average)
2. **Sharpe vs historical estimate.** Three months ≈ 60–65 trading days. With 20d_top1 that's ~3 rebalances — too few for a meaningful Sharpe estimate. With v3_5d_top2 that's ~12–13 rebalances, which is the minimum for a noisy Sharpe estimate. Realized Sharpe over the quarter should land roughly in [0, 2] for any of the LambdaRank profiles; *anything outside that range warrants attention*. Historical backtest Sharpe estimates for context: v3_5d_top2 = 1.18, 20d_top1 ≈ 0.76 (derived from per-fold metrics).
3. **Decision criteria for continuing vs winding down.**
   - **Continue** if realized IR vs equal-weight is non-negative across rebalance periods in the quarter, OR if a single sharp drawdown explains the underperformance and the asset selection itself was reasonable in hindsight (the campaign showed the system avoids catastrophes when it picks SHY/UUP defensively).
   - **Pause and review** if realized IR is strongly negative across 6+ consecutive rebalance periods. The empirical universe ceiling means the strategy's edge is small to begin with; sustained underperformance suggests either regime change or implementation drift.
   - **Wind down** is the appropriate action if both quarters are bad AND the per-asset feature distributions have shifted materially from the training period (a quant-research call, not a snap decision).

## Annual review

After 12 months of live data:

1. **Strategy meeting expectations.** Compute realized Sharpe, IR, max drawdown, turnover, and cost drag for the full year. Compare to backtest expectations from [RIDGE_BASELINE_RESULTS.md](RIDGE_BASELINE_RESULTS.md) Section 1 (the 2×2 grid). The realized values should be in the same ballpark as the backtest, but expect 1× annual returns to be noisy — even a 12-month sample is well below the statistical-power threshold for the small per-trade edge being measured.
2. **Tax planning.** 20d profiles produce ~13 rebalances/year, 5d profiles ~50. Most rebalances generate short-term capital gains/losses. Plan in advance: tax-advantaged accounts make the 5d profile materially more viable; in taxable accounts, prefer the 20d profile to reduce realization frequency.
3. **Retraining checkpoint.** The live script retrains on the most recent 1008 trading days every time you run it — no scheduled retraining is needed. But once per year is a good time to confirm the profile's `feature_importance` rank order is still consistent with the campaign's top features (`xs_rank_vol_20d`, `realized_vol_20`, `relative_vol_ratio`, etc.). A material shift in the importance pattern means the model has adapted to a new regime — fine in principle, but worth noting.

## What NOT to do

These are anti-patterns. Each one has a specific failure mode documented in the campaign chain or implied by the empirical universe ceiling.

- **Don't overtrade.** Every additional rebalance erodes the small edge with transaction costs and slippage. The profile's rebalance interval is the design — running predictions in between rebalances and acting on them defeats the point.
- **Don't override the model based on intuition.** "It's picking BTC; the news today says crypto will fall" is exactly the kind of override the campaign showed has no positive expectancy. If you're going to discretionarily override, do so consistently and track *that* strategy's performance separately.
- **Don't scale capital from one good month.** A single month of outperformance against a benchmark with Sharpe ~1 is well within noise for a small-edge strategy. Position sizing should be set once based on backtest economics and risk tolerance; don't ratchet it up because the model just had a hot streak.
- **Don't change features or models mid-deployment.** Any change to a profile's feature set, normalization, or model invalidates comparability with prior predictions in the same `predictions_log.jsonl`. If you must change something, mark the cohort break clearly: rotate to a new `output-dir`, retire the old profile name, define a new one. The `profile.signature()` field in each JSON record is the cohort marker.
- **Don't run with `--skip-refresh` then ignore stale-cache warnings.** The live prediction script's `cache_last_bar_date` field is the only signal you have that the data behind the prediction is fresh. If it's days old, the prediction is days-old too.
- **Don't ignore the operational log.** It's append-only and small. Read it on every rebalance and skim it during monthly reviews. Silent failures are the worst kind.
- **Don't add new profiles without a campaign-level experiment.** The four current profiles are the deployable survivors of the campaign. Adding a fifth ("what if 60d_top1?") without going through the full walk-forward CV + matched-null testing chain produces unvalidated picks that look indistinguishable from validated ones. The audit trail in `PATCH_CROSS_ASSET_RANKING_*.md` is the bar for adding a profile.
