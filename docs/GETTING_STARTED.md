# Getting Started

First-prediction orientation. Aim: from a fresh clone (or a fresh look after months away) to a real prediction in under 10 minutes.

## What this is

This is a personal-investor quantitative system that picks 1–2 ETFs to hold over a 5- or 20-trading-day horizon out of an 18-asset universe (major equity indices, bonds, commodities, currency, real estate, BTC). The model is a LambdaRank ranker on a Gradient-Boosted Decision Tree backbone (LightGBM). Empirically it produces a modest but real edge over equal-weight allocation (~16% Sharpe improvement in backtest) — not a "strong signal" alpha model.

Read [TECHNICAL_DESCRIPTION.md](TECHNICAL_DESCRIPTION.md) for the algorithm, feature engineering, evaluation metrics, and the empirical universe ceiling that bounds what this system can do.

## The four deployable profiles

LambdaRank only. Ridge profiles were tested and dropped (Spearman below trivial baseline) — see [RIDGE_BASELINE_RESULTS.md](RIDGE_BASELINE_RESULTS.md). Any future model swap must respect per-date grouping (pooled regression fails the cross-sectional comparability test).

| Profile | Horizon | Top-k | Notes |
| --- | ---: | ---: | --- |
| `v3_5d_top2` | 5 trading days | 2 | Best raw economics (Sharpe ~1.18, IR +0.748). Highest turnover. |
| `20d_top1` | 20 trading days | 1 | Best per-trade efficiency. Concentration risk (single asset). |
| `20d_top2` | 20 trading days | 2 | Concentration-hedged variant of `20d_top1`. |
| `v1_5d_top2` | 5 trading days | 2 | Rank-quality reference. Retained for validation, **not for deployment.** |

If unsure, start with `20d_top1` — it has the lowest turnover, the most positive drop-best-fold IR, and the smallest cost drag.

## Quick start

```bash
# 1. Sanity check: caches exist
ls -la data/multi_asset_cache/

# 2. Refresh caches and predict in one command
uv run python -m scripts.run_operational_prediction \
    --profile 20d_top1 \
    --output-dir predictions/

# 3. Inspect today's pick (substitute the date the wrapper used)
ls -lt predictions/*_20d_top1.json | head -1
uv run python -c "import json, sys; d=json.load(open(sys.argv[1])); print('Pick:', d['top_k_picks']); print('Top features:', [f['feature'] for f in d['feature_importance'][:5]])" predictions/<latest-date>_20d_top1.json

# 4. Confirm the operational log captured the run
tail -1 predictions/operational_log.jsonl

# 5. Confirm the prediction log captured the record
tail -1 predictions/predictions_log.jsonl
```

If you only want to see what a run *would* do without touching anything, use `--dry-run` on `scripts.run_live_prediction` directly:

```bash
uv run python -m scripts.run_live_prediction --dry-run --profile 20d_top1
```

## What to do with the output

The wrapper prints the top-*k* picks to stdout and writes a JSON record per prediction. **Trade execution is manual** — there is no brokerage integration. The workflow is:

1. Read the picks from stdout.
2. Place the order(s) yourself at your broker, equal-weighted across the top-*k* assets.
3. Log the trade in `predictions/trade_log.csv` (date, action, ticker, executed price, position size). Create the file with header `date,action,ticker,price,position_size,notes` on first use.
4. Hold until the next rebalance (5 or 20 trading days forward, depending on the profile).

For the full procedure including weekly/monthly/quarterly reviews, see [OPERATIONAL_RUNBOOK.md](OPERATIONAL_RUNBOOK.md).

## Where to go next

- **Day-to-day procedure** → [OPERATIONAL_RUNBOOK.md](OPERATIONAL_RUNBOOK.md)
- **Something broke** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **How the algorithm works** → [TECHNICAL_DESCRIPTION.md](TECHNICAL_DESCRIPTION.md)
- **Why Ridge isn't in the profile list** → [RIDGE_BASELINE_RESULTS.md](RIDGE_BASELINE_RESULTS.md)
- **Why the live script's design differs from the campaign runner** → [PATCH_LIVE_PREDICTION_SCRIPT.md](PATCH_LIVE_PREDICTION_SCRIPT.md)
- **Why VIX is always refreshed** → [PATCH_OPERATIONAL_WRAPPER.md](PATCH_OPERATIONAL_WRAPPER.md)
- **Full campaign audit trail** → `PATCH_CROSS_ASSET_RANKING_*.md` (chronological decision log)

When in doubt about a non-obvious choice the system makes, the patch documents are the authoritative source.
