# Patch: Expanded Cross-Asset Ranking Universe + LambdaRank

Date: 2026-05-11
Run timestamp: `20260511T122454Z` → `results/cross_asset_ranking_expanded_lambdarank_5d/`

## TL;DR verdict

**Universe expansion from 6 to 18 ETFs did not produce a validated cross-sectional ranking signal under LambdaRank.** Per-date Spearman ρ went from **+0.014 to +0.010** (worse overall, marginally), mean IR went from **+0.337 to −0.174** (top-1) and from **−0.253 to +0.002** (top-2), and 0/5 folds clear p ≤ 0.05 in either policy. The "regime-aware selection" pattern that made the 6-asset top-1 result behaviorally interesting did not survive — instead, the 18-asset top-2 model concentrates heavily on BTC + USO in some splits (split 2: BTC 59.5%, USO 53.6%; split 3: BTC 69.4%, USO 56.3%).

The structural diagnosis from the prior patch is confirmed: **per-date Spearman near zero is not a universe-width problem; the features themselves do not carry strong forward-20d cross-sectional ranking signal.** The next correct test is a **target/horizon alignment** (e.g. forward-5d target for 5d rebalance) or a **different ranking objective** (forward volatility / drawdown rather than risk-adjusted return). Continuing to tune the current setup at 6 or 18 assets is unlikely to find a signal that isn't there.

```yaml
expanded_18_asset_lambdarank_5d_normed:
  status: failed_decision_grade
  top1_mean_ir: -0.174
  top2_mean_ir: +0.002
  per_date_spearman_overall: +0.0097
  folds_pass_p05: 0/5 (both top-1 and top-2)
  median_p_top1: 0.487
  median_p_top2: 0.431
  drop_best_fold_mean_ir_top1: -0.255
  conclusion: |
    Universe expansion did not unlock cross-sectional ranking signal.
    Per-date discrimination remains essentially zero on a 5d-rebalance,
    20d-target setup. Issue is structural to target/horizon, not universe.
```

## Universe and cache status

Expanded universe (18 ETFs):

```text
Equity:           SPY, QQQ, IWM, DIA, EFA, EEM
Rates / bonds:    TLT, IEF, SHY, LQD, HYG
Commodities:      GLD, SLV, USO, DBA
Defensive / FX:   UUP, VNQ
Crypto:           BTC-USD
```

Cache fetch on this machine via `data.market_cache.ensure_universe_cache` (single batch, yfinance):

| ticker | rows | first date | last date | status |
|---|---|---|---|---|
| SPY | 4112 | 2010-01-04 | 2026-05-08 | pre-cached (prior patch) |
| QQQ | 4112 | 2010-01-04 | 2026-05-08 | pre-cached |
| IWM | 4112 | 2010-01-04 | 2026-05-08 | pre-cached |
| **DIA** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **EFA** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **EEM** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| TLT | 4112 | 2010-01-04 | 2026-05-08 | pre-cached |
| **IEF** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **SHY** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **LQD** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **HYG** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| GLD | 4112 | 2010-01-04 | 2026-05-08 | pre-cached |
| **SLV** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **USO** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **DBA** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **UUP** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| **VNQ** | 4112 | 2010-01-04 | 2026-05-08 | **fetched** |
| BTC-USD | 4254 | 2014-09-17 | 2026-05-10 | pre-cached |
| ^VIX | 4112 | 2010-01-04 | 2026-05-08 | pre-cached |

All 12 newly-fetched tickers produced full 2010-01-04 → 2026-05-08 spans with no ticker failures. The fetch wrote 12 new CSV files plus 12 new `_daily.meta.json` sidecars (yfinance version 4.6.0, fetch_date_utc 2026-05-11). **No ticker failures, no silent drops.**

Prepared-frame shape per asset (after `prepare_single_asset_feature_frame` warmup):

```text
SPY 3315, QQQ 3346, IWM 3346, DIA 3346, EFA 3346, EEM 3346
TLT 3346, IEF 3346, SHY 3346, LQD 3346, HYG 3285
GLD 3346, SLV 3346, USO 3346, DBA 3284, UUP 3346, VNQ 3331
BTC-USD 2452 (2015-01-09 → 2026-05-07)
```

VNQ's last prepared row is 2026-04-16, ~3 weeks behind the other ETFs — this trims split 4's test window slightly. HYG and DBA also have ~60 fewer rows than the other ETFs but no recent gap. None of these required dropping a ticker.

## Commands

Dry-run accepted the 18-asset universe and printed `models: ['lambdarank']`, `feature_normalization: per_asset_train_zscore`, `rebalance_every: 5`.

Execute:

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM DIA EFA EEM TLT IEF SHY LQD HYG GLD SLV USO DBA UUP VNQ BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_expanded_lambdarank_5d \
  --forward-horizon 20 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 500 --random-state 42 \
  --rebalance-every 5 --feature-normalization per_asset_train_zscore \
  --run-purpose decision_grade --decision-grade
```

5 walk-forward splits × 2 top-k × 500 random nulls = 5,000 null allocations on the 18-asset panel. `metadata.json` records `data_downloaded: false` for the experiment run itself (the yfinance fetch happened in the prior step via `ensure_universe_cache`, not within the experiment). All other safety flags `false`.

## Output files

9 timestamped files under `results/cross_asset_ranking_expanded_lambdarank_5d/`.

## Head-to-head — LambdaRank 6-asset vs 18-asset (both 5d rebalance, normalized features)

| run / policy | mean IR | net Sharpe | turnover | cost drag | folds (+IR) | folds p≤0.05 | median p | BTC % | top-3 conc % | distinct ≥5% | drop-best mean | max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lambdarank_6 top-1 | +0.337 | 0.866 | 0.261 | 0.017 | **4/5** | 0/5 | **0.216** | 22.8 | 63.3 | 6 | **+0.229** | -0.30 |
| Lambdarank_6 top-2 | -0.253 | 0.638 | 0.202 | 0.012 | 2/5 | 0/5 | 0.609 | **40.6** | 57.7 | 6 | -0.546 | -0.24 |
| **Lambdarank_18 top-1** | **-0.174** | 0.172 | 0.290 | 0.015 | 2/5 | 0/5 | 0.487 | 24.8 | 45.8 | 5 | -0.255 | -0.33 |
| **Lambdarank_18 top-2** | **+0.002** | 0.487 | 0.252 | 0.014 | 2/5 | 0/5 | 0.431 | **39.0** | 42.3 | **16** | -0.163 | -0.25 |
| equal-weight 6 ref | 0.000 | 1.118 | 0.004 | 0.0002 | n/a | n/a | n/a | 100 | n/a | 6 | n/a | -0.19 |
| equal-weight 18 ref | 0.000 | 1.018 | 0.004 | 0.0002 | n/a | n/a | n/a | n/a | n/a | 18 | n/a | -0.13 |

The 18-asset equal-weight net Sharpe (1.018) is slightly lower than the 6-asset (1.118), so a model that produces flat IR with 18 assets has effectively earned the equal-weight return. That is what top-2 does (+0.002 IR, net Sharpe 0.49 — below equal-weight after cost). Top-1 actively loses (−0.17 IR, net Sharpe 0.17).

## The headline metric — per-date Spearman score-vs-target rank correlation

| run | mean overall | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| Lambdarank_6 5d | **+0.0138** | -0.110 | +0.078 | +0.052 | +0.040 | +0.010 |
| **Lambdarank_18 5d** | **+0.0097** | -0.018 | -0.006 | +0.026 | **+0.083** | -0.037 |

Going from 6 to 18 assets:
- Overall mean Spearman dropped from +0.014 to +0.010.
- Folds with positive mean ρ dropped from 4/5 to 2/5.
- The single best fold (split 3) improved from +0.040 to +0.083 — but that's also the smallest gain anywhere, and it didn't translate to a clear IR win.

This is the cleanest possible falsification of the universe-width hypothesis. The user's pre-specified threshold for "minimum interesting" was per-date Spearman ≥ +0.05; the result is +0.0097 overall and only +0.083 in the single best fold. The hypothesis that LambdaRank needed more cross-sectional breadth to learn ranking is **not supported by the data**.

## Per-fold IR detail

```text
top-1:  s0:-0.039  s1:-0.328  s2:+0.059  s3:+0.152  s4:-0.713
top-2:  s0:-0.784  s1:+0.375  s2:-0.133  s3:+0.665  s4:-0.112
```

Variance dropped versus the 6-asset run (top-1 std went from ~0.6 to ~0.32). But that's variance compression around a smaller mean, not signal stability. No fold reaches IR > +1.0 in either top-k; the best single fold is top-2 split 3 at +0.665.

## Asset selection — LambdaRank top-2, 18 assets (assets selected on ≥ 8% of dates per split)

```text
split 0 (2020-02 → 2021-02, COVID rebound):
  SHY 31.7%  TLT 24.6%  BTC-USD 23.8%  DBA 23.8%  IEF 13.9%  GLD 11.9%  EFA 9.9%
  (defensive cash + bonds during a recovery — wrong; lost -0.78 IR)

split 1 (2021-02 → 2022-02, sideways / inflation buildup):
  IEF 21.8%  SHY 21.8%  HYG 19.8%  QQQ 17.9%  EFA 14.7%  EEM 13.9%  UUP 13.9%
  VNQ 11.9%  USO 11.9%  DIA 9.9%
  (broad allocation; won +0.38 IR — only fold to show real diversification)

split 2 (2022-02 → 2023-05, bear / rate shock):
  BTC-USD 59.5%  USO 53.6%  GLD 24.6%  SLV 21.8%  DBA 11.9%
  (oil + crypto concentration; oil was actually OK but BTC was the worst; -0.13 IR)

split 3 (2023-05 → 2024-08, AI rally):
  BTC-USD 69.4%  USO 56.3%  GLD 29.8%  SLV 13.9%
  (extreme concentration in 2 assets; +0.67 IR — got lucky that BTC ran)

split 4 (2024-08 → 2025-08, risk-on):
  BTC-USD 36.5%  VNQ 19.8%  HYG 17.9%  SLV 17.9%  USO 17.9%  IWM 13.9%
  SHY 13.9%  EFA 9.9%
  (broader allocation again; -0.11 IR)
```

Two patterns emerge:

1. **The model alternates between extremely concentrated and broadly diversified selections.** Splits 2 and 3 are essentially "BTC + USO" bets (top-2 inclusion >50% of dates for each of those two assets). Splits 1 and 4 are broadly diversified across 8-10 assets. There is no consistent "regime detector" — the model is opportunistically betting on whatever the prevailing trend looked like in the train window.
2. **The single-fold win in split 3 (+0.67 IR top-2) is a BTC-rally story.** BTC + USO was selected 69% + 56% of split-3 dates. BTC ran hard in 2023-2024, so this worked, but it's not a ranking edge — it's a momentum bet that happened to align with realized BTC strength.

The 6-asset LambdaRank top-1 had cleaner regime-aware selection (TLT in bond-friendly periods, GLD in bear, BTC in risk-on). The 18-asset version lost that property — too many bond/cash alternatives competed for low-vol-favored slots, and the model defaulted to either ultra-conservative cash/bond combinations or ultra-aggressive BTC + commodities concentrations.

## Pass criteria evaluation

| Criterion | Threshold | Top-1 result | Top-1 pass? | Top-2 result | Top-2 pass? |
|---|---|---|---|---|---|
| Mean IR > 0 | > 0 | -0.174 | ✗ | +0.002 | borderline (zero) |
| Annualized active return > 0 | > 0 | -0.027 | ✗ | -0.009 | ✗ |
| Random null p ≤ 0.05 in majority of folds | ≥ 3/5 | 0/5 | ✗ | 0/5 | ✗ |
| Per-date Spearman improves materially | target +0.05, ideally +0.10 | +0.0097 | ✗ | +0.0097 | ✗ |
| Drop-best-fold mean IR positive | > 0 | -0.255 | ✗ | -0.163 | ✗ |
| Result not BTC-dominated | < 50% | 24.8% | ✓ | 39.0% | borderline |
| No single asset dominates selection (per-split) | look at split-3 | BTC 25.8% top-1 | ✓ | BTC 69.4% top-2 in split 3 | ✗ |
| Turnover reasonable | similar to 5d baseline | 0.290 | ✓ | 0.252 | ✓ |

**Top-1: 2 of 8 criteria pass. Top-2: 3 of 8 (with one borderline).** This is a clear decision-grade failure on every economic and statistical gate that matters.

## Stop / go verdict

```yaml
expanded_universe_lambdarank:
  status: failed
  universe_expansion_hypothesis: rejected
  reason: |
    Going from 6 to 18 assets did not improve per-date Spearman
    (+0.014 → +0.010), did not improve mean IR (it dropped),
    and produced split-level asset concentration (BTC + USO >50%
    in two splits) that is worse than the 6-asset version's
    regime-aware selection.
  next_hypothesis: target_horizon_alignment_or_target_reset
```

The 6-asset LambdaRank top-1 result (mean IR +0.337, 4/5 positive folds, Spearman +0.014, regime-aware selection) was the high-water mark of the cross-asset-ranking-by-forward-20d-risk-adjusted-return track. The 18-asset run does not improve on it; it makes things worse. That settles the universe-width question.

## What this leaves us with

The cross-asset ranking framework now has three pieces of evidence that the **target** is the binding problem, not the model or the universe:

1. HGB regression (6-asset): per-date Spearman +0.051, but proven to be a scale-bet not a ranking edge (split-2 forensics).
2. LambdaRank (6-asset): per-date Spearman +0.014 with normalized features. Regime-aware selection, but signal too weak to clear nulls.
3. LambdaRank (18-asset): per-date Spearman +0.010. Universe expansion did not unlock additional signal.

All three setups use `forward_20d_risk_adjusted_return = forward_20d_return / trailing_20d_realized_vol` as the target. The model fits this target through a 5d rebalance window — meaning the target horizon (20d) is **4× longer** than the decision horizon (5d). The mismatch is structural: the model is trained to predict 20-day-ahead outcomes but evaluated on 5-day-ahead trading decisions.

The user's pre-specified path forward when expanded-universe LambdaRank fails:

> If expanded LambdaRank still has near-zero Spearman, then the issue is probably not universe width. The features may not support 20-day forward ranking, and we should move to either:
> 1. forward volatility / risk ranking
> 2. different horizon alignment, e.g. 5-day target for 5-day rebalance

Both are reasonable. I would recommend **(2) first** — target/horizon alignment — because it is a smaller change and tests the simpler hypothesis (target-horizon mismatch is the problem) before invoking the larger one (the target metric is wrong).

Concretely, the next experiment should keep the 6-asset universe (or stay at 18, doesn't matter much given the result) and run LambdaRank with:

- `forward_horizon = 5` (5-day forward return)
- `vol_window = 20` (keep the same risk normalization)
- `rebalance_every = 5`
- normalized features
- 500 nulls per fold

If 5d target + 5d rebalance + 5d evaluation horizon also produces Spearman near zero, then conclude the target metric (forward risk-adjusted return) doesn't carry enough cross-sectional structure at any horizon supported by the current feature set, and pivot to forward volatility / drawdown ranking.

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used` all `false` in metadata.
- The expanded-universe yfinance fetch was an *explicitly authorized* prior step via `ensure_universe_cache` and is documented in the cache status table above. No silent fetches occurred within the experiment runner itself (`--prepare-missing` was not set; `data_downloaded: false`).
- No ticker failures; no silent drops.
- All cached files have `_daily.meta.json` sidecars with fetch metadata for future drift audits.
- No champion manifest changes.
- No existing result files overwritten.
- Static-import test continues to enforce no-legacy / no-Optuna / no-deep / no-stacking invariants.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Expanded universe cache checked / fetched | ✓ 12 fetched, 6 pre-cached, 0 failures |
| 2 | Dry-run passes | ✓ |
| 3 | Decision-grade run completes or failure documented | ✓ completed |
| 4 | No legacy workflow used | ✓ |
| 5 | No Optuna / deep / stacking | ✓ |
| 6 | No untracked silent data source changes | ✓ fetch logged explicitly above |
| 7 | Report created | ✓ this file |

## Recommended next step

**Target/horizon alignment: rerun LambdaRank with `forward_horizon = 5` and `rebalance_every = 5`.** This is a one-flag change (the existing CLI already exposes `--forward-horizon`). If the 5d-target + 5d-rebalance version still has Spearman near zero and no folds passing p ≤ 0.05, the conclusion is that **`forward_20d_risk_adjusted_return` as a target does not carry enough cross-sectional structure to support a ranking strategy on this feature set**, and the next pivot should be to **forward-volatility ranking** (predict which assets will be the LEAST volatile, allocate accordingly) or **forward-drawdown ranking** (predict which assets will have the smallest drawdown). Both are simpler economic targets with arguably more predictive structure in stationary features.

Do **not** run seed sensitivity, deeper model tuning, or additional universe variations until the target/horizon question is answered. The diagnosis chain so far points consistently at the target, not the model or the universe.
