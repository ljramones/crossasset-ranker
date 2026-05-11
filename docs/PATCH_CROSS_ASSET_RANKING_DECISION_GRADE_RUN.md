# Patch: Decision-Grade Cross-Asset Ranking Run

Date: 2026-05-10
Run timestamp: `20260510T231114Z`
Output dir: `results/cross_asset_ranking_decision_run/`

## TL;DR verdict

| Candidate | Verdict | Why |
|---|---|---|
| `momentum_baseline` top-1 | **FAIL** | Only 2/5 folds pass random null at p≤0.05 |
| `momentum_baseline` top-2 | **FAIL** | BTC selected 54% of dates (>50% gate); minority of positive-IR folds |
| `hist_gradient_boosting` top-1 | **FAIL** | Only 1/5 folds pass random null at p≤0.05 |
| **`hist_gradient_boosting` top-2** | **PROVISIONAL PASS** | All hard gates met: positive mean IR, positive active return, 3/5 folds pass null, 3/5 folds positive IR, BTC 30%, all 6 assets selected ≥5% |
| `linear_regression` top-1 / top-2 | **FAIL** | Negative IR, loses to random nulls |
| Equal-weight (k=6) | benchmark | net Sharpe 1.12 reference; not a candidate |

Status: **promising cross-asset allocation candidate** for `hist_gradient_boosting` top-2 only. Result is borderline: median p-value 0.042 passes, geometric-mean p-value 0.105 doesn't. Mean IR is heavily lifted by split 0 (one strong fold). **Not a validated alpha — proceed to robustness testing, not production.**

## Input / cache source

Cached data only. No yfinance calls during the experiment.

```text
SPY        rows=  3315   2010-06-11 → 2026-05-07
QQQ        rows=  3346   2010-04-28 → 2026-05-07
IWM        rows=  3346   2010-04-28 → 2026-05-07
TLT        rows=  3346   2010-04-28 → 2026-05-07
GLD        rows=  3346   2010-04-28 → 2026-05-07
BTC-USD    rows=  2452   2015-01-09 → 2026-05-07
```

Cross-asset panel after strict inner-join: **2452 unique dates × 6 assets = 14,712 rows**, 2015-01-09 → 2026-05-07. Feature count: **32**.

## Dry-run command and result

```bash
python -m scripts.run_cross_asset_ranking_experiment \
  --dry-run --assets SPY QQQ IWM TLT GLD BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_decision_run \
  --forward-horizon 20 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 \
  --top-k 1 2 \
  --random-null-runs 500 \
  --random-state 42 \
  --run-purpose decision_grade --decision-grade
```

Result: prints resolved configuration; no data load, no model fit, no file writes.

## Execute command and result

Same arguments with `--dry-run` swapped for `--execute`. Run completed end-to-end with 5 walk-forward splits, 3 model fits per split (= 15 fits), 7,000 random-null allocations (5 splits × 2 top-k × 500 runs).

## Output files

```text
results/cross_asset_ranking_decision_run/
  cross_asset_ranking_summary_20260510T231114Z.csv          (1.2 KB)
  cross_asset_ranking_fold_details_20260510T231114Z.csv     (7.1 KB)
  cross_asset_ranking_scored_panel_20260510T231114Z.csv     (1.3 MB)
  cross_asset_ranking_allocations_20260510T231114Z.csv      (3.5 MB)
  cross_asset_ranking_portfolio_returns_20260510T231114Z.csv (779 KB)
  cross_asset_ranking_random_nulls_20260510T231114Z.csv     (2.3 MB)
  cross_asset_ranking_null_pvalues_20260510T231114Z.csv     (2.1 KB)
  cross_asset_ranking_report_20260510T231114Z.md            (10.3 KB)
  cross_asset_ranking_metadata_20260510T231114Z.json        (2.2 KB)
```

## Metadata safety validation

Required flags present in `cross_asset_ranking_metadata_20260510T231114Z.json`:

```json
{
  "decision_grade": true,
  "run_purpose": "decision_grade",
  "main_py_used": false,
  "prepare_experiment_used": false,
  "old_model_zoo_used": false,
  "optuna_used": false,
  "deep_models_used": false,
  "stacking_used": false,
  "data_downloaded": false,
  "split_count": 5,
  "panel_row_count": 14712,
  "panel_date_start": "2015-01-09",
  "panel_date_end": "2026-05-07",
  "feature_count": 32,
  "lag_convention": "weights.shift(1) * returns — weights at close of t apply to returns of t+1",
  "calendar_convention": "strict inner-join across asset date sets",
  "policy": "daily reallocation; equal weight among top-k by score"
}
```

## Aggregate metrics (mean across 5 folds)

| model | top_k | net_sharpe | IR vs ew | ann active ret | active calmar | max_dd | turnover | cost_drag | mean_assets |
|---|---|---|---|---|---|---|---|---|---|
| equal_weight | 6 | 1.118 | 0.000 | 0.000 | 0.000 | -0.192 | 0.004 | 0.0002 | 6.0 |
| hist_gradient_boosting | **2** | **1.385** | **0.408** | **0.077** | **1.794** | **-0.213** | 0.706 | 0.049 | 2.0 |
| hist_gradient_boosting | 1 | 1.292 | 0.579 | 0.138 | 1.756 | -0.262 | 0.858 | 0.065 | 1.0 |
| momentum_baseline | 2 | 1.137 | 0.396 | 0.113 | 1.084 | -0.183 | 0.331 | 0.023 | 2.0 |
| momentum_baseline | 1 | 0.715 | 0.270 | 0.177 | 1.356 | -0.291 | 0.361 | 0.026 | 1.0 |
| linear_regression | 2 | 0.517 | -0.556 | -0.100 | -0.249 | -0.254 | 0.544 | 0.031 | 2.0 |
| linear_regression | 1 | 0.229 | -0.446 | -0.158 | 0.154 | -0.366 | 0.802 | 0.045 | 1.0 |

## Random null p-values (500 nulls per fold)

Per-(model, top_k) summary:

| model | top_k | folds | folds_pass_05 | folds_pass_01 | median_p | geomean_p |
|---|---|---|---|---|---|---|
| `hist_gradient_boosting` | **2** | 5 | **3** | 0 | **0.042** | 0.105 |
| `hist_gradient_boosting` | 1 | 5 | 1 | 1 | 0.108 | 0.093 |
| `linear_regression` | 1 | 5 | 1 | 0 | 0.677 | 0.350 |
| `linear_regression` | 2 | 5 | 0 | 0 | 0.545 | 0.410 |
| `momentum_baseline` | 1 | 5 | 2 | 0 | 0.273 | 0.157 |
| `momentum_baseline` | 2 | 5 | 1 | 0 | 0.303 | 0.192 |

Per-fold detail:

| split | model | top_k | model IR | p_value |
|---|---|---|---|---|
| 0 | momentum_baseline | 1 | +1.808 | 0.020 ✓ |
| 0 | momentum_baseline | 2 | +1.613 | 0.042 ✓ |
| 0 | hist_gradient_boosting | 1 | +1.055 | 0.120 |
| 0 | hist_gradient_boosting | **2** | **+1.720** | **0.036 ✓** |
| 0 | linear_regression | 1 | -1.245 | 0.868 |
| 0 | linear_regression | 2 | -1.810 | 0.944 |
| 1 | momentum_baseline | 1 | +1.413 | 0.044 ✓ |
| 1 | hist_gradient_boosting | 1 | +1.073 | 0.094 |
| 1 | hist_gradient_boosting | **2** | -0.017 | 0.363 |
| 2 | (all models) | * | **strongly negative** | all > 0.5 (failure regime) |
| 3 | hist_gradient_boosting | 1 | +2.185 | **0.006 ✓✓** |
| 3 | hist_gradient_boosting | **2** | **+1.229** | **0.042 ✓** |
| 3 | linear_regression | 1 | +1.719 | 0.018 ✓ |
| 4 | hist_gradient_boosting | 2 | +1.430 | 0.024 ✓ |
| 4 | (other models) | * | mixed | all > 0.10 |

Notable observations:

- **Split 2 was a universally bad fold for active allocation** — every model on every top-k policy posted a negative IR with p > 0.4. Either a regime where forward 20-day risk-adjusted return became unrankable from these features, or a transition period where the trained model's prior assumptions inverted. Worth investigating in the robustness pass.
- **Split 3 was a strong fold** — HGB top-1 hit p = 0.006 (best single result in the run). Linear regression also came alive in this fold, suggesting the regime favored learnable cross-asset structure.
- **HGB top-2** passed at p ≤ 0.05 in 3 of 5 folds (splits 0, 3, 4). The two failures were split 1 (essentially zero IR) and split 2 (universal failure regime).

## Asset selection diagnostics — BTC dominance check

| model | top_k | BTC % | dominance flag (>50%) |
|---|---|---|---|
| `hist_gradient_boosting` | 1 | 18.9% | False |
| **`hist_gradient_boosting`** | **2** | **30.2%** | **False** |
| `linear_regression` | 1 | 20.9% | False |
| `linear_regression` | 2 | 33.4% | False |
| `momentum_baseline` | 1 | 46.0% | False |
| `momentum_baseline` | 2 | **54.0%** | **True ⚠** |
| equal_weight | 6 | 100.0% | by design |

Distinct assets selected in ≥5% of test dates:

| model | top_k | distinct assets ≥5% |
|---|---|---|
| `hist_gradient_boosting` | 1 | 6 |
| `hist_gradient_boosting` | 2 | 6 |
| `linear_regression` | 1 | 6 |
| `linear_regression` | 2 | 6 |
| `momentum_baseline` | 1 | 5 (SPY 2.1%, below cutoff) |
| `momentum_baseline` | 2 | 6 |

Per-asset average selection across all model/top_k strategies (excluding equal-weight):

| asset | avg pct held |
|---|---|
| BTC-USD | 43.3% |
| GLD | 39.8% |
| QQQ | 37.6% |
| IWM | 32.3% |
| SPY | 31.7% |
| TLT | 29.5% |

The momentum baseline is the BTC-tilted strategy. **HGB top-2's selections are well-distributed** — no single-asset dominance, every asset gets a meaningful share.

## Fold stability diagnostics

| model | top_k | folds | mean IR | median IR | min IR | max IR | std IR | folds (+) | folds (−) |
|---|---|---|---|---|---|---|---|---|---|
| `hist_gradient_boosting` | 1 | 5 | 0.579 | 1.055 | -2.217 | 2.185 | 1.652 | 4 | 1 |
| **`hist_gradient_boosting`** | **2** | 5 | 0.408 | **1.229** | -2.319 | 1.720 | 1.663 | **3** | 2 |
| `linear_regression` | 1 | 5 | -0.446 | -0.802 | -1.245 | 1.719 | 1.234 | 1 | 4 |
| `linear_regression` | 2 | 5 | -0.556 | -0.697 | -1.810 | 1.065 | 1.037 | 1 | 4 |
| `momentum_baseline` | 1 | 5 | 0.270 | 0.105 | -1.741 | 1.808 | 1.414 | 3 | 2 |
| `momentum_baseline` | 2 | 5 | 0.396 | -0.024 | -0.331 | 1.613 | 0.852 | 2 | 3 |

Single-fold concentration check: removing the single best fold from each candidate's mean IR:

| model | top_k | mean IR (all) | mean IR (drop best) | drop |
|---|---|---|---|---|
| `hist_gradient_boosting` | 1 | 0.579 | 0.178 | -69% |
| `hist_gradient_boosting` | **2** | 0.408 | 0.081 | -80% |
| `momentum_baseline` | 1 | 0.270 | -0.110 | wipes signal |
| `momentum_baseline` | 2 | 0.396 | 0.094 | -76% |

**The signal is concentrated in 1–2 strong folds for every candidate.** This is a real concern but not surprising at 5 folds — and HGB top-2's distribution still has 3 of 5 folds positive, with positive median IR of 1.23. This is the kind of result that *demands* robustness testing (different seeds, different rebalance frequencies, different universes) before any allocation decision.

## Gate evaluation — `hist_gradient_boosting` top-2

| Gate | Result | Pass? |
|---|---|---|
| IR vs equal-weight > 0 | 0.408 | ✓ |
| Annualized active return > 0 | 0.077 | ✓ |
| Random top-k p ≤ 0.05 in majority of folds | 3/5 | ✓ |
| Positive IR in majority of folds | 3/5 (median +1.23) | ✓ |
| Not dominated by single fold | best fold contributes ~80% of mean | ⚠ borderline |
| Not dominated by BTC alone | BTC = 30.2% of selections | ✓ |
| At least 3 distinct assets selected ≥5% of dates | 6 ✓ | ✓ |
| Turnover and cost drag reasonable | turnover 0.71/day, cost drag 0.049 ann; **net Sharpe 1.39 still beats benchmark 1.12** | ✓ (cost is paid, signal survives) |

**Provisional verdict: PASS** with explicit caveats on fold-concentration of signal and an unexplained universal failure in split 2.

## Stop / go decision

```text
Status: promising cross-asset allocation candidate for hist_gradient_boosting top-2.
Next step: robustness testing, not production.
```

The other candidates fail decision-grade:

- **Momentum baseline**: BTC-dominant in top-2 (54%); only 1–2 folds pass null in either configuration.
- **Linear regression / Ridge**: actively negative IR; loses to random allocation.
- **HGB top-1**: positive IR but only 1 fold passes null at 0.05 — too lucky-looking.

## Limitations

- **Sample size for null inference is per-fold (n=500), not pooled.** A pooled test (Fisher's combined test or a stratified permutation across splits) would tighten the inferential picture. Geometric-mean p of 0.105 for HGB top-2 is a reasonable approximation and it's not significant.
- **Only 5 splits.** Each split's test window is 252 days; the panel is bounded by BTC-USD's 2015-01-09 inception. A pre-2015 sub-experiment that excludes BTC would more than double the effective history but changes the universe.
- **Daily reallocation drives turnover to ~0.7 of NAV per day** for HGB top-2. The strategy still earns net Sharpe 1.39 vs equal-weight's 1.12, but a 5- or 20-day rebalance cadence would materially change the cost picture and is essential to test before any allocation decision.
- **Split 2 is unexplained.** Every model failed in this period. Likely a regime where the trained model's prior assumptions inverted, but the run does not isolate the cause. A per-split inspection of which assets dominated forward returns vs. which the model selected would diagnose this.
- **Single random seed.** Re-running with seeds 1, 7, 13 etc. and verifying HGB top-2 still passes would harden the verdict.
- **Score-based ranking is regression-fit, not rank-fit.** The model is trained to predict the value of `forward_20d_risk_adjusted_return`, not the cross-sectional rank. A LambdaRank / pairwise-ranking objective could be a meaningful test of whether the bottleneck is the model or the loss.

## Robustness follow-ups (recorded for the next pass — do not change in this run)

1. **Per-asset feature normalization.**
   - Today's panel pools features raw across assets. BTC's volatility scale (~5×) dominates the feature distribution.
   - Options to test: per-asset rolling z-score for `return_1d/5d/20d/vol_ratio/realized_vol_20`; cross-sectional z-score per date for the same.
   - Hypothesis: linear regression's universal failure is partly a scale problem.

2. **Rebalance frequency sensitivity.**
   - Current prototype reallocates every bar.
   - Test cadences: 5-day, 10-day, 20-day. Compare net Sharpe after costs.
   - Implementation: hold weights constant for `R` bars, then rescore. Should be a single CLI flag.

3. **Cash-out option.**
   - Today, top-k always allocates even when all model scores are negative.
   - Test: if all asset scores fall below a validation-set-derived threshold, hold cash for that bar.
   - Hypothesis: split 2's universal failure may have been a regime where every score was negative; cash would have outperformed.

4. **Pre-BTC sub-period.**
   - Run on the 5-asset universe (drop BTC-USD) for 2010-04 → 2015-01 plus the panel period — gives more pre-2015 history and isolates whether the result depends on BTC.

5. **Different random seeds for the model fits and the null sampler.**
   - If HGB top-2's PASS doesn't survive seeds {1, 7, 13, 99}, the result was a fluke.

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded` are all `false` in metadata.
- The static-import test (`test_no_legacy_or_optuna_imports_in_experiment_modules`) continues to enforce these in source.
- The `--prepare-missing` flag was not set; no yfinance fetch occurred.
- All raw caches were already on disk from `docs/PATCH_DATA_REBUILD_ACTIVE_TRACK_REPORT.md` and were used as-is.
- No champion manifest was modified.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Cached universe data confirmed | ✓ all 7 (6 assets + ^VIX) present |
| 2 | Dry-run succeeds | ✓ |
| 3 | Decision-grade execute succeeds | ✓ |
| 4 | 500 random nulls used | ✓ 500 per (split, top_k) — 7,000 total |
| 5 | Multiple splits used | ✓ 5 walk-forward splits |
| 6 | Output bundle generated | ✓ 9 files |
| 7 | No old workflows | ✓ |
| 8 | No Optuna | ✓ |
| 9 | No deep / stacking | ✓ |
| 10 | No data download | ✓ |
| 11 | Summary metrics extracted | ✓ |
| 12 | Random null p-values extracted | ✓ |
| 13 | Asset selection concentration inspected | ✓ |
| 14 | Fold stability inspected | ✓ |
| 15 | Stop/go verdict documented | ✓ above |
| 16 | Documentation report created | ✓ this file |

## Recommended next step

Robustness testing of `hist_gradient_boosting` top-2:

1. **Rebalance frequency sweep**: rerun at 5-day and 20-day rebalance cadences. If the signal survives at 20-day with substantially lower turnover, that's a much stronger result than daily.
2. **Seed sensitivity**: rerun with `--random-state` ∈ {1, 7, 13, 99} for both model fits and null sampling. PASS only if HGB top-2 still passes ≥3/5 folds in 3 of 4 seeds.
3. **Investigate split 2 failure**: produce a per-split asset-vs-model comparison to see what HGB chose during split 2's test window vs. what actually outperformed.
4. **Per-asset feature normalization sensitivity**: rerun with per-asset rolling z-scores for the return/vol features. If linear regression also comes alive, the signal is real but the model was undermined by scale; if HGB's signal degrades, the prior result was leaning on BTC's feature scale being distinguishable.

Until those four pass cleanly, treat the current verdict as **provisional**.
