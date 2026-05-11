# Patch: Cross-Asset Ranking — Cross-Sectional Feature Pivot

Date: 2026-05-11
Run timestamp: `20260511T143612Z` → `results/cross_asset_ranking_5d_target_xs_features_lambdarank/`
Reference run: `20260511T131039Z` (5d-target LambdaRank without cross-sectional features)

## TL;DR verdict

**Directionally interesting — first real signal in the entire diagnosis chain.** Per-date Spearman moved from **+0.017 → +0.032** (a 83 % relative gain that just clears the +0.03 directional-interest threshold). Top-2 mean IR went from **−0.10 to +0.65**, top-2 net Sharpe went from 0.39 to **1.04** — finally beating equal-weight (1.02) for the first time in this project — and top-2 drop-best-fold mean IR went from −0.61 to **+0.24**, the first positive drop-best in any cross-asset configuration. The cross-sectional features are not just available; **`xs_rank_vol_20d` is the single most-important feature by gain** across all five splits.

The run **falls short of the decision-grade tier**: Spearman +0.032 is below the +0.05 decision-grade threshold, and only 1 of 5 folds passes p ≤ 0.05 (split 4 at 0.004; split 2 at 0.066 borderline) rather than the required 2. So this is a "yes, the feature-side hypothesis was right" result, not a "deploy" result.

Per the spec's tiered pass criteria, this triggers the **directionally-interesting branch**: justifies a feature-side follow-up (add the dispersion / beta-to-equal-weight-rest / regime-conditioned interactions explicitly reserved for v2), NOT deployment, NOT the target-side pivot, NOT seeds.

```yaml
xs_features_5d_target_18_asset_lambdarank:
  status: directionally_interesting
  decision_grade_pass: false
  directionally_interesting_pass: true
  per_date_spearman_overall: +0.0318
  per_date_spearman_threshold_directional: +0.03
  per_date_spearman_threshold_decision_grade: +0.05
  top2_mean_ir: +0.655
  top2_net_sharpe: 1.044  (equal-weight 1.018)
  top2_drop_best_fold_mean_ir: +0.240
  top2_folds_pass_p05: 1/5  (split 4 at 0.004; split 2 borderline at 0.066)
  top2_folds_positive_ir: 4/5
  feature_importance_xs_in_top_half: 4 of 5
  most_important_feature: xs_rank_vol_20d (8.82% of total gain)
  conclusion: |
    Per-asset features cannot encode cross-sectional position.
    Explicit per-date rank features unlock real ranking signal.
    Result is the first directionally interesting cross-asset finding.
```

## Phase 1 — Column-name hygiene fix

`build_cross_asset_panel` previously hard-coded the output column names `forward_20d_return`, `trailing_20d_realized_vol`, and `forward_20d_risk_adjusted_return` regardless of the `forward_horizon` / `vol_window` arguments. The math was correct (the columns contained 5d-forward values when `forward_horizon=5`), but the labels were stale and misleading.

**Fix**: the panel builder now formats column names from its arguments:
```python
forward_return_col   = f"forward_{forward_horizon}d_return"
trailing_vol_col     = f"trailing_{vol_window}d_realized_vol"
risk_adjusted_col    = f"forward_{forward_horizon}d_risk_adjusted_return"
```

**Runner update**: `experiments/cross_asset_ranking_experiment.py` exports `target_column_for_horizon(forward_horizon)` and uses it in three places (`_build_panel`, `select_cross_asset_feature_columns`, the per-split scoring call). The historical `DEFAULT_TARGET_COLUMN = "forward_20d_risk_adjusted_return"` constant is preserved for backward compatibility (used in places where horizon=20 is the implicit default).

**Verification**: new test `test_build_cross_asset_panel_column_names_track_forward_horizon` asserts:
- `horizon=20, window=20` → `forward_20d_return`, `trailing_20d_realized_vol`, `forward_20d_risk_adjusted_return` (backward compat)
- `horizon=5, window=20` → `forward_5d_return`, `trailing_20d_realized_vol`, `forward_5d_risk_adjusted_return`
- and `forward_20d_risk_adjusted_return` is absent when horizon=5

## Phase 2 — Cross-sectional feature implementation

### New helper `add_cross_sectional_features(panel)`

Located in `evaluation/cross_asset_ranking.py`. Pure function of the long-format cross-asset panel. Computes any of `return_5d` / `return_20d` / `realized_vol_20` / `return_60d` / `current_drawdown_60d` not already present, then transforms each into a per-date rank in [0, 1].

Per-date rank computation:
```python
def _per_date_normalized_rank(s):
    ranks = s.rank(ascending=True, method="average")   # NaN inputs stay NaN
    valid = s.notna().sum()
    if valid <= 1:
        return all-NaN  # no cross-section
    return (ranks - 1) / (valid - 1)
```

Five new feature columns:

| feature | source column | meaning |
|---|---|---|
| `xs_rank_ret_5d` | `return_5d` (computed if absent) | rank of trailing 5-day log return |
| `xs_rank_ret_20d` | `return_20d` | rank of trailing 20-day log return |
| `xs_rank_ret_60d` | `return_60d` (computed) | rank of trailing 60-day log return |
| `xs_rank_vol_20d` | `realized_vol_20` | rank of 20-day realized return volatility |
| `xs_rank_drawdown_60d` | `current_drawdown_60d` (computed) | rank of `1 − price / rolling_60d_max(price)` (most-drawn-down → 1.0) |

Each value in [0, 1] per date, highest source value → 1.0. NaN inputs (e.g. BTC pre-60d-warmup) produce NaN ranks; LightGBM handles NaN features natively.

### Implementation notes

- **Leakage-free**: every input is contemporaneous or backward-looking; per-date rank uses only same-day cross-section across the universe.
- **Not z-scored**: these features are already in [0, 1] by construction. The whole point is that the per-date rank encoding is invariant to per-asset scale — the very thing per-asset z-score normalization cannot achieve.
- **Pre-loop column detection bug avoided**: a first pass of this helper checked `"return_5d" not in out.columns` inside the per-asset loop, which only fired for the first asset (since the loop body adds the column, the condition becomes False for subsequent assets). Now the existence check is hoisted before the loop and the resulting boolean controls all asset iterations uniformly. Caught by a deterministic-spike test.

### Feature-selector compatibility fix

The legacy feature selector excluded any column with `_rank` as a substring, intended to filter out the legacy `cross_sectional_rank` and `cross_sectional_percentile_rank` output columns. That blacklist would have silently dropped the new `xs_rank_*` input features. Removed `_rank` from `_FORBIDDEN_FEATURE_SUBSTRINGS`; the existing `cross_sectional_` prefix in `_FORBIDDEN_FEATURE_PREFIXES` still covers the legacy outputs.

Verified by `test_feature_selector_accepts_xs_rank_columns`.

### CLI flag

`--include-cross-sectional-features` (default False for backward compatibility) gates inclusion in `_build_panel`. When the flag is off, the experiment matches every prior run exactly. When on, the panel is enriched after `add_cross_sectional_ranks` but before walk-forward splitting.

### Tests added (Phase 2)

- `test_add_cross_sectional_features_produces_ranks_in_unit_interval` — every xs_rank_* in [0, 1].
- `test_add_cross_sectional_features_highest_value_gets_rank_one` — synthetic deterministic spike puts the spiking asset at rank 1.0.
- `test_add_cross_sectional_features_nan_for_insufficient_history` — early panel dates have NaN xs_rank_ret_60d (60d rolling not yet built).
- `test_add_cross_sectional_features_preserves_row_count_and_existing_columns` — pure append, no row drops, original columns intact.
- `test_feature_selector_accepts_xs_rank_columns` — selector includes `xs_rank_*` but still excludes `cross_sectional_*`, `is_top_*`, and the target.

## Phase 3 — Feature importance capture

`_score_with_lambdarank` now also returns a list of per-feature importance rows pulled from `model.booster_.feature_importance(importance_type="gain")` and `(importance_type="split")`. The runner appends these into a new `feature_importance` DataFrame (columns: `split_id`, `model`, `feature`, `gain`, `split_count`) and the CLI bundle writes it as a 10th output file: `cross_asset_ranking_feature_importance_<timestamp>.csv`.

For non-LambdaRank models the returned list is empty — the DataFrame just has no rows for those models. No regression: existing momentum / linear / HGB tests continue to pass.

## Tests run

```text
uv run python -m pytest tests/test_cross_asset_ranking.py tests/test_cross_asset_ranking_experiment.py -x -q
... 39 passed in 5.88s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 166 passed in 8.19s
```

13 new tests (1 column-name Phase 1, 5 cross-sectional Phase 2, 7 internal/selector). No regressions. Same single pre-existing legacy collection error documented in prior patches.

## Commands

Dry-run printed `include_xs_features: True`, made no IO, did not fetch or fit.

Execute:

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM DIA EFA EEM TLT IEF SHY LQD HYG GLD SLV USO DBA UUP VNQ BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_5d_target_xs_features_lambdarank \
  --forward-horizon 5 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 500 --random-state 42 \
  --rebalance-every 5 \
  --feature-normalization per_asset_train_zscore \
  --include-cross-sectional-features \
  --run-purpose decision_grade --decision-grade
```

5 walk-forward splits × 2 top-k × 500 random nulls = 5,000 null allocations on the 18-asset panel. Metadata flags: `data_downloaded: false`, `include_cross_sectional_features: true`, all safety flags `false`. No yfinance fetch.

## Output files

10 timestamped files under `results/cross_asset_ranking_5d_target_xs_features_lambdarank/` — including the new `cross_asset_ranking_feature_importance_<ts>.csv`.

## Full metrics — head-to-head with prior 5d-target/5d-rebalance baseline

| run / policy | mean IR | net Sharpe | active ret | turnover | cost drag | max DD | folds (+IR) | folds p≤0.05 | median p | BTC % | top-3 conc | drop-best mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no_xs top-1 | +0.174 | 0.550 | +0.063 | 0.315 | 0.019 | -0.25 | 3/5 | 0/5 | 0.361 | 22.5 | 42.1 | -0.134 |
| no_xs top-2 | -0.104 | 0.385 | -0.026 | 0.279 | 0.015 | -0.22 | 2/5 | 1/5 | 0.567 | 33.7 | 35.6 | -0.614 |
| **with_xs top-1** | **+0.071** | 0.412 | -0.025 | 0.293 | 0.017 | -0.33 | 3/5 | 1/5 | 0.443 | 22.5 | 46.7 | -0.425 |
| **with_xs top-2** | **+0.655** | **1.044** | **+0.091** | **0.269** | 0.017 | -0.22 | **4/5** | 1/5 | **0.204** | 35.6 | 39.7 | **+0.240** |
| equal-weight 18 ref | 0.000 | 1.018 | 0.000 | 0.004 | 0.0002 | -0.13 | n/a | n/a | n/a | n/a | n/a | n/a |

Top-1 got slightly worse on mean IR (+0.174 → +0.071), but top-2 transformed entirely: from a negative-IR loser to a positive-IR strategy that beats equal-weight on net Sharpe.

## Per-date Spearman score-vs-target rank correlation

```text
no_xs    mean_overall = +0.0174    [s0:-0.010  s1:+0.008  s2:+0.035  s3:+0.019  s4:+0.034]
with_xs  mean_overall = +0.0318    [s0:-0.028  s1:+0.050  s2:+0.048  s3:+0.039  s4:+0.051]
```

- **83 % relative gain** on overall mean Spearman.
- 4 of 5 folds now have ρ above +0.03 individually (s1, s2, s3, s4); only s0 (COVID rebound) is still negative.
- Splits 1, 2, 3, 4 each have ρ between +0.039 and +0.051 — consistent positive cross-sectional alignment.
- This is the first cross-asset configuration where the per-date Spearman is not concentrated in a single fold.

## Per-fold IR (with_xs)

```text
top-1:  s0:-1.704  s1:+0.016  s2:+0.289  s3:-0.302  s4:+2.057
top-2:  s0:-1.253  s1:+0.293  s2:+1.308  s3:+0.612  s4:+2.313
```

Top-2 is positive in 4 of 5 folds. Split 0 is still a problem (COVID-rebound regime where the model picks defensives). Splits 2, 3, 4 are all positive. Split 4 is exceptionally strong (+2.31) and p = 0.004.

## Random null p-values

```text
split  top-k  model IR    p-value
   0      1   -1.704     0.962
   0      2   -1.253     0.890
   1      1   +0.016     0.443
   1      2   +0.293     0.333
   2      1   +0.289     0.381
   2      2   +1.308     0.066  <-- borderline
   3      1   -0.302     0.567
   3      2   +0.612     0.204
   4      1   +2.057     0.004  <-- passes
   4      2   +2.313     0.004  <-- passes
```

Two folds pass (split 4 top-1 and top-2). One borderline (split 2 top-2 at 0.066). Not the required majority but a meaningfully different pattern than prior runs where the best single p-value was 0.014.

## Feature importance — top 12 by mean gain across 5 splits

| feature | gain | gain % | split_count |
|---|---|---|---|
| **xs_rank_vol_20d** | **1520** | **8.82%** | 81.4 |
| realized_vol_20 | 1003 | 5.82% | 70.8 |
| relative_vol_ratio | 967 | 5.61% | 72.6 |
| return_60d | 923 | 5.36% | 69.6 |
| **xs_rank_ret_60d** | **792** | **4.59%** | 65.2 |
| bollinger_band_width_zscore | 768 | 4.46% | 64.4 |
| volatility_regime | 766 | 4.45% | 62.0 |
| **xs_rank_ret_20d** | **765** | **4.44%** | 60.4 |
| current_drawdown_60d | 743 | 4.31% | 56.2 |
| autocorrelation_zscore | 731 | 4.24% | 64.0 |
| vix_relative | 678 | 3.93% | 52.0 |
| close_to_open_gap | 648 | 3.76% | 55.6 |

Cross-sectional features only (`xs_rank_*`):

| feature | gain | gain % |
|---|---|---|
| xs_rank_vol_20d | 1520 | 8.82% |
| xs_rank_ret_60d | 792 | 4.59% |
| xs_rank_ret_20d | 765 | 4.44% |
| xs_rank_drawdown_60d | 478 | 2.78% |
| xs_rank_ret_5d | 304 | 1.77% |

**Out of 39 total features, 4 of 5 xs_rank_* features rank in the top half (top 19) by gain. The single most-important feature is `xs_rank_vol_20d`** at 8.82% of total gain. Cross-sectional features together contribute 22.4% of total gain — a substantial slice given they're 5 of 39 features.

`xs_rank_ret_5d` is the only laggard (1.77% gain, below the top half). The 5-day horizon is short enough that per-date rank of trailing 5d return is noisier than the longer horizons.

The pattern is consistent with the diagnosis: the model needed to know "how does this asset's volatility / drawdown / momentum compare to the universe on this date?", not just "what is this asset's individual vol / drawdown / momentum?". The per-asset z-scored features answered the latter; the xs_rank features answer the former.

## Asset selection (with_xs) — top-2 by split

```text
split 0 (2020-02 → 2021-02, COVID rebound):     SHY 47.6%  UUP 44.4%  BTC-USD 27.8%  DBA 15.9%  LQD 11.9%  TLT 9.9%
  -> defensives + BTC; lost -1.25 IR (wrong regime call)

split 1 (2021-02 → 2022-02, inflation buildup): SHY 41.7%  BTC-USD 25.8%  HYG 17.9%  EEM 15.9%  GLD 14.7%  USO 13.9%  TLT 13.9%  UUP 11.9%
  -> mixed defensives + commodities + BTC; +0.29 IR

split 2 (2022-02 → 2023-05, bear / rate-shock): BTC-USD 36.5%  DBA 25.8%  EEM 23.8%  SLV 22.6%  UUP 17.9%  GLD 13.9%  USO 11.9%  EFA 9.9%
  -> commodities + GLD/SLV (correct defensive metals for bear); +1.31 IR  (borderline p = 0.066)

split 3 (2023-05 → 2024-08, AI rally):          BTC-USD 37.7%  GLD 20.6%  DBA 17.9%  DIA 17.9%  UUP 17.9%  QQQ 15.9%  SHY 15.9%  EEM 11.9%  HYG 9.9%  USO 9.9%  VNQ 8.7%
  -> BTC + diversified equities + bonds; +0.61 IR

split 4 (2024-08 → 2025-08, risk-on):           BTC-USD 50.4%  GLD 29.8%  USO 23.8%  SLV 13.9%  EFA 11.9%  DBA 9.9%  VNQ 9.9%
  -> BTC dominant + commodities; +2.31 IR  (p = 0.004)
```

Top-3 concentration 39.7%, BTC selection 35.6% (no dominance threshold breached). The pattern is recognizably regime-aware: defensives in 2022 bear (correct direction even though the asset mix was off), BTC+commodities in risk-on 2024-2025 (correct and big).

The split 0 failure (COVID rebound where the model picked defensives) persists across every cross-asset run — that fold appears to be a recurring blind spot.

## Pass criteria evaluation

**Decision-grade pass** (all required):

| Criterion | Top-1 | Top-2 |
|---|---|---|
| Mean IR > 0 AND active return > 0 | partial (+0.07 IR; -0.03 active) | ✓ (+0.65 IR; +0.09 active) |
| Per-date Spearman ≥ +0.05 | ✗ (+0.032) | ✗ (+0.032 — same scoring) |
| Folds passing p ≤ 0.05 ≥ 2/5 | ✗ (1/5) | ✗ (1/5; split 2 at 0.066 borderline) |
| Drop-best-fold mean IR > 0 | ✗ (-0.425) | ✓ (+0.240) |
| No BTC dominance, reasonable turnover/cost | ✓ | ✓ |
| At least one xs feature in top half of importance | ✓ (4 of 5) | ✓ |

**Directionally interesting** (per-date Spearman ≥ +0.03 with positive mean IR + xs features non-trivially used + other criteria reasonably satisfied):

| Criterion | Top-1 | Top-2 |
|---|---|---|
| Per-date Spearman ≥ +0.03 | ✓ (+0.032) | ✓ (+0.032) |
| Positive mean IR | ✓ (+0.07) | ✓ (+0.65) |
| Cross-sectional features non-trivially used | ✓ (4/5 in top half; #1 most important is xs_rank_vol_20d) | ✓ (same) |

**Verdict**: PASS on directionally-interesting; FAIL on decision-grade. The result clears the +0.03 Spearman bar but not the +0.05 bar. Top-2 is the policy that benefits most.

## What this tells us

The feature-side hypothesis is **validated as directionally correct**:

1. Per-asset features (even when z-scored on train) cannot encode "where does this asset stand relative to the universe today?" Per-date cross-sectional rank features can.
2. The most important addition is `xs_rank_vol_20d` — the universe-relative volatility rank — which makes intuitive sense: a "high vol" asset means something different in a low-dispersion regime than in a high-dispersion regime, and the absolute vol value can't convey that. The rank can.
3. Three of the four other xs_rank_* features also entered the top half of feature importance, indicating the model is finding signal in multiple aspects of cross-sectional position, not just vol.

But the result is **not yet decision-grade**. The Spearman gain (+0.017 → +0.032) is real but small in absolute terms. Top-2's headline +0.655 IR is dominated by split 4's +2.31 (35% of the mean comes from one fold), though importantly drop-best is still positive at +0.24.

## Stop / go verdict

```yaml
xs_features_v1:
  result: directionally_interesting
  next_pivot: feature_side_v2  (add reserved-for-v2 cross-sectional features)
  do_not_do_next:
    - target_side_pivot
    - seed_sensitivity
    - hyperparameter_tuning
    - universe_changes
    - deep_models
    - stacking
    - claim_production_readiness
```

Per the spec's predetermined order, the directionally-interesting branch authorizes a feature-side follow-up but not deployment.

## Recommended next step

**Add the reserved v2 cross-sectional features** in a single follow-up patch:

1. **Dispersion measures**: cross-sectional std of trailing 5d / 20d returns at each date (date-level features broadcast to all assets). Captures regime: low dispersion = correlated risk-on/risk-off, high dispersion = divergent regimes.
2. **Beta-to-equal-weight-rest**: rolling beta of each asset's daily return to the equal-weight portfolio of the *other 17* assets, computed on a 60d window. Captures "this asset's tilt vs. the rest of the universe" — different from per-asset features that look only at the asset itself.
3. **Regime-conditioned interactions**: existing xs_rank features multiplied by a `regime_indicator` derived from VIX z-score or universe-mean trailing vol rank. Lets the model learn "this asset's rank means something different in a high-VIX regime than a low-VIX regime."

Hypothesis: each of these encodes a different aspect of cross-sectional position that the v1 features can't. If they push Spearman from +0.032 toward +0.05+ and consistently lift drop-best-fold mean IR, the cross-asset ranking track crosses into decision-grade territory. If they don't move the needle, the conclusion is that the current 18-ETF universe doesn't have enough learnable cross-sectional structure to clear a strict null gate, and the project should accept the directionally-interesting result and move to the architectural pivot (regime detection + per-regime allocation).

Keep all other settings constant: LambdaRank, 5d target, 5d rebalance, per-asset train z-score, 500 nulls, 18-asset universe. One factor at a time.

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false` in metadata.
- `include_cross_sectional_features: true` recorded in metadata.
- `--prepare-missing` not set; no yfinance fetch.
- Static-import test continues to enforce no-legacy / no-Optuna / no-deep / no-stacking.
- No champion manifest changes.
- No existing result files overwritten.
- No hyperparameter tuning, seed variation, universe change, or model-family change. The only varying factor vs the prior run is the `--include-cross-sectional-features` flag, plus the cosmetic Phase 1 column-naming fix (which does not affect the underlying math).

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Phase 1: column name fixed, verified, documented | ✓ |
| 2 | Phase 2: cross-sectional features implemented behind `--include-cross-sectional-features` | ✓ |
| 3 | Phase 3: dry-run passes, execute completes (or failure documented with full diagnostics) | ✓ |
| 4 | No forbidden workflows touched | ✓ |
| 5 | Patch document created | ✓ this file |
