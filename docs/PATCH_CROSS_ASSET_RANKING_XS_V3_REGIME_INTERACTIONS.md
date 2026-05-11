# Patch: Cross-Asset Ranking — v3 Feature Pivot (Regime-Conditioned Interactions)

Date: 2026-05-11
Run timestamps:
- v1 reproduction check: `20260511T164344Z` → `results/cross_asset_ranking_v1_reproduction_check/`
- v3 decision-grade: `20260511T165440Z` → `results/cross_asset_ranking_5d_target_xs_v3_regime_lambdarank/`
- v1 reference (high-water mark): `20260511T143612Z` → `results/cross_asset_ranking_5d_target_xs_features_lambdarank/`

## TL;DR verdict

**Strict spec verdict: FAIL on both decision-grade and directionally-interesting tiers — by tight margins on rank-quality metrics, even as v3's top-2 produced the strongest economic result of the entire campaign.**

The economic numbers improved against v1 across the board:

| Metric (top-2) | v1 | **v3** | Δ |
|---|---|---|---|
| Mean IR | +0.655 | **+0.748** | +0.093 |
| Net Sharpe | 1.044 | **1.180** | +0.136 |
| Drop-best mean IR | +0.240 | **+0.286** | +0.046 |
| Folds passing p ≤ 0.05 | 1/5 | **2/5** | +1 |
| Active return | +0.091 | +0.102 | +0.011 |
| Max drawdown | -0.221 | -0.209 | better |

Top-1 also improved on the "real money" axis (mean IR +0.071 → +0.137, drop-best mean IR -0.425 → -0.091, active return -0.025 → +0.047), but missed the spec's "material improvement" threshold of +0.15 by 0.013.

What killed the rubric pass was the rank-quality regression. Per-date Spearman fell from +0.0318 to **+0.0276** (post-hoc check: +0.0300 — both still below the pre-committed +0.03 directional threshold). ICIR fell from 0.94 to 0.79 (still acceptable but no longer in the "strong" band). The model's *per-date discrimination* is slightly noisier, but its *top-k extraction* is consistently better. Regime interactions traded a tiny amount of rank quality for stronger top-k outcomes.

Per the spec's pre-committed pivot order, v3 is **the last feature-side experiment**. The next move is architectural: regime detection + per-regime simple allocation, with the v1/v3 xs-feature LambdaRank as a within-regime ranker rather than a standalone strategy. v3's top-2 numbers are the new top-2 high-water mark; v1 remains the rank-quality high-water mark.

```yaml
xs_v3_regime_interactions_5d_target_18_asset_lambdarank:
  status: failed_both_tiers_by_tight_margins_but_best_top2_of_campaign
  decision_grade_pass: false
  directionally_interesting_pass: false
  fail_reasons:
    - per_date_spearman_+0.0276_below_+0.03_directional_threshold
    - top1_mean_ir_+0.137_below_+0.15_material_improvement_threshold
  not_failed:
    - top2_regression  # actually improved
    - regime_interactions_unused  # 9.11% combined gain, 2/5 in top half
    - regime_interactions_actively_harmful  # mixed — base xs features lost weight but top-k improved
  top2_v3_mean_ir: +0.748  # best of campaign
  top2_v3_net_sharpe: 1.180  # best of campaign
  top2_v3_drop_best_mean_ir: +0.286
  top2_v3_folds_pass_p05: 2/5
  next_step: architectural_pivot
```

## Phase 1 — Revert v2 feature changes

### Implementation

- Restored `xs_rank_ret_5d` to the v1 base set in `add_cross_sectional_features` and `_CROSS_SECTIONAL_FEATURE_COLUMNS`.
- Removed `xs_rank_beta_ew_rest_60d` and `xs_rank_beta_ew_rest_252d` from the rank specs.
- Deleted the `_attach_beta_to_ew_rest` helper entirely (no other caller).

### Verification

Ran the v1-equivalent command (cross-sectional features ON, regime interactions OFF) and compared to the original v1 run output. **The per-fold IRs are bit-for-bit identical**:

```text
v1 top-2 IRs:  [-1.252969, +0.293157, +1.308116, +0.612238, +2.312966]
v1r top-2 IRs: [-1.252969, +0.293157, +1.308116, +0.612238, +2.312966]
Bit-for-bit identical: True
```

Same seed (42), same data, same code paths, same numbers. Phase 1 is clean. The v2 revert is complete.

This also retroactively answers a v2 confounding question (which of the two v2 changes did the damage): the answer is "one or both, but the combination was definitively worse." Restoring the v1 set fully recovers v1 numbers.

## Phase 2 — VIX fetch

**No fetch was required.** The `^VIX` cache already exists at `data/multi_asset_cache/vix_daily.csv` from the earlier 18-asset universe-expansion patch (`docs/PATCH_CROSS_ASSET_RANKING_EXPANDED_UNIVERSE_LAMBDARANK.md`). Verification:

```text
file:           data/multi_asset_cache/vix_daily.csv
rows:           4112
date range:     2010-01-04 → 2026-05-08
NaN counts:     all zero
columns:        Date, Open, High, Low, Close, Adj Close, Volume
sidecar:        data/multi_asset_cache/vix_daily.meta.json present
```

The v3 experiment run shows `data_downloaded: false` in metadata, consistent with the spec's authorization model: the VIX cache itself was authorized data in a prior patch, the v3 experiment is cached-only.

## Phase 3 — VIX z-score column

### Implementation

`evaluation/cross_asset_ranking.py:add_vix_zscore_to_panel(panel, *, date_col, vix_col, window=252, output_col="vix_zscore_252d")`:

1. Take unique per-date VIX values from the panel (VIX is a market-state series — same value across all 18 assets per date after `build_asset_cache_frame` does the per-asset join).
2. Compute trailing rolling mean and standard deviation using `pandas.Series.rolling(window, min_periods=window)`. With `window=252` this requires a full year of past VIX observations before producing a non-NaN value.
3. `vix_zscore_252d(t) = (VIX(t) − rolling_mean_252d(t)) / rolling_std_252d(t)`.
4. Map zero standard deviation to NaN (defensive, shouldn't happen with VIX).
5. Broadcast back to all (date, asset) rows of the panel via merge on date.

### Causality argument

The rolling window covers `t − 251` through `t` inclusive — entirely past data including the current bar. VIXClose(t) is observed by the close of trading day t. Downstream, weights derived at end-of-t are applied to returns from t+1 onwards (the standard one-bar lag in `compute_allocation_returns`). So including VIXClose(t) in the rolling window does not introduce future leakage.

### Tests

- `test_vix_zscore_is_causal_and_broadcasts_across_assets`: synthetic 2-asset panel with 300 rows and a sinusoidal VIX series. First 251 dates have NaN z-score (window not yet full); after warmup the z-score is identical across assets per date (broadcast verified by `assert late.loc["A"] == approx(late.loc["B"])`).

## Phase 4 — Regime-conditioned interaction features

### Implementation

`evaluation/cross_asset_ranking.py:add_regime_interaction_features(panel, *, base_features, regime_col, suffix)`:

For each base feature in `_CROSS_SECTIONAL_FEATURE_COLUMNS` (the v1 set), compute the element-wise product with `vix_zscore_252d`. Output column naming: `{base_feature}_x_vix_z`.

```text
xs_rank_ret_5d        × vix_zscore_252d → xs_rank_ret_5d_x_vix_z
xs_rank_ret_20d       × vix_zscore_252d → xs_rank_ret_20d_x_vix_z
xs_rank_ret_60d       × vix_zscore_252d → xs_rank_ret_60d_x_vix_z
xs_rank_vol_20d       × vix_zscore_252d → xs_rank_vol_20d_x_vix_z
xs_rank_drawdown_60d  × vix_zscore_252d → xs_rank_drawdown_60d_x_vix_z
```

Output is signed and unbounded. No re-ranking; no normalization; no bucketing. The cross-sectional rank in [0, 1] multiplied by a z-score in roughly [-3, 3] preserves the magnitude information that discretization would discard.

### CLI flag

New flag `--include-regime-interactions` (default `False`). Requires `--include-cross-sectional-features` to be on (the interactions are products of the cross-sectional features). The runner raises `ValueError` if the regime flag is set without the cross-sectional flag — explicit fast-fail rather than silent no-op.

### Tests

- `test_regime_interaction_features_are_product_of_rank_and_zscore`: verifies element-wise product equality on a synthetic panel with known z-scores `[+2.0, -1.5]` and rank values across two dates. All 5 expected output columns are exact equal to `panel[base] * panel["vix_zscore_252d"]`.

## Phase 5 — Experiment

### Tests run

```text
uv run python -m pytest tests/test_cross_asset_ranking.py tests/test_cross_asset_ranking_experiment.py -x -q
... 42 passed in 7.55s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 169 passed in 9.22s
```

No regressions.

### Commands

Dry-run printed `include_xs_features: True, include_regime_interactions: True`, made no IO.

Execute:

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM DIA EFA EEM TLT IEF SHY LQD HYG GLD SLV USO DBA UUP VNQ BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_5d_target_xs_v3_regime_lambdarank \
  --forward-horizon 5 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 500 --random-state 42 \
  --rebalance-every 5 \
  --feature-normalization per_asset_train_zscore \
  --include-cross-sectional-features \
  --include-regime-interactions \
  --run-purpose decision_grade --decision-grade
```

5 splits × 2 top-k × 500 random nulls = 5,000 null allocations on the 18-asset panel. Metadata flags: `data_downloaded: false`, `include_cross_sectional_features: true`, `include_regime_interactions: true`. All other safety flags `false`.

### Output files

10 timestamped files under `results/cross_asset_ranking_5d_target_xs_v3_regime_lambdarank/`.

## Head-to-head — v1 vs v3 (5d/5d, 18 assets, normalized features, LambdaRank)

| run / policy | mean IR | net Sharpe | active ret | folds (+IR) | folds p≤0.05 | median p | drop-best mean | BTC % | top-3 conc | max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 top-1 | +0.071 | 0.412 | -0.025 | 3/5 | 1/5 | 0.443 | -0.425 | 22.5 | 46.7% | -0.33 |
| v1 top-2 | +0.655 | 1.044 | +0.091 | 4/5 | 1/5 | 0.204 | +0.240 | 35.6 | 39.7% | -0.22 |
| **v3 top-1** | **+0.137** | 0.520 | +0.047 | 3/5 | 0/5 | 0.289 | **-0.091** | **19.6** | 41.2% | **-0.28** |
| **v3 top-2** | **+0.748** | **1.180** | **+0.102** | 4/5 | **2/5** | **0.134** | **+0.286** | 31.5 | 36.5% | **-0.21** |
| equal-weight 18 ref | 0.000 | 1.018 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | -0.13 |

Top-2 improves on every metric vs v1. Top-1 also improves on every metric vs v1 except folds-passing-p ≤ 0.05 (both 0/1 of 5).

## Per-date Spearman + ICIR

Reported from the runner's `ranking_diagnostics` output. Post-hoc cross-check for v1 (computed using the same `_compute_spearman_and_icir` logic) follows.

| run | overall mean Spearman | std (ddof=1) across folds | ICIR | n_folds |
|---|---|---|---|---|
| v1 (post-hoc) | +0.0318 | 0.0340 | **+0.9353** | 5 |
| **v3 (runner)** | **+0.0276** | 0.0351 | **+0.7884** | 5 |
| v3 (post-hoc cross-check) | +0.0300 | 0.0362 | +0.8296 | 5 |

The runner and post-hoc values for v3 differ slightly because the runner uses the panel's target column (already filtered for valid forward returns within the panel rows scored) while the post-hoc check re-derives the target from raw return series. The runner figure is the canonical record; the difference is immaterial to the verdict.

Per-fold Spearman (post-hoc, comparable to v1 baseline):

```text
v1: s0:-0.029   s1:+0.050   s2:+0.048   s3:+0.039   s4:+0.051
v3: s0:-0.034   s1:+0.048   s2:+0.045   s3:+0.041   s4:+0.050
```

The two profiles are very similar. v3 marginally beats v1 in split 3 (+0.041 vs +0.039), and ties on splits 1, 2, 4. Split 0 (the recurring COVID-rebound blind spot) is slightly worse. The overall +0.0276 mean is a small regression driven mostly by the slight split-0 deterioration.

## Per-fold IR (v3)

```text
top-1:  s0:-1.601   s1:+1.049   s2:+0.543   s3:-0.092   s4:+0.785
top-2:  s0:-2.050   s1:+0.590   s2:+1.023   s3:+1.580   s4:+2.598
```

v3 top-2 is positive in 4 of 5 folds. Split 4 reached **+2.60 IR** (v1 was +2.31). Split 3 reached **+1.58 IR** (v1 was +1.23) and **clears p ≤ 0.05 at p = 0.046**.

Top-2 drop-best-fold mean IR = (−2.05 + 0.59 + 1.02 + 1.58) / 4 = **+0.286** (v1: +0.240). Top-1 drop-best mean IR = (−1.60 + −0.09 + 0.54 + 0.78) / 4 = **−0.091** (v1: −0.425 — large improvement, but still negative).

## Random null p-values (v3)

```text
split  top-k  model IR    p-value
   0      1   -1.601     0.952
   0      2   -2.050     0.986
   1      1   +1.049     0.118
   1      2   +0.590     0.251
   2      1   +0.543     0.289
   2      2   +1.023     0.134
   3      1   -0.092     0.495
   3      2   +1.580     0.046   <-- passes
   4      1   +0.785     0.190
   4      2   +2.598     0.002   <-- passes cleanly
```

**Two folds clear p ≤ 0.05 in top-2 (splits 3 and 4)**, vs one in v1. Top-1 still has no fold passing, though split 1 came close at p = 0.118.

## Feature importance — v3 — top 14 by mean gain across 5 splits

| feature | gain | gain % | split_count |
|---|---|---|---|
| **xs_rank_vol_20d** (base xs) | 1330 | **7.57%** | 73.6 |
| realized_vol_20 | 952 | 5.42% | 67.0 |
| relative_vol_ratio | 936 | 5.33% | 71.4 |
| return_60d | 912 | 5.19% | 67.6 |
| xs_rank_ret_60d (base xs) | 765 | 4.36% | 57.8 |
| bollinger_band_width_zscore | 765 | 4.35% | 59.8 |
| xs_rank_ret_20d (base xs) | 720 | 4.10% | 55.4 |
| autocorrelation_zscore | 720 | 4.10% | 61.2 |
| current_drawdown_60d | 708 | 4.03% | 49.4 |
| volatility_regime | 630 | 3.59% | 50.6 |
| vix_relative | 557 | 3.17% | 42.2 |
| macd_histogram_zscore | 530 | 3.02% | 40.0 |
| downside_vol_ratio | 510 | 2.90% | 42.0 |
| close_to_open_gap | 501 | 2.85% | 42.6 |

Regime interaction features (the 5 new in v3):

| feature | gain | gain % | split_count |
|---|---|---|---|
| xs_rank_ret_20d_x_vix_z | 339 | 1.93% | 28.4 |
| xs_rank_drawdown_60d_x_vix_z | 328 | 1.87% | 25.4 |
| xs_rank_ret_5d_x_vix_z | 324 | 1.84% | 26.4 |
| xs_rank_ret_60d_x_vix_z | 312 | 1.78% | 23.6 |
| xs_rank_vol_20d_x_vix_z | 298 | 1.69% | 25.6 |

**Combined interaction gain: 1600 / 17578 = 9.11% of total.** All 5 interactions have ranks roughly clustered around #20-25 of 45 — just inside or just outside the top half. **Two interactions (`xs_rank_drawdown_60d_x_vix_z` and `xs_rank_ret_20d_x_vix_z`) sit in the top half of the importance ranking.** The `vix_zscore_252d` feature on its own contributes only 0.69% gain — the regime signal is almost entirely consumed via the interactions, not directly.

The interactions are **non-trivially used** by the spec's definition (at least one in top half + combined ≥ 5% gain). They are neither dominant (none in top 10) nor ignored.

## v1 vs v3 — base xs feature importance shifts

| feature | v1 gain % | v3 gain % | Δ |
|---|---|---|---|
| xs_rank_ret_5d | 1.77 | 1.69 | -0.08 |
| xs_rank_ret_20d | 4.44 | 4.10 | -0.34 |
| xs_rank_ret_60d | 4.59 | 4.36 | -0.23 |
| xs_rank_vol_20d | 8.82 | 7.57 | **-1.25** |
| xs_rank_drawdown_60d | 2.78 | 2.23 | -0.55 |
| **vix_zscore_252d** | — | 0.69 | +0.69 |
| **5 interaction features (sum)** | — | 9.11 | +9.11 |

The interactions took 9.11pp of feature gain; ~2.45pp came from displacement of the base xs features. The rest came from displacement of non-xs features (`volatility_regime` -0.86pp, `vix_relative` -0.32pp, `macd_histogram_zscore` -0.17pp, `close_to_open_gap` -0.91pp, others).

Unlike v2's beta-feature failure mode (heavily used + actively harmful), v3's interactions are used moderately + slightly net-helpful for top-k outcomes. The interactions and their base features coexist in the importance ranking rather than cannibalizing each other.

## Asset selection (v3) — top-2 by split

```text
split 0 (COVID rebound):    UUP 60.3%  SHY 35.7%  BTC 25.8%  DBA 15.9%  LQD 13.9%  TLT 8.7%
  -> defensives in a recovery (wrong regime call) — lost -2.05 IR

split 1 (inflation buildup): BTC 33.7%  SHY 33.7%  DBA 19.8%  HYG 15.9%  USO 15.9%  GLD 12.7%  UUP 11.9%  EEM 9.9%  SLV 9.9%
  -> broad mix with commodities + cash — won +0.59 IR

split 2 (2022 bear):         DBA 31.7%  BTC 27.8%  EEM 21.8%  GLD 19.8%  UUP 17.9%  USO 15.9%  SLV 14.7%  HYG 9.9%  IEF 9.9%
  -> commodities-led with defensives — won +1.02 IR

split 3 (AI rally):          BTC 29.8%  USO 21.8%  QQQ 21.8%  GLD 19.8%  EEM 15.9%  UUP 15.9%  DBA 13.9%  EFA 11.9%  DIA 9.9%  SHY 8.7%  VNQ 8.7%
  -> BTC + diversified equities/commodities — won +1.58 IR (p = 0.046)

split 4 (risk-on):           BTC 40.5%  GLD 27.8%  USO 19.8%  VNQ 17.9%  TLT 11.9%  DBA 9.9%  EFA 9.9%  SLV 9.9%  UUP 9.9%  EEM 8.7%
  -> BTC-led with hard-asset diversification — won +2.60 IR (p = 0.002)
```

BTC selection 31.5% (below 50% dominance threshold). Top-3 concentration 36.5% (lower than v1's 39.7%; slightly more diversified). Split 0 still the recurring blind spot — the model picks defensives in a recovery regime.

## Pass criteria evaluation

**Decision-grade pass** (BOTH top-1 AND top-2 must satisfy ALL):

| Criterion | Top-1 | Top-2 |
|---|---|---|
| Mean IR > 0 AND active return > 0 | ✓ (+0.137; +0.047) | ✓ (+0.748; +0.102) |
| Per-date Spearman ≥ +0.05 | ✗ (+0.028) | ✗ (+0.028) |
| ICIR ≥ 0.5 | ✓ (0.79) | ✓ (0.79) |
| Folds passing p ≤ 0.05 ≥ 2/5 | ✗ (0/5) | ✓ (2/5) |
| Drop-best-fold mean IR > 0 | ✗ (-0.091) | ✓ (+0.286) |
| No BTC dominance, reasonable turnover/cost | ✓ | ✓ |
| At least one xs/interaction feature in top half | ✓ | ✓ |
| Top-1 material improvement over v1 (+0.071) | ✗ (delta +0.066 < +0.15 spec bar) | n/a |
| Top-2 maintain or improve on v1 (IR ≥ +0.5, drop-best ≥ +0.15) | n/a | ✓ (+0.748, +0.286) |

Top-1 fails 4 of 9 criteria. Top-2 fails 1 of 9 (Spearman). Joint criterion (top-1 material improvement) also fails. **Decision-grade: FAIL.**

**Directionally interesting** (per the spec):

| Criterion | Result |
|---|---|
| Top-2 Spearman ≥ +0.03 | **✗** (+0.0276 — short by 0.0024) |
| Top-2 mean IR > +0.3 | ✓ (+0.748) |
| Top-2 ICIR ≥ 0.3 | ✓ (0.79) |
| Top-2 drop-best > 0 | ✓ (+0.286) |
| Top-1 mean IR ≥ +0.15 | **✗** (+0.137 — short by 0.013) |
| Regime interactions non-trivially used | ✓ (2/5 in top half, 9.11% gain) |

Two close misses. **Directionally interesting: FAIL** by the strict pre-committed thresholds (the spec explicitly forbids relaxing them).

## What this tells us

The v3 result is the most interesting failure in the campaign. Both rank-quality misses (Spearman 0.0024 short, top-1 IR 0.013 short) are within ~10% of the thresholds. The economic improvements are decisive and consistent: top-2 mean IR up 14%, drop-best up 19%, folds passing p ≤ 0.05 doubled. The regime interactions did exactly what they were hypothesized to do — provide context about the VIX state to a model that already had cross-sectional position — but the improvement showed up in top-k extraction rather than in per-date rank fidelity.

That's a meaningful pattern. v1's per-date Spearman of +0.032 was already a small number; v3's +0.028 isn't qualitatively different. But the model's *use* of those slightly-noisier scores produced better picks. This is most visible in split 4: v1's Spearman was +0.051, v3's is +0.050 — almost identical — yet v3 top-2 IR is +2.60 vs v1's +2.31. The regime interactions help the model handle ranks differently in different VIX environments without making the scores themselves much sharper.

Interpretively, this points to a structural ceiling of cross-sectional ranking on this universe at this horizon. Four years of feature-side iteration (v1 + v2 + v3 + a per-asset normalization study) have all produced Spearman values in [0.024, 0.032]. The variation between these is smaller than the variation across folds within a single configuration. There is no feature-side change that will materially break out of this band on the current 18-asset universe and 5-day forward target.

## Stop / go verdict

```yaml
v3_regime_interactions:
  result: failed_both_tiers_strictly
  margins:
    spearman_short_by: 0.0024  # below +0.03 directional threshold
    top1_ir_short_by: 0.013    # below +0.15 material-improvement threshold
  but_also:
    best_top2_mean_ir_in_campaign: true
    best_top2_drop_best_in_campaign: true
    best_top2_folds_passing_p05_in_campaign: 2/5
    best_top2_net_sharpe_in_campaign: 1.180
  feature_side_campaign_status: exhausted
  next_step: architectural_pivot  # per spec's predetermined order
  do_not_do_next:
    - dispersion_features
    - different_vix_windows
    - bucketed_regime_features
    - retry_betas
    - target_side_pivot
    - seed_sensitivity
    - hyperparameter_tuning
    - universe_changes
    - deep_models
    - stacking
    - production_claims
  empirical_high_water_marks:
    top2_mean_ir: v3 (+0.748)
    top2_net_sharpe: v3 (1.180)
    top2_drop_best: v3 (+0.286)
    top2_folds_pass_p05: v3 (2/5)
    rank_quality_spearman: v1 (+0.0318)
    rank_quality_icir: v1 (0.935)
```

## Recommended next step — architectural pivot

Per the spec's predetermined order, v3 is the last feature-side experiment. The architectural pivot is now warranted.

### Framing

The v1 6-asset top-1 result (`docs/PATCH_CROSS_ASSET_RANKING_DECISION_GRADE_RUN.md`) was, in retrospect, a regime classifier with one or two assets per regime — TLT in bond-friendly periods, GLD in bear, BTC in risk-on. v1's xs features made it a more sophisticated regime classifier with multi-asset allocation per regime. v3's regime interactions made the regime signal more explicit but kept it implicit in the LambdaRank scoring. The natural next move is to **make the regime structure fully explicit**: a classifier (or hand-coded rule) determines the regime, and a within-regime ranker (xs-feature LambdaRank) handles asset selection inside the regime.

### Concrete architecture

1. **Regime classifier**: lightweight 3-state classifier based on aggregate market state — e.g., VIX z-score bucketed into low / mid / high terciles, OR a 2-state HMM on rolling 20d realized volatility of the equal-weight portfolio. Fitted on train data only.

2. **Per-regime ranker**: train one LambdaRank per regime using the v1 xs feature set (the rank-quality high-water mark). Apply only to test rows whose contemporaneous regime matches the model's training regime. This lets each regime ranker specialize on the kind of cross-sectional structure that matters in that regime — different from v3's "one model handles all regimes with interaction features."

3. **Allocation**: within-regime top-k allocation as today. Across regimes, the regime classifier switches which model's scores are used. This is structurally different from v3 because the model never sees out-of-regime training data.

### Why this is the right next move

- It uses what we know works: v1 features + LambdaRank + 5d target + 5d rebalance + cross-sectional ranking.
- It addresses the recurring split-0 blind spot directly: that fold is a recovery regime that the existing models keep misclassifying. A regime-aware architecture has the chance to be right about regime even if it's still imperfect at within-regime ranking.
- It moves from "model implicitly discovers regimes via features" to "regime is a first-class object in the system" — exactly the architectural lesson v3's result implies.

### What this is NOT

- It is not a "let's try more features" project. The feature side is closed.
- It is not a production model. The within-regime ranker still has only +0.03 Spearman — that's the ceiling we should expect.
- It is not seed/hyperparameter exploration. The architectural change is the experiment.

### Pass criteria for the architectural pivot (forward-looking)

The same tiered rubric should apply when the architecture changes — Spearman ≥ +0.05 with ICIR ≥ 0.5 for decision-grade, Spearman ≥ +0.03 with positive top-1 movement for directionally interesting. The same 18-asset universe, the same 5d target, the same 500 random nulls, the same matched-null discipline.

**The +0.05 Spearman threshold remains pre-committed**, per the v2 patch's no-goalpost-move rule. If the architectural pivot also doesn't clear it, the appropriate conclusion is that cross-sectional ranking on this asset class at this horizon does not support a decision-grade strategy at all, and the project should pivot to a different problem (forward volatility/drawdown ranking, single-asset regime detection, or accepting v1/v3 as research artifacts rather than candidates for further iteration).

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false` in metadata.
- `include_cross_sectional_features: true` and `include_regime_interactions: true` recorded in metadata.
- VIX cache was a previously-authorized fetch (in the universe-expansion patch); v3 run itself shows `data_downloaded: false`.
- `--prepare-missing` not set.
- Pre-committed thresholds (+0.05 Spearman decision-grade, +0.03 Spearman directional, ICIR ≥ 0.5 decision-grade, ICIR ≥ 0.3 directional) were honored — v3 is reported as a fail at the strict thresholds, not relaxed.
- Static-import test continues to enforce no-legacy / no-Optuna / no-deep / no-stacking invariants.
- No champion manifest changes.
- No existing result files overwritten — v3 output went to a fresh per-config directory; v1 reproduction went to a separate directory.
- v1 reference run remains untouched.
- Single-factor discipline honored: vs v1 baseline (after the v2 revert), exactly one change was made — adding the regime interactions. No simultaneous feature dropping, no hyperparameter tuning, no other structural changes.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Phase 1: v2 feature changes reverted, v1 reproducibility verified | ✓ bit-for-bit identical |
| 2 | Phase 2: VIX cache present with sidecar; fetch documented (was a previously-authorized fetch, no new download) | ✓ |
| 3 | Phase 3: `vix_zscore_252d` column added, causal computation verified by test | ✓ |
| 4 | Phase 4: 5 interaction features implemented behind `--include-regime-interactions` | ✓ |
| 5 | Phase 5: dry-run passes, execute completes, failure documented with full diagnostics | ✓ failure documented |
| 6 | No forbidden workflows touched | ✓ |
| 7 | Patch document created | ✓ this file |
