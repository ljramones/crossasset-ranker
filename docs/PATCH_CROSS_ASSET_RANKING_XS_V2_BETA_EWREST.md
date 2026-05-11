# Patch: Cross-Asset Ranking — v2 Feature Pivot (Beta-to-Equal-Weight-Rest) + ICIR Formalization

Date: 2026-05-11
Run timestamp: `20260511T152211Z` → `results/cross_asset_ranking_5d_target_xs_v2_lambdarank/`
Reference run (v1): `20260511T143612Z` → `results/cross_asset_ranking_5d_target_xs_features_lambdarank/`

## TL;DR verdict

**FAIL on both decision-grade and directionally-interesting tiers — the v2 feature swap regressed both top-1 and top-2 vs the v1 baseline.** Top-2 mean IR collapsed from **+0.655 → −0.040** (a 0.70 absolute drop). Top-1 mean IR fell from **+0.071 → −0.024**. Per-date Spearman regressed from **+0.0318 → +0.0242**. Zero folds clear p ≤ 0.05 in either top-k (vs 1/5 at v1). The two new beta features ARE heavily used by the model (combined gain 11.76% of total; `xs_rank_beta_ew_rest_252d` is the #2 feature overall) — they're not unused, they're actively *misleading*.

ICIR was formalized as a permanent rubric element this round. Retrospectively v1 had ICIR ≈ 0.94, v2 has ICIR ≈ 0.99. ICIR even ticked up slightly under v2 — but importantly, **a higher ICIR around a lower mean is stable noise, not improved signal**. That sharpens the diagnosis: v2 didn't fail by becoming erratic; it failed by becoming consistently worse-aligned with the realized target.

The result triggers the spec's pre-committed v3 branch: **regime-conditioned interactions** (xs features × VIX-derived regime indicator). This is the last feature-side option. If v3 also fails, the spec mandates an architectural pivot (regime detection + per-regime allocation with v1 features as within-regime ranker).

```yaml
xs_v2_beta_ew_rest_5d_target_18_asset_lambdarank:
  status: failed_both_tiers
  decision_grade_pass: false
  directionally_interesting_pass: false
  top1_v2_mean_ir: -0.024
  top1_v1_mean_ir: +0.071
  top1_delta: -0.095   # REGRESSION
  top2_v2_mean_ir: -0.040
  top2_v1_mean_ir: +0.655
  top2_delta: -0.695   # SEVERE REGRESSION
  per_date_spearman_v2: +0.0242
  per_date_spearman_v1: +0.0318
  icir_v2: +0.9932
  icir_v1: +0.9353
  beta_features_used: true  (combined 11.76% gain, #2 + #3 overall)
  conclusion: |
    Beta-to-EW-rest features supplant some of v1's working features in
    importance ranking but produce systematically worse cross-sectional
    rankings. Top-2 regressed from a directionally-interesting result to
    a clear loss.
```

## Phase 1 — Drop `xs_rank_ret_5d`

The v1 report flagged `xs_rank_ret_5d` as the only xs feature below the top half of importance (1.77% gain, 5th of 5 xs features). It was the natural candidate to drop.

Implementation:
- Removed from the `rank_specs` tuple in `add_cross_sectional_features`.
- Removed the `need_return_5d` / `return_5d` computation block since no other feature consumes `return_5d` (verified by grep across the codebase).
- Updated `_CROSS_SECTIONAL_FEATURE_COLUMNS` constant to drop the entry.

Verification:
- New test `test_add_cross_sectional_features_drops_xs_rank_ret_5d` asserts the column is absent.
- Existing v1 tests for the retained 4 xs features still pass.

## Phase 2 — Beta-to-equal-weight-rest features

### Implementation

`evaluation/cross_asset_ranking.py:_attach_beta_to_ew_rest(panel, *, date_col, asset_col, return_col, windows)`. Vectorized in wide format, then stacked back to long.

For each (asset i, window W) the function computes:

```text
EW_rest_i(t) = (sum of daily returns of all assets present at t — return_i(t)) / (n_present(t) − 1[i present])
```

then a rolling OLS beta over W trading days ending at t:

```text
beta_i,W(t) = Cov_W(r_i, EW_rest_i) / Var_W(EW_rest_i)
            = [E_W[r_i · EW_rest_i] − E_W[r_i]·E_W[EW_rest_i]]
              / [E_W[EW_rest_i²] − E_W[EW_rest_i]²]
```

with `pandas.rolling(window, min_periods=window)` — so partial windows yield NaN (no leakage from incomplete data). Negative-numerical-noise variance values are mapped to NaN before division.

Cross-sectional rank of the raw betas is computed per date in [0, 1] using the same `_per_date_normalized_rank` helper as v1 (highest raw beta → 1.0, lowest → 0.0). The raw beta columns are dropped from the panel after ranking; only `xs_rank_beta_ew_rest_60d` and `xs_rank_beta_ew_rest_252d` survive.

Two new features:

| feature | window | meaning |
|---|---|---|
| `xs_rank_beta_ew_rest_60d` | 60 trading days | responsive medium-term beta rank |
| `xs_rank_beta_ew_rest_252d` | 252 trading days | structural long-term beta rank |

Both gated by the existing `--include-cross-sectional-features` flag — no new CLI surface area.

### Tests added

- `test_beta_ew_rest_features_isolate_high_beta_asset`: synthetic 4-asset panel where asset A's returns are `3 × mean(B, C, D) + tiny noise`. After 60 bars asset A must land at the top of the cross-sectional beta-EW-rest rank on the final date. Passes.
- `test_beta_ew_rest_features_nan_during_warmup`: panel dates within the first 55 rows have NaN `xs_rank_beta_ew_rest_60d` (rolling window not yet full). Passes.

## Phase 3 — ICIR formalization

### Definition

```text
ICIR_model = mean(spearman_per_fold_model) / std(spearman_per_fold_model, ddof=1)
```

where `spearman_per_fold_model` is the per-fold mean of per-date Spearman rank correlations between model scores and the realized forward-horizon risk-adjusted return target. Sample-std denominator (ddof=1) over the 5 fold values.

### Computation location

`experiments/cross_asset_ranking_experiment.py:_compute_spearman_and_icir(...)` runs once at the end of the experiment loop, after every fold's scores have been collected. For each `(model, split_id)`:

1. Merge the scored panel slice with the realized target column from the panel on `(date, asset)`.
2. For each unique date in the slice, compute the rank-correlation `score.rank() · target.rank()` Pearson coefficient (this is Spearman by construction).
3. Average across dates within the fold → `mean_spearman` for that `(model, split_id)`.

Then for each model: aggregate the 5 fold means → `overall_mean_spearman`, `spearman_std_across_folds`, `icir`.

### Output locations

- `fold_details`: new column `per_fold_mean_spearman`.
- `summary`: three new columns broadcast across top_k rows for each model — `overall_mean_spearman`, `spearman_std_across_folds`, `icir`. Plus `n_folds` for cross-checking.
- New CSV file `cross_asset_ranking_ranking_diagnostics_<timestamp>.csv` — one row per model with the same four fields.

### Retrospective v1 calculation

Per-fold Spearmans for the v1 run (computed post-hoc from the saved scored panel):

```text
v1: split 0: -0.0285   split 1: +0.0496   split 2: +0.0476   split 3: +0.0392   split 4: +0.0511
    overall_mean = +0.0318
    std (ddof=1) = +0.0340
    ICIR        = +0.9353
```

V1's ICIR ≈ 0.94 is in the "acceptable" band of the formalized rubric (≥ 0.3 acceptable, ≥ 0.5 strong, ≥ 1.0 exceptional). That's consistent with the v1 verdict of "directionally interesting" — the signal was modest but stable.

### Threshold rationale

Industry quant practice for IC information ratios:

| ICIR | interpretation |
|---|---|
| ≥ 1.0 | exceptional |
| ≥ 0.5 | strong |
| ≥ 0.3 | acceptable for further investigation |
| < 0.3 | likely noise |

The decision-grade tier in the v2 pass criteria requires **ICIR ≥ 0.5** and **mean Spearman ≥ +0.05**. The directionally-interesting tier requires **ICIR ≥ 0.3** and **mean Spearman ≥ +0.03**.

## Phase 4 — Experiment

### Commands

Dry-run printed `include_xs_features: True`, made no IO. Execute:

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM DIA EFA EEM TLT IEF SHY LQD HYG GLD SLV USO DBA UUP VNQ BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_5d_target_xs_v2_lambdarank \
  --forward-horizon 5 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 500 --random-state 42 \
  --rebalance-every 5 \
  --feature-normalization per_asset_train_zscore \
  --include-cross-sectional-features \
  --run-purpose decision_grade --decision-grade
```

5 splits × 2 top-k × 500 nulls = 5,000 null allocations on the 18-asset panel. Metadata flags: `data_downloaded: false`, `include_cross_sectional_features: true`, all safety flags `false`. No yfinance fetch.

### Output files

10 timestamped files. Plus the new `cross_asset_ranking_ranking_diagnostics_<ts>.csv`.

### Tests run

```text
uv run python -m pytest tests/test_cross_asset_ranking.py tests/test_cross_asset_ranking_experiment.py -x -q
... 42 passed in 6.33s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 169 passed in 8.69s
```

3 new tests (Phase 1 + Phase 2). No regressions.

## Full metrics — head-to-head v1 vs v2

(Top-1 v1 numbers retrieved from `results/cross_asset_ranking_5d_target_xs_features_lambdarank/` — the v1 patch report had not surfaced top-1 in the head-to-head.)

| run / policy | mean IR | net Sharpe | active ret | folds (+IR) | folds p≤0.05 | median p | drop-best mean | BTC % | max DD |
|---|---|---|---|---|---|---|---|---|---|
| v1 top-1 | +0.071 | 0.412 | -0.025 | 3/5 | 1/5 | 0.443 | -0.425 | 22.5 | -0.33 |
| **v1 top-2** | **+0.655** | **1.044** | **+0.091** | **4/5** | 1/5 | 0.204 | **+0.240** | 35.6 | -0.22 |
| v2 top-1 | -0.024 | 0.352 | -0.043 | 2/5 | 0/5 | 0.469 | -0.271 | 22.4 | -0.35 |
| v2 top-2 | -0.040 | 0.503 | -0.045 | 2/5 | 0/5 | 0.477 | -0.394 | 37.2 | -0.28 |
| equal-weight 18 ref | 0.000 | 1.018 | 0.000 | n/a | n/a | n/a | n/a | n/a | -0.13 |

Top-2 regression dominates the story. The single v1 directional win (top-2 mean IR +0.65, net Sharpe beats equal-weight) is gone. Top-1 also worsens, though less dramatically.

## Per-date Spearman + ICIR

| run | overall mean Spearman | std (ddof=1) across folds | ICIR | n_folds |
|---|---|---|---|---|
| v1 | +0.0318 | 0.0340 | **+0.9353** | 5 |
| **v2** | **+0.0242** | 0.0244 | **+0.9932** | 5 |

Per-fold Spearman:

```text
v1: s0:-0.029   s1:+0.050   s2:+0.048   s3:+0.039   s4:+0.051
v2: s0:-0.010   s1:+0.052   s2:+0.027   s3:+0.041   s4:+0.012
```

Reading:
- v2 improved split 0 slightly (less negative) and split 1 marginally.
- v2 regressed splits 2 (from +0.048 to +0.027) and 4 (from +0.051 to +0.012).
- Split 3 essentially unchanged.
- Overall mean dropped 24%; std also dropped 28%; ICIR ticked up because std shrank more than mean.

**Important interpretation of the higher ICIR**: v1's ICIR was around a meaningful directional signal. v2's ICIR is around a much weaker signal. Higher ICIR with lower mean is "stable noise" — exactly the kind of trap the formalized rubric is designed to flag. The decision-grade tier requires *both* ICIR ≥ 0.5 *and* mean Spearman ≥ +0.05; v2 clears the first easily and fails the second by a wide margin (+0.024 vs +0.05).

## Per-fold IR (v2)

```text
top-1:  s0:-1.200   s1:-0.097   s2:+0.217   s3:+0.968   s4:-0.006
top-2:  s0:-1.692   s1:-0.038   s2:-0.598   s3:+1.375   s4:+0.750
```

Top-2 went from v1's 4 positive folds (s1, s2, s3, s4) to v2's 2 positive folds (s3, s4). Split 4 — which was v1's strongest fold at +2.31 IR — fell to +0.75. Split 2 — borderline-significant in v1 at +1.31 — fell to −0.60.

Drop-best-fold mean IR for top-2 went from **+0.240** (v1) to **−0.394** (v2). The single positive contribution of v1 (a robust drop-best mean) is gone.

## Random null p-values (v2)

```text
split  top-k  model IR    p-value
   0      1   -1.200     0.894
   0      2   -1.692     0.966
   1      1   -0.097     0.483
   1      2   -0.038     0.477
   2      1   +0.217     0.403
   2      2   -0.598     0.707
   3      1   +0.968     0.142
   3      2   +1.375     0.054   <-- borderline (was 0.066 in v1)
   4      1   -0.006     0.469
   4      2   +0.750     0.156
```

One fold (split 3 top-2) sits at p = 0.054, just barely missing 0.05. Otherwise nothing clears.

## Feature importance — v2 — top 14 by mean gain across 5 splits

| feature | gain | gain % | split_count |
|---|---|---|---|
| **xs_rank_vol_20d** | 1636 | **9.19%** | 77.2 |
| **xs_rank_beta_ew_rest_252d** (new) | 1068 | **6.00%** | 80.8 |
| **xs_rank_beta_ew_rest_60d** (new) | 1024 | **5.75%** | 74.6 |
| realized_vol_20 | 1000 | 5.62% | 70.0 |
| relative_vol_ratio | 961 | 5.40% | 70.6 |
| return_60d | 840 | 4.72% | 59.0 |
| xs_rank_ret_60d | 733 | 4.12% | 56.8 |
| bollinger_band_width_zscore | 722 | 4.06% | 58.4 |
| autocorrelation_zscore | 717 | 4.03% | 60.8 |
| current_drawdown_60d | 715 | 4.02% | 52.4 |
| xs_rank_ret_20d | 671 | 3.77% | 53.2 |
| volatility_regime | 629 | 3.53% | 51.0 |
| vix_relative | 621 | 3.49% | 48.8 |
| downside_vol_ratio | 561 | 3.15% | 47.6 |

Cross-sectional features only (v2 set, 6 features):

| feature | gain | gain % |
|---|---|---|
| xs_rank_vol_20d | 1636 | 9.19% |
| **xs_rank_beta_ew_rest_252d** | **1068** | **6.00%** |
| **xs_rank_beta_ew_rest_60d** | **1024** | **5.75%** |
| xs_rank_ret_60d | 733 | 4.12% |
| xs_rank_ret_20d | 671 | 3.77% |
| xs_rank_drawdown_60d | 453 | 2.55% |

**Combined beta-feature gain: 11.76% of total** (2092 / 17794). **Sum of all xs_rank_* features: 31.39% of total** (vs 22.4% at v1 — the cross-sectional family is even more dominant in importance, just less effective at predicting). All 6 xs features in the top half (top 20 of 40).

The beta features pass the "non-trivially used" gate clearly: combined 11.76% gain, ranked #2 and #3 overall, and both have higher split-count than `xs_rank_vol_20d`. The model is making heavy decisions on them. The decisions just aren't producing better cross-sectional rankings.

## v1 vs v2 feature-importance shifts on retained features

| feature | v1 gain % | v2 gain % | Δ |
|---|---|---|---|
| xs_rank_vol_20d | 8.82 | 9.19 | +0.37 |
| realized_vol_20 | 5.82 | 5.62 | -0.20 |
| relative_vol_ratio | 5.61 | 5.40 | -0.21 |
| return_60d | 5.36 | 4.72 | -0.64 |
| xs_rank_ret_60d | 4.59 | 4.12 | -0.47 |
| bollinger_band_width_zscore | 4.46 | 4.06 | -0.40 |
| volatility_regime | 4.45 | 3.53 | -0.92 |
| xs_rank_ret_20d | 4.44 | 3.77 | -0.67 |
| autocorrelation_zscore | 4.24 | 4.03 | -0.21 |
| current_drawdown_60d | 4.31 | 4.02 | -0.29 |
| vix_relative | 3.93 | 3.49 | -0.44 |

The two new beta features collectively took ~11.8 percentage points of feature gain. They displaced the most-displacement on `volatility_regime` (-0.92pp), `return_60d` (-0.64pp), `xs_rank_ret_20d` (-0.67pp), `xs_rank_ret_60d` (-0.47pp), and `vix_relative` (-0.44pp). The model substituted cross-sectional beta information for vol-regime / momentum signals — but the substitution was a net loss in actual predictive accuracy.

## Pass criteria evaluation

**Decision-grade pass** (BOTH top-1 AND top-2 must satisfy ALL):

| Criterion | Top-1 | Top-2 |
|---|---|---|
| Mean IR > 0 AND active return > 0 | ✗ (-0.024 IR; -0.043 active) | ✗ (-0.040 IR; -0.045 active) |
| Per-date Spearman ≥ +0.05 | ✗ (+0.024) | ✗ (+0.024) |
| ICIR ≥ 0.5 | ✓ (0.99) | ✓ (0.99) |
| Folds passing p ≤ 0.05 ≥ 2/5 | ✗ (0/5) | ✗ (0/5; split 3 at 0.054) |
| Drop-best-fold mean IR > 0 | ✗ (-0.271) | ✗ (-0.394) |
| No BTC dominance, reasonable turnover/cost | ✓ | ✓ |
| At least one xs feature in top half of importance | ✓ (all 6) | ✓ |
| **Top-1 must show material improvement over v1 baseline top-1** | ✗ (delta -0.095) | n/a |

**Directionally interesting**:

| Criterion | Result |
|---|---|
| Top-2 maintains or improves on v1's Spearman ≥ +0.03 AND mean IR > 0 AND ICIR ≥ 0.3 | ✗ — mean IR negative (-0.040 vs v1 +0.655) |
| Top-1 shows positive movement (mean IR delta > 0, Spearman delta > 0) | ✗ — delta -0.095, Spearman delta -0.008 |
| Beta features non-trivially used | ✓ (#2 and #3 by gain) |

**Verdict: FAIL on both tiers.** All three of the fail criteria explicitly enumerated in the spec are triggered:
- Top-1 did not improve over v1 baseline (delta -0.095).
- Top-2 regressed materially below v1's directional thresholds (mean IR -0.040 vs +0.655; Spearman +0.024 vs +0.032).
- (The beta features WERE used — that branch did not trigger, but the other two are sufficient.)

## What this tells us

The hypothesis going in was that beta-to-equal-weight-rest features would help top-1 specifically (single-pick should target diversification, and low-beta-to-rest is exactly diversification). The empirical outcome is the opposite: adding the betas hurt top-1 modestly and hurt top-2 catastrophically.

Several plausible explanations, none verified by this patch:

1. **Multicollinearity-driven instability**: the 60d and 252d beta features are correlated with each other and with `realized_vol_20`, `relative_vol_ratio`, and `xs_rank_vol_20d`. With LightGBM the model can't easily "share" the signal across correlated features in a way that survives across folds; instead it picks one set in one split and another in the next, producing fold-level inconsistency.
2. **Dropping `xs_rank_ret_5d` was a mistake**: even at 1.77% gain, it may have been providing a high-resolution short-term momentum signal that the longer-window features cannot replace. The v2 changes confound dropping it with adding the betas; we cannot separate the two effects without an additional ablation.
3. **The beta features are right but misaligned in time**: 60-day rolling beta is an aggregate of the last 60 days, but the target horizon is only 5 days. The 252-day beta is a multi-quarter aggregate. Both may be encoding regime persistence rather than the near-future ranking signal. The model uses them heavily because they're stable signals, but stable signals about a 6-month relationship don't help predict a 5-day ranking.

Hypothesis (3) is the most interesting because it implies a horizon mismatch within the feature side: the *target* is short-horizon (5d), but the *new features* are very long-horizon (60d, 252d). v1's working features (`xs_rank_vol_20d`, `xs_rank_ret_20d`, `xs_rank_drawdown_60d`) span 20-60 days — closer to the target horizon. The shorter `xs_rank_ret_5d` was discarded *because* its individual gain was low, but it may have been the only feature in the set with horizon parity to the target.

## Stop / go verdict

```yaml
xs_v2_beta_ew_rest:
  result: failed_both_tiers
  v1_baseline_now_best: true
  next_pivot: v3_regime_conditioned_interactions
  do_not_do_next:
    - target_side_pivot
    - seed_sensitivity
    - hyperparameter_tuning
    - universe_changes
    - dispersion_features  (reserved per spec)
    - deep_models
    - stacking
    - re-introducing the betas at v3
  fallback_if_v3_fails: |
    Accept v1 as the empirical ceiling of feature-side improvement on
    this asset class. Pivot architecturally to regime detection + per-
    regime simple allocation, with v1 xs-feature LambdaRank as a within-
    regime ranker.
```

Per the spec's predetermined order, the v2 failure routes to v3 regime-conditioned interactions, not to architectural pivot yet. The v1 baseline is now the high-water mark — every future patch report must compare against v1, not v2.

## Recommended next step

**v3 — regime-conditioned interactions.** Per the spec, this is the last feature-side option before architectural pivot. Concretely:

1. **Restore the v1 xs feature set** (drop the betas, add back `xs_rank_ret_5d` — the v2 change should be reverted as a baseline).
2. **Construct a regime indicator** from VIX z-score (or universe-mean trailing vol rank) — e.g., a categorical regime label (0/1/2 for low/mid/high VIX z-score quantiles) or a smooth indicator in [0, 1].
3. **Add interaction features**: each existing xs feature × the regime indicator. With 5 v1 xs features and one regime indicator, that's 5 new interaction columns.
4. **Run with the same configuration**: LambdaRank, 5d target, 5d rebalance, per-asset train z-score, 500 nulls, 18-asset universe. One factor changed vs v1.

Hypothesis: the model needs to know that the meaning of a high `xs_rank_vol_20d` differs between low-VIX and high-VIX regimes. v1's xs features don't carry that context; v3's interactions would.

If v3 also fails the directional tier (mean Spearman < +0.03 OR no top-2 improvement), the spec mandates accepting v1 as the ceiling of feature-side and pivoting architecturally. Do **not** re-test betas, dispersion, expanded universe, seeds, or hyperparameter variants — those are explicitly off the table per the predetermined pivot order.

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false` in metadata.
- `--prepare-missing` not set; no yfinance fetch.
- `+0.05` decision-grade Spearman threshold was honored — not relaxed despite the v2 failure (+0.024 is reported as a fail, not as "directionally interesting at a lower bar").
- Static-import test continues to enforce no-legacy / no-Optuna / no-deep / no-stacking.
- No champion manifest changes.
- No existing result files overwritten — output went to a fresh per-config directory.
- No hyperparameter tuning, seed variation, or universe change. The only varying factors vs v1 are the two feature-set changes (drop xs_rank_ret_5d, add the two beta features) plus the ICIR/Spearman wiring (which produces output but does not affect the model's training data or predictions).

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Phase 1: `xs_rank_ret_5d` removed, tests updated, no regression on remaining features | ✓ |
| 2 | Phase 2: beta-to-EW-rest features implemented, gated by existing flag, tests pass | ✓ |
| 3 | Phase 3: ICIR computed in runner, formalized in patch doc with retrospective v1 calculation (0.94), thresholds documented | ✓ |
| 4 | Phase 4: dry-run passes, execute completes, failure documented with full diagnostics | ✓ failure documented |
| 5 | No forbidden workflows touched | ✓ |
| 6 | Patch document created | ✓ this file |
