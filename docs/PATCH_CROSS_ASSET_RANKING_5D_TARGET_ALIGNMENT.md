# Patch: Cross-Asset Ranking Horizon Alignment — 5d Target / 5d Rebalance

Date: 2026-05-11
Run timestamp: `20260511T131039Z` → `results/cross_asset_ranking_5d_target_lambdarank/`
Reference run: 18-asset LambdaRank at 20d-target / 5d-rebalance (`20260511T122454Z`)

## TL;DR verdict

**FAIL on both tiers.** Per-date Spearman ρ improved from **+0.010 to +0.017** under horizon alignment (a real, consistent gain — 4 of 5 folds positive, up from 2 of 5), but the absolute level is still **below the +0.03 directionally-interesting threshold** and well below the +0.05 decision-grade pass threshold. Top-1 mean IR flipped from −0.17 to +0.17 and one fold (top-2 split 4) actually clears p ≤ 0.05 at 0.014 — but drop-best-fold mean IR is negative for both top-1 (−0.134) and top-2 (−0.614), so the headline numbers are one-fold artifacts.

Horizon alignment is **moving the needle** but not enough. The diagnosis chain is now four points long (HGB regression Spearman +0.051 [scale bet], LambdaRank-6 +0.014, LambdaRank-18 +0.010, LambdaRank-18 5d-target **+0.017**) and the dominant story is that the model has near-zero cross-sectional ranking skill regardless of target-horizon at the chosen evaluation frequency. The remaining hypothesis is **feature-side** — the per-asset technical features being fed to the model do not encode cross-sectional information well, regardless of what target they're paired with.

Per the spec's predetermined failure-pivot ordering: **recommend the feature-side pivot next**, not the target-side pivot.

```yaml
horizon_alignment_5d_target_18_asset_lambdarank:
  status: failed_both_tiers
  decision_grade_pass: false
  directionally_interesting_pass: false
  per_date_spearman_overall: +0.0174
  per_date_spearman_threshold_directional: +0.03
  per_date_spearman_threshold_decision_grade: +0.05
  top1_mean_ir: +0.174
  top1_drop_best_mean_ir: -0.134
  top1_folds_pass_p05: 0/5
  top2_mean_ir: -0.104
  top2_drop_best_mean_ir: -0.614
  top2_folds_pass_p05: 1/5  (split 4 at p=0.014; rest 0.45-0.96)
  signal_consistency: spearman_positive_in_4_of_5_folds  (improvement)
  conclusion: |
    Horizon alignment helped top-1 sign and per-date Spearman consistency,
    but absolute discrimination is still near zero. Single-fold artifacts
    drive the headline IRs. Target/horizon was not the binding problem.
```

## Codebase note (column naming)

The panel builder (`build_cross_asset_panel`) hard-codes the column name `"forward_20d_risk_adjusted_return"` regardless of the actual `forward_horizon` value. When `--forward-horizon 5` is passed, the column **contains 5d-forward risk-adjusted returns** (math is correct) but the label is stale. This means the experiment's `DEFAULT_TARGET_COLUMN = "forward_20d_risk_adjusted_return"` continues to resolve correctly. The misnomer is documented here and should be cleaned up in a future small patch — but it does not affect the substance of this run.

The Spearman comparisons below correctly use the **realized 5d** target for the 5d-target run and the **realized 20d** target for the 20d-target reference run.

## Commands

Dry-run accepted `--forward-horizon 5 --rebalance-every 5` cleanly and printed `forward_horizon: 5` in the config. Execute:

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM DIA EFA EEM TLT IEF SHY LQD HYG GLD SLV USO DBA UUP VNQ BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_5d_target_lambdarank \
  --forward-horizon 5 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 500 --random-state 42 \
  --rebalance-every 5 --feature-normalization per_asset_train_zscore \
  --run-purpose decision_grade --decision-grade
```

5 walk-forward splits × 2 top-k × 500 random nulls = 5,000 null allocations on the 18-asset panel. Metadata `data_downloaded: false` for this run; all other safety flags `false`. `--prepare-missing` not set; all caches pre-existing from the prior universe-expansion patch.

## Output files

9 timestamped files under `results/cross_asset_ranking_5d_target_lambdarank/`.

## Head-to-head — 5d-target vs 20d-target (both 18 assets, 5d rebalance, normalized features, LambdaRank)

| run / policy | mean IR | net Sharpe | active ret | turnover | cost drag | max DD | folds (+IR) | folds p≤0.05 | median p | BTC % | top-3 conc | distinct ≥5% | drop-best mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20d-target top-1 | -0.174 | 0.172 | -0.027 | 0.290 | 0.015 | -0.33 | 2/5 | 0/5 | 0.487 | 24.8 | 45.8 | 5 | -0.255 |
| 20d-target top-2 | +0.002 | 0.487 | -0.009 | 0.252 | 0.014 | -0.25 | 2/5 | 0/5 | 0.431 | 39.0 | 42.3 | 16 | -0.163 |
| **5d-target top-1** | **+0.174** | 0.550 | +0.063 | 0.315 | 0.019 | -0.25 | **3/5** | 0/5 | **0.361** | 22.5 | 42.1 | **8** | -0.134 |
| 5d-target top-2 | -0.104 | 0.385 | -0.026 | 0.279 | 0.015 | -0.22 | 2/5 | **1/5** | 0.567 | 33.7 | 35.6 | 16 | -0.614 |
| equal-weight 18 ref | 0.000 | 1.018 | 0.000 | 0.004 | 0.0002 | -0.13 | n/a | n/a | n/a | n/a | n/a | 18 | n/a |

What changed from 20d-target to 5d-target:

- **Top-1 mean IR flipped from −0.174 to +0.174.** Net Sharpe rose from 0.17 to 0.55. Three positive folds (up from 2). Max drawdown improved from −0.33 to −0.25. **This is the clearest economic improvement so far in the cross-asset track.**
- **Top-2 went the other way.** Mean IR −0.104 (slightly worse than 20d's near-zero). But got one fold passing p ≤ 0.05 (split 4 at 0.014).
- **BTC selection** dropped slightly in top-1 (24.8% → 22.5%) — still no dominance.
- **Distinct assets ≥5%** rose for top-1 (5 → 8 — more diversified selection), stayed at 16 for top-2 (most of the universe gets picked at some point).
- **Drop-best-fold mean IR** is negative in both policies. The headline numbers are propped up by one or two strong folds.

## The headline metric — per-date Spearman score-vs-target rank correlation

For each run the Spearman is computed against the **model's actual training target** (5d for the 5d-target run, 20d for the 20d-target run).

| run | mean overall | s0 | s1 | s2 | s3 | s4 | folds ρ > 0 |
|---|---|---|---|---|---|---|---|
| 20d-target | +0.0097 | -0.018 | -0.006 | +0.026 | **+0.083** | -0.037 | 2/5 |
| **5d-target** | **+0.0174** | -0.010 | +0.008 | +0.035 | +0.019 | +0.034 | **4/5** |

Two patterns:

- **Mean overall Spearman improved 75% relative** (+0.010 → +0.017). The directional improvement is real.
- **Per-fold consistency improved much more.** 5d-target has 4 of 5 folds with positive mean ρ, including split 0 at only −0.010 (essentially flat) and split 4 flipping from −0.037 to +0.034. The 20d-target's win was concentrated in a single fold (split 3 at +0.083); the 5d-target spreads the small win across four folds.
- **Absolute level is still below thresholds.** +0.017 vs the directional-interest threshold of +0.03 and decision-grade threshold of +0.05.

## Per-fold IR detail (5d-target run)

```text
top-1: s0:IR=-1.672   s1:IR=+1.407   s2:IR=+0.356   s3:IR=-0.037   s4:IR=+0.819
top-2: s0:IR=-1.568   s1:IR=+0.050   s2:IR=-0.694   s3:IR=-0.245   s4:IR=+1.938
```

Split 1 (2021 inflation-buildup year) and split 4 (2024-2025 risk-on year) are the strong folds. Split 0 (COVID rebound) is a disaster for both top-1 and top-2.

The single-fold contribution problem:

- top-1 mean +0.174 with split 1 (+1.41) doing all the work; drop-best mean = −0.134.
- top-2 mean −0.104 with split 4 (+1.94) propping it up; drop-best mean = −0.614.

Per-fold IR is *less* variable than the 20d-target run for top-1 (sd 1.10 vs 1.42), but the same fold-concentration story applies.

## Random null p-values for the 5d-target run

```text
split  top-k  model IR    p-value
   0      1   -1.672     0.958
   0      2   -1.568     0.964
   1      1   +1.407     0.058   <-- just misses 0.05
   1      2   +0.050     0.449
   2      1   +0.356     0.361
   2      2   -0.694     0.749
   3      1   -0.037     0.471
   3      2   -0.245     0.567
   4      1   +0.819     0.168
   4      2   +1.938     0.014   <-- passes (single fold)
```

Top-1 split 1 just misses (0.058). Top-2 split 4 clearly passes (0.014). Neither pattern constitutes "majority of folds" or "strong aggregate evidence."

## Asset selection — 5d-target run by split

```text
top-1 (assets ≥ 8% of dates):
  split 0 (2020-02 → 2021-02, COVID rebound):   UUP 25.8%  IEF 16.7%  DBA 13.9%
    -> dollar/bonds during a recovery; lost -1.67 IR
  split 1 (2021-02 → 2022-02, inflation):       SHY 25.8%  USO 13.9%  HYG 11.9%  BTC 9.9%
    -> short-bonds/cash + commodities; won +1.41 IR
  split 2 (2022-02 → 2023-05, bear/rate-shock): BTC 24.6%  SLV 13.9%  USO 13.9%  DBA 9.9%
    -> commodities heavy; won +0.36 IR (USO was strong; BTC was weak but only 25% of dates)
  split 3 (2023-05 → 2024-08, AI rally):        BTC 37.7%  USO 21.8%
    -> BTC + oil concentration; flat -0.04 IR
  split 4 (2024-08 → 2025-08, risk-on):         BTC 36.5%  GLD 11.9%
    -> BTC + gold; won +0.82 IR

top-2 (assets ≥ 8% of dates):
  split 0:  UUP 37.7%  SHY 27.8%  DBA 23.8%  IEF 22.6%  LQD 19.8%  TLT 15.9%  BTC 9.9%
    -> ultra-defensive in a recovery; -1.57 IR
  split 1:  SHY 39.7%  HYG 25.8%  USO 25.8%  BTC 15.9%  UUP 13.9%  IEF 11.9%  GLD 10.7%  TLT 10.7%  EEM 9.9%  IWM 9.9%
    -> broadly distributed; +0.05 IR
  split 2:  BTC 38.5%  USO 23.8%  SLV 21.8%  UUP 19.8%  DBA 17.9%  GLD 15.9%  EEM 14.7%
    -> commodities-heavy with BTC; -0.69 IR
  split 3:  BTC 49.6%  USO 29.8%  GLD 23.8%  SLV 17.9%  QQQ 13.9%  EEM 11.9%  DIA 9.9%  SPY 9.9%
    -> BTC concentration creeping up; -0.25 IR
  split 4:  BTC 54.4%  SLV 19.8%  GLD 17.9%  VNQ 15.9%  TLT 11.9%  USO 11.9%  SHY 9.9%
    -> BTC + commodities; +1.94 IR (BTC ran in late 2024)
```

The top-1 selection is more diversified and regime-varying than the 20d-target version (which had narrower concentration on TLT/BTC). The top-2 selection still drifts heavily toward BTC in the later splits (3-4), and split 0's defensive-tilt mistake persists.

Notable signs of life: split 1's commodities + cash mix in top-1 captured 2021's inflation regime correctly (+1.41 IR is the largest single-fold win in any cross-asset run in the project), and split 4's BTC/commodities exposure caught the late-2024 risk-on rally. These look like model-driven regime calls, not lucky asset-scale bets — but they are individual folds, not a sustained edge.

## Pass criteria evaluation (tiered)

**Decision-grade pass** (all must hold):

| Criterion | Top-1 | Top-2 |
|---|---|---|
| Mean IR > 0 AND active return > 0 | ✓ (+0.17, +0.06) | ✗ (-0.10, -0.03) |
| Per-date Spearman ≥ +0.05 | ✗ (+0.017) | ✗ (+0.017) |
| Random top-k p ≤ 0.05 in majority of folds OR strong aggregate evidence | ✗ (0/5) | ✗ (1/5) |
| Drop-best-fold mean IR > 0 | ✗ (-0.134) | ✗ (-0.614) |
| No BTC dominance, reasonable turnover/cost | ✓ | borderline (33.7%) |

**Directionally interesting** (per-date Spearman ≥ +0.03 with positive mean IR; other criteria above otherwise satisfied):

| Criterion | Top-1 | Top-2 |
|---|---|---|
| Per-date Spearman ≥ +0.03 | ✗ (+0.017) | ✗ (+0.017) |
| Positive mean IR | ✓ | ✗ |

**Fail.** Spearman is below +0.03; both tiers reject. The 75% relative gain in Spearman from 20d → 5d is real and worth recording, but it does not cross the bar.

## What this leaves us with — full diagnosis chain

| setup | per-date Spearman | mean IR | conclusion |
|---|---|---|---|
| HGB regression (6-asset, unnormalized) | +0.051 | +0.408 | scale bet (forensics confirmed) |
| HGB regression (6-asset, normalized) | -0.004 | +0.088 | signal removed with scale |
| LambdaRank (6-asset, normalized, 5d rebal, 20d target) | +0.014 | +0.337 | regime-aware but null fail |
| LambdaRank (18-asset, normalized, 5d rebal, 20d target) | +0.010 | -0.174 | universe expansion didn't help |
| **LambdaRank (18-asset, normalized, 5d rebal, 5d target)** | **+0.017** | **+0.174** | horizon alignment small gain, still null fail |

Four independent setups, four near-zero Spearman values. The variance between them is small (range +0.010 to +0.017 once the scale-bet HGB run is excluded). This is a strong indicator that the binding constraint is **not** the model family, **not** the universe size, **not** the target-horizon mismatch, and **not** the normalization choice. The remaining unexplored variable is the **features themselves** — every setup has used per-asset technical indicators (returns, vol, RSI, MACD, momentum, etc.) computed independently for each asset.

Per the spec's predetermined ordering:

> 1. **Feature-side pivot (preferred next step)** — keep the forward-return target, but switch from per-asset technical features to genuinely cross-sectional features: relative momentum vs universe median, cross-sectional vol rank, dispersion measures, regime-conditioned interactions. Hypothesis: 18 daily ETFs riding a common risk-on/risk-off factor are not cross-sectionally separable by per-asset features at any horizon; cross-sectional features explicitly encode relative position.

This is the right next move. The current features encode each asset's own technical state but say nothing about *relative* state vs the universe. A relative-momentum or relative-volatility rank computed per date is a fundamentally different input. With per-asset features and normalization, two assets with identical individual technical states get identical scores from the model regardless of what the other 17 assets are doing — but cross-sectional ranking is exactly the question of "what is this asset's position *relative to the others on this date*?"

The target-side pivot (forward volatility / drawdown ranking) is fallback only — the spec correctly notes that forward volatility is more autocorrelated and therefore more learnable, but the trade collapses to a low-vol factor bet rather than a real alpha.

## Stop / go verdict

```yaml
horizon_alignment_test:
  result: fail_both_tiers
  spearman_gain: 0.010 -> 0.017  (real but insufficient)
  next_pivot: feature_side_cross_sectional
  next_pivot_target: keep forward_20d_risk_adjusted_return (or forward_5d)
  next_pivot_features: relative_momentum_rank, cross_sectional_vol_rank,
                       dispersion_measures, regime_conditioned_interactions
  do_not_do_next:
    - target_side_pivot_yet
    - seed_sensitivity
    - hyperparameter_tuning
    - universe_changes
    - deep_models
    - stacking
```

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false`.
- `--prepare-missing` not set; no yfinance fetch.
- No champion manifest changes.
- No existing result files overwritten — output went to a fresh per-config directory.
- No hyperparameter tuning, seed variation, or universe change beyond the single-factor `forward_horizon` switch from 20 to 5.
- Static-import test continues to enforce no-legacy / no-Optuna / no-deep / no-stacking.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Dry-run passes | ✓ |
| 2 | Execute completes or failure documented | ✓ completed |
| 3 | No forbidden workflows touched | ✓ |
| 4 | Patch document created | ✓ this file |

## Recommended next step

**Feature-side pivot: cross-sectional features.** Concretely, prototype the following per-date features on top of the existing per-asset feature panel:

1. **Relative-momentum rank**: for each (date, asset), rank by trailing 20d return across the universe → integer rank 0..17.
2. **Cross-sectional vol rank**: similarly rank trailing 20d realized vol per date.
3. **Dispersion**: cross-sectional std of trailing returns at each date (a date-level feature broadcast to all assets).
4. **Beta vs equal-weight universe**: rolling beta of each asset's return to the equal-weight portfolio of the remaining 17.
5. **Regime-conditioned interactions**: existing per-asset features × an aggregate indicator (e.g. VIX z-score, or universe-mean trailing return).

Implement as additional columns in `prepare_single_asset_feature_frame` or in a new `build_cross_sectional_features(panel)` step inside the experiment runner (computed after `build_cross_asset_panel` returns the long-format panel, so the per-date grouping is natural).

Keep LambdaRank as the model. Keep the 18-asset universe. Keep 5d rebalance and 5d target (the one factor that did show small directional improvement). Run a decision-grade evaluation. Compare per-date Spearman against the +0.017 baseline established by this patch.

**Do not** run the target-side pivot until the feature-side pivot is tested. The diagnosis chain points at features, not target metric.

Also recommended as a cheap cleanup before the next experiment: fix the column-name misnomer in `build_cross_asset_panel` so the column reflects the actual `forward_horizon` (`forward_5d_return`, `forward_5d_risk_adjusted_return`, etc.) rather than always being labeled `forward_20d_*`. Behavior-neutral; eliminates a confusing trap for future readers.
