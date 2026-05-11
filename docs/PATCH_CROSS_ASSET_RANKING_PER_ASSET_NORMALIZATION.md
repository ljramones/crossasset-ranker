# Patch: Cross-Asset Ranking Per-Asset Feature Normalization Sensitivity

Date: 2026-05-10
Run timestamps:
- Normalized daily: `20260511T005714Z` → `results/cross_asset_ranking_norm_daily/`
- Normalized 5d:    `20260511T010337Z` → `results/cross_asset_ranking_norm_5d/`
- Reference unnormalized daily: `20260510T231114Z` → `results/cross_asset_ranking_decision_run/`
- Reference unnormalized 5d: `20260511T000316Z` → `results/cross_asset_ranking_rebalance_5d/`

## TL;DR verdict

**Outcome C from the spec — normalization worsens overall results.** Mean per-date Spearman rank correlation collapsed from **+0.051 to −0.004** across folds. Mean IR went from +0.408 to −0.818 (daily) and from +0.492 to +0.088 (5d). Per-asset z-scoring confirms the split-2 forensics: the unnormalized "edge" was almost entirely the model identifying assets through raw feature scale, not genuine per-date ranking skill.

A more nuanced sub-finding rescues the diagnosis: **normalization fixed split-2's specific failure** — GLD selection jumped from 18.7% to 41.3% (daily) / 42.5% (5d), and split-2 IR moved from −2.32 to +0.35 at 5d. Normalization gave the model a *better* view of GLD's true momentum strength in the bear regime. But every other fold lost more than split 2 gained, because the wins in those folds were driven by the very scale-based shortcut that normalization removed.

Recommendation: **proceed to a rank-based loss (LambdaRank / pairwise ranking) experiment.** The current regression-on-`forward_20d_risk_adjusted_return` target is not teaching the model to discriminate within-date. **Do not run seed sensitivity** — the variance source is now identified.

## Files changed

- `evaluation/cross_asset_ranking.py` — added `normalize_features_per_asset_train_only(panel, *, train_dates, feature_columns, ...)`. Per-asset, train-only mean/std; constants → 0; row order preserved; non-feature columns untouched.
- `experiments/cross_asset_ranking_experiment.py` — added `feature_normalization: str = "none"` to `CrossAssetRankingConfig`; added `FEATURE_NORMALIZATION_CHOICES`; per-split, when `per_asset_train_zscore` is selected, builds a separate normalized `feature_panel` (used for model fitting/scoring) while `compute_allocation_returns` still reads `return_1d` from the *raw* panel; recorded `feature_normalization` in metadata.
- `scripts/run_cross_asset_ranking_experiment.py` — added `--feature-normalization {none,per_asset_train_zscore}` flag.
- `tests/test_cross_asset_ranking.py` — 5 new tests: train-only stats per asset; outliers in val/test do not affect train z-scores; target/forward/rank columns untouched; zero-std safe; row order preserved.
- `docs/PATCH_CROSS_ASSET_RANKING_PER_ASSET_NORMALIZATION.md` — this file.

## Tests run

```text
uv run python -m pytest tests/test_cross_asset_ranking.py -x -q
... 20 passed in 0.37s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 153 passed in 6.85s
```

No regressions. Same single pre-existing legacy collection error as documented in prior patches.

## Implementation notes

**Why two panel views.** The naive implementation would normalize the whole panel and then both feed it to the model and read `return_1d` for allocation returns from the same frame. That would silently break P&L computation — `return_1d` is in the feature list, so it would be normalized, and `compute_allocation_returns` would multiply weights by z-scored returns. To avoid that, the per-split loop now keeps two views:

```text
train_panel_raw / test_panel_raw   <- raw return_1d, used for compute_allocation_returns
train_features  / test_features    <- normalized features (when configured), used for model fit/score
```

**Per-split application.** Train-only statistics are recomputed for every split using only that split's `train_dates`. This is fold-safe — no leakage from validation or test into the normalization parameters.

**Other models not affected.** `momentum_baseline` reads `return_20d` from `test_features`. Under normalization that means it ranks by the per-asset z-score of return_20d — which is actually the *better* momentum heuristic for cross-asset comparison than raw return_20d. Linear regression and HGB also see the normalized features.

## Dry-run commands

Same arg shape as the spec, with output dirs `results/cross_asset_ranking_norm_daily/` and `results/cross_asset_ranking_norm_5d/`. Both dry-runs printed `feature_normalization: per_asset_train_zscore` and made no IO.

## Execute commands and results

Full command for normalized daily (5d differs only by `--rebalance-every 5` and `--output-dir`):

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment \
  --execute --assets SPY QQQ IWM TLT GLD BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_norm_daily \
  --forward-horizon 20 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 2 \
  --models hist_gradient_boosting \
  --random-null-runs 500 --random-state 42 \
  --rebalance-every 1 \
  --feature-normalization per_asset_train_zscore \
  --run-purpose decision_grade --decision-grade
```

Both runs completed end-to-end on the 5-split, 14,712-row panel. 500 nulls per (split, top_k). No yfinance fetch. Metadata `data_downloaded: false`, `feature_normalization: per_asset_train_zscore`, all safety flags `false`.

## Output files

Per run, 9 timestamped files (`summary`, `fold_details`, `scored_panel`, `allocations`, `portfolio_returns`, `random_nulls`, `null_pvalues`, `report`, `metadata`).

## Normalized vs unnormalized — head-to-head

HGB top-2 across all four configurations:

| run | mean IR | median IR | active ret | net Sharpe | turnover | cost drag | max dd | folds (+IR) | folds p≤0.05 | median p | BTC % | GLD % in split 2 | split-2 IR | split-2 top-2 hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| daily_unnorm | +0.408 | +1.229 | +0.077 | 1.385 | 0.706 | 0.049 | -0.213 | 3/5 | 3/5 | 0.042 | 30 | **18.7%** | -2.32 | 0.627 |
| 5d_unnorm | +0.492 | +0.712 | +0.087 | 1.362 | 0.212 | 0.014 | -0.209 | 4/5 | 0/5 | 0.196 | 28 | **13.9%** | -1.10 | 0.651 |
| **daily_normed** | **−0.818** | -0.614 | -0.153 | 0.319 | 0.799 | 0.043 | -0.270 | 0/5 | 0/5 | 0.555 | 33 | **41.3%** | -1.42 | 0.631 |
| **5d_normed** | **+0.088** | +0.178 | +0.005 | 0.901 | 0.214 | 0.013 | -0.216 | 3/5 | 0/5 | 0.391 | 29 | **42.5%** | **+0.35** | 0.627 |

## The headline metric — per-date Spearman score-vs-target rank correlation

```text
daily_unnorm   mean_overall = +0.0509    [s0:+0.168  s1:-0.069  s2:+0.036  s3:+0.068  s4:+0.050]
5d_unnorm      mean_overall = +0.0509    (same scored panel as daily; rebalance is downstream)
daily_normed   mean_overall = -0.0036    [s0:+0.022  s1:+0.037  s2:+0.002  s3:+0.020  s4:-0.098]
5d_normed      mean_overall = -0.0036    (same scored panel as daily)
```

This is the cleanest finding in the patch. Per-date rank correlation between HGB scores and the realized target dropped from **+0.051 to −0.004** under normalization. The model has *no per-date ranking skill at all* once feature scales are equalized across assets.

The unnormalized scoring concentrated its small skill in split 0 (ρ = +0.168). Normalization redistributed scoring across folds — every fold's |ρ| is now ≤ 0.10 — and the split-0 advantage disappeared with it. There is no fold in the normalized runs where the model has more than trivial per-date alignment with the realized target.

## Per-fold IR

|  | daily_unnorm | 5d_unnorm | daily_normed | 5d_normed |
|---|---|---|---|---|
| split 0 (2020-02 → 2021-02) | +1.72 | +0.59 | -1.85 | -0.40 |
| split 1 (2021-02 → 2022-02) | -0.02 | +1.27 | -0.20 | +0.18 |
| **split 2 (2022-02 → 2023-05)** | **-2.32** | **-1.10** | -1.42 | **+0.35** |
| split 3 (2023-05 → 2024-08) | +1.23 | +0.98 | -0.61 | -0.19 |
| split 4 (2024-08 → 2025-08) | +1.43 | +0.71 | -0.01 | +0.51 |
| **mean** | **+0.41** | **+0.49** | **−0.82** | **+0.09** |
| std across folds | 1.49 | 0.83 | 0.71 | 0.34 |
| drop-best-fold mean | +0.08 | +0.30 | -1.02 | -0.02 |

Two patterns to note:

1. **Variance across folds collapsed under normalization.** Std of fold IRs went from 1.49 to 0.71 (daily) and 0.83 to 0.34 (5d). The model's directional bets — the source of both the wins and the catastrophic split-2 loss — got muted.
2. **Split 2 is the only fold that improved** under normalization (5d: −1.10 → +0.35). Every other fold regressed.

## Asset selection diagnostics

Split-2 GLD selection percentage tells the cleanest story:

| run | split-2 GLD % |
|---|---|
| daily_unnorm | **18.7%** (least of any asset — wrong, GLD was the only winner) |
| 5d_unnorm | **13.9%** |
| daily_normed | **41.3%** (now most-selected — correct) |
| 5d_normed | **42.5%** (most-selected — correct) |

**Normalization fixed the specific symptom from the forensics.** With per-asset z-scoring, the model could see GLD's strong normalized momentum during the 2022 bear and selected it appropriately.

But the top-2 hit rate barely moved (0.627 daily_unnorm → 0.631 daily_normed → 0.627 5d_normed). All four configurations remain near or below the random-baseline 0.600 for top-2 hit rate. The model's per-date discrimination is essentially random regardless of whether features are normalized.

BTC selection across all four runs stays in 28-33% — no BTC dominance under either normalization regime.

## Stop / go evaluation against spec criteria

Pass criteria: per-asset normalization is useful only if **all** of the following hold.

| Criterion | Result | Pass? |
|---|---|---|
| Per-date Spearman correlations improve materially | -0.004 vs +0.051 — they got worse | ✗ |
| Split 2 improves, especially GLD selection or realized rank | GLD went 19% → 42%, split-2 IR went −2.32 → +0.35 (5d) | ✓ |
| 5d IR remains positive | +0.088 (essentially zero) | borderline |
| Random-null p-values improve or remain competitive | All folds 0/5 at 5d, median p worse (0.391 vs 0.196) | ✗ |
| Turnover remains reasonable | 0.21/day at 5d (unchanged) | ✓ |
| Result not dominated by one fold | Folds more even but at zero average | mixed |
| No BTC dominance | 29% at 5d normed | ✓ |

**Two of seven gates fail outright; two are borderline.** Aggregate verdict: normalization is *not* useful as a fix.

## Mapping to the spec's outcome categories

> A. Normalization improves split 2 and rank correlations: proceed to seed sensitivity.
> B. Normalization improves economics but not rank correlations: still fragile; consider rank-based loss.
> C. Normalization worsens results: scale/asset identity was part of what made the signal work; consider target/loss redesign.
> D. Normalization inconclusive: proceed to rank-based loss design or rebalance/target horizon alignment.

This is **Outcome C** for overall results, with a sub-finding consistent with **Outcome A specifically for split 2**.

The combined reading: the *symptom* (wrong picks in split 2) was caused by raw feature scale; normalization fixes that symptom. But the *underlying weakness* (no genuine per-date ranking skill) was previously masked by the model's directional bets aligning with the train regime, which itself was enabled by raw feature scales. Removing the scale removes both the symptom and the masking — and the underlying weakness is now visible.

## Stop / go verdict

```yaml
cross_asset_hgb_top2_with_per_asset_normalization:
  status: signal_not_present
  daily_normed_ir: -0.818
  5d_normed_ir:    +0.088
  per_date_spearman_overall: -0.004
  conclusion: per-asset feature normalization is not the fix; the model has
              no genuine per-date ranking skill under normalized features.

cross_asset_hgb_top2_unnormalized:
  status: previous_provisional_pass_invalidated
  reason: the daily 3/5-fold null pass was driven by raw feature scale acting
          as an asset/regime identifier, not by per-date ranking. Once removed,
          the apparent edge disappears.
```

## Recommended next step

Per the spec's Outcome C and the user's framing:

**Switch the target/loss to a rank-based objective.** Replace the regression-on-`forward_20d_risk_adjusted_return` with a within-date pairwise ranking loss — for each (date, asset_i, asset_j) pair where `target_i > target_j`, the model must score `score_i > score_j`. Concretely:

1. Use `sklearn.ensemble.HistGradientBoostingRegressor` with a custom training loop that converts each date's panel into pairwise comparisons, OR
2. Use **LightGBM with `objective="lambdarank"`** (LightGBM is already a dependency per `pyproject.toml`). Group by date; relevance label is the asset's rank by realized `forward_20d_risk_adjusted_return` within that date.

This directly addresses the diagnosis: the model is currently being trained to predict the *value* of forward risk-adjusted return — with target distributions that differ by asset (BTC's variance dominates) — and never sees the cross-sectional ordering it's actually being evaluated on at allocation time. A rank loss inside each date forces the model to learn "is asset i better than asset j *on this date*" — which is the actual ranking question.

Sanity check before that: it would be cheap to also try a **per-date cross-sectional z-score** of features (instead of per-asset across time) as a complementary normalization. That's symmetric to the per-asset variant and tests a different hypothesis (whether the model needs cross-sectional invariance per date).

Do **not** run seed sensitivity. The signal source is now identified; seed variation would just measure the variance of a model that doesn't have per-date ranking skill in the first place.

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false` in both new metadata files.
- `--prepare-missing` not set; no yfinance fetch.
- Static-import test continues to enforce no-legacy invariants.
- No champion manifest changes.
- No existing result files overwritten — both runs wrote to fresh per-config output directories.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | feature-normalization option exists | ✓ `--feature-normalization {none, per_asset_train_zscore}` |
| 2 | Tests pass | ✓ 5 new tests, 20/20 ranking suite, 153/153 collectible suite |
| 3 | Normalized daily run completes or failure documented | ✓ |
| 4 | Normalized 5d run completes or failure documented | ✓ |
| 5 | No old workflows used | ✓ |
| 6 | No Optuna / deep / stacking / data download | ✓ |
| 7 | Report created | ✓ this file |
