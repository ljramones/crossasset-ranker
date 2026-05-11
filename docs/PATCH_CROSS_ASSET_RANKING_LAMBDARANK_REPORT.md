# Patch: Cross-Asset Ranking Rank-Objective Prototype — LightGBM LambdaRank

Date: 2026-05-11
LightGBM: 4.6.0 (`LGBMRanker` available)

Run timestamps:
- Plumbing (1 split, 10 nulls): `20260511T013711Z` → `results/cross_asset_ranking_lambdarank_plumbing/`
- Decision-grade 5d (5 splits, 500 nulls, normalized): `20260511T013743Z` → `results/cross_asset_ranking_lambdarank_5d/`

## TL;DR verdict

**LambdaRank produces a structurally different result than HGB regression, but does not validate as a ranking edge.** The hard pass criteria fail — most importantly the per-date Spearman ρ remains tiny (+0.014 mean across folds) and 0/5 folds pass the random-null gate.

That said, LambdaRank top-1 is the **first configuration in the project where the per-fold IR sign is mostly positive (4/5 folds) and the drop-best-fold mean stays positive (+0.229)**. It also fixed the split-2 failure on its own merits (IR moved from −2.32 under HGB to **+0.770**, GLD selection in split 2 rose from 18.7% to 33.7%) without relying on the asset-scale shortcut that propped up the earlier HGB result.

What this means for the broader project:

- The original regression-target HGB result was definitively a regime/scale bet — LambdaRank with normalized features shows different asset selection that varies sensibly by regime (TLT in bond-friendly years, GLD in bear, BTC in risk-on).
- But the per-date ranking signal is still too weak to clear random-null tests. The 6-asset universe simply doesn't carry strong enough features for forward 20-day risk-adjusted ranking at the bar resolution we're using.

```yaml
lambdarank_top1_5d_normed:
  status: behaviorally_better_than_HGB_but_fails_null_gate
  mean_ir: +0.337
  folds_positive_ir: 4/5
  folds_pass_null_p05: 0/5
  median_null_p: 0.216
  per_date_spearman_overall: +0.014
  split_2_ir: +0.770
  split_2_gld_selection: 33.7%
  btc_selection: 22.8%
  turnover_per_day: 0.26
  drop_best_fold_mean_ir: +0.229
  conclusion: structural improvement on resilience and split-2 behavior,
              but no genuine per-date ranking edge (Spearman ~ 0)
```

## Files changed

- `evaluation/cross_asset_ranking.py` — added `make_lambdarank_relevance_labels(panel, ...)` (per-date integer ranks, NaN-safe, deterministic tie-breaking via stable rank ordering) and `build_lambdarank_groups(panel, ...)` (contiguous per-date group sizes).
- `experiments/cross_asset_ranking_experiment.py` — added `KNOWN_MODELS` constant including `"lambdarank"`; added `_score_with_lambdarank(...)` which (i) filters rows with NaN target, (ii) drops dates with <2 valid rows (no rankable pairs), (iii) sorts by (date, asset) for contiguous groups, (iv) fits `LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=100, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=42, verbose=-1)`, (v) scores test rows individually.
- `scripts/run_cross_asset_ranking_experiment.py` — CLI validation now uses `KNOWN_MODELS` so `--models lambdarank` is accepted.
- `tests/test_cross_asset_ranking.py` — 5 new tests for label/group helpers.
- `tests/test_cross_asset_ranking_experiment.py` — 2 new tests: lambdarank runs end-to-end on a synthetic panel and metadata flags stay false; CLI dry-run accepts `--models lambdarank --feature-normalization per_asset_train_zscore`.
- `docs/PATCH_CROSS_ASSET_RANKING_LAMBDARANK_REPORT.md` — this file.

## Tests run

```text
uv run python -m pytest tests/test_cross_asset_ranking.py tests/test_cross_asset_ranking_experiment.py -x -q
... 33 passed in 5.56s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 160 passed in 7.88s
```

No regressions. Same single pre-existing legacy collection error documented in prior patches.

## Dry-run + plumbing + decision-grade commands

Dry-run accepted `--models lambdarank --feature-normalization per_asset_train_zscore` and printed the resolved config without IO.

Plumbing run (single split via `--step-size 5000`, `--random-null-runs 10`, `--run-purpose plumbing`):

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM TLT GLD BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_lambdarank_plumbing \
  --forward-horizon 20 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 5000 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 10 --random-state 42 \
  --rebalance-every 5 --feature-normalization per_asset_train_zscore \
  --run-purpose plumbing
```

Decision-grade 5d run:

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM TLT GLD BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_lambdarank_5d \
  --forward-horizon 20 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 500 --random-state 42 \
  --rebalance-every 5 --feature-normalization per_asset_train_zscore \
  --run-purpose decision_grade --decision-grade
```

Both runs completed end-to-end on the 5-split, 14,712-row panel. No yfinance fetch. Metadata `data_downloaded: false`, `optuna_used: false`, `deep_models_used: false`, `stacking_used: false`.

## Output files

Per run: 9 timestamped files (`summary`, `fold_details`, `scored_panel`, `allocations`, `portfolio_returns`, `random_nulls`, `null_pvalues`, `report`, `metadata`).

## Plumbing-only result (do not interpret)

| model | top_k | net Sharpe | IR | turnover | p-value |
|---|---|---|---|---|---|
| equal_weight | 6 | 1.554 | 0.000 | 0.004 | — |
| lambdarank | 1 | 0.174 | -0.788 | 0.242 | 0.727 |
| lambdarank | 2 | 0.360 | -1.306 | 0.159 | 0.818 |

Single split (the COVID-rebound split 0), n=10 nulls. Negative but not interpretable.

## Decision-grade 5d result vs HGB references

Side-by-side at 5d rebalance, normalized features for the comparable HGB:

| run / policy | mean IR | net Sharpe | turnover | cost drag | folds (+IR) | folds p≤0.05 | median p | BTC % | GLD % in split 2 | split-2 IR | drop-best mean IR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HGB_normed_5d top-2 | +0.088 | 0.901 | 0.213 | 0.013 | 3/5 | 0/5 | 0.391 | 29.3 | 42.5 | +0.350 | -0.016 |
| **LambdaRank_normed_5d top-1** | **+0.337** | 0.866 | 0.261 | 0.017 | **4/5** | 0/5 | **0.216** | **22.8** | 33.7 | **+0.770** | **+0.229** |
| LambdaRank_normed_5d top-2 | -0.253 | 0.638 | 0.202 | 0.012 | 2/5 | 0/5 | 0.609 | 40.6 | 43.7 | +0.917 | -0.546 |
| equal-weight (k=6) ref | 0.000 | 1.118 | 0.004 | 0.0002 | n/a | n/a | n/a | 100 | — | — | — |

Two observations:

1. **LambdaRank top-1 is the best non-baseline configuration on multiple fold-level metrics.** It posts the highest mean IR (+0.337), the most positive folds (4/5), the lowest median p-value (0.216, though still > 0.05), and the most robust drop-best-fold mean (+0.229). It also has the largest split-2 improvement.
2. **LambdaRank top-2 underperforms top-1.** Mean IR is negative. This is the opposite of what we saw with HGB regression (where top-2 was the better policy). With LambdaRank, picking only the top-ranked asset works better than diversifying across the top two — consistent with the loss directly optimizing the ordering of the very top of each date's ranking.

## The headline metric — per-date Spearman score-vs-target rank correlation

| run | mean overall | per-split breakdown |
|---|---|---|
| HGB_unnorm_daily/5d | **+0.0509** | s0:+0.168  s1:-0.069  s2:+0.036  s3:+0.068  s4:+0.050 |
| HGB_normed_5d | -0.0036 | s0:+0.022  s1:+0.037  s2:+0.002  s3:+0.020  s4:-0.098 |
| **LambdaRank_normed_5d** | **+0.0138** | s0:**-0.110**  s1:+0.078  s2:+0.052  s3:+0.040  s4:+0.010 |

The pattern:

- LambdaRank improves Spearman on **4 of 5 folds** vs normalized HGB (s1, s2, s3, s4).
- It is meaningfully worse on split 0 (COVID rebound) — losing that fold to a Spearman of −0.11.
- Overall mean Spearman is **+0.014** — better than normalized HGB (−0.004) but worse than unnormalized HGB (+0.051, which was the scale-bet).
- All values are well below the user's target of **+0.10 to +0.15** for a "real" ranking signal.

Reading: rank-loss training does produce a different model than regression, and the per-date alignment is genuinely better in most folds. But the absolute level is still too small to translate into a statistically significant strategy via top-k extraction.

## Per-fold IR detail

Top-1 IRs (the policy where LambdaRank wins by mean):

| run | s0 | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|---|
| HGB_unnorm_daily | +1.055 | +1.073 | **-2.217** | +2.185 | +0.800 | +0.579 |
| **LambdaRank_normed_5d** | **-0.788** | +0.344 | **+0.770** | +0.755 | +0.605 | **+0.337** |

LambdaRank trades the huge swings of HGB (range -2.22 to +2.19, std 1.65) for a much narrower distribution (range -0.79 to +0.77, std ~0.6). The single bad fold (split 0) is half as bad as HGB's worst, and the four positive folds together produce a more durable signal.

Top-2 IRs:

| run | s0 | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|---|
| HGB_unnorm_daily | +1.720 | -0.017 | -2.319 | +1.229 | +1.430 | +0.408 |
| HGB_unnorm_5d | +0.587 | +1.271 | -1.095 | +0.984 | +0.712 | +0.492 |
| HGB_normed_5d | -0.402 | +0.178 | +0.350 | -0.191 | +0.505 | +0.088 |
| LambdaRank_normed_5d | -1.306 | -0.955 | +0.917 | +0.412 | -0.333 | -0.253 |

LambdaRank top-2 is the worst top-2 configuration. The mean is negative and only 2 of 5 folds are positive.

## Asset selection — LambdaRank top-1 by split

```text
split 0 (2020-02 → 2021-02, COVID rebound):    TLT 38.5%  SPY 17.9%  QQQ 13.9%  BTC-USD 9.9%  GLD 9.9%  IWM 9.9%
split 1 (2021-02 → 2022-02, sideways/inflation): TLT 32.5%  BTC-USD 19.8%  GLD 13.9%  QQQ 13.9%  SPY 13.9%  IWM 6.0%
split 2 (2022-02 → 2023-05, bear/rate-shock):   GLD 33.7%  BTC-USD 27.8%  QQQ 20.6%  TLT 9.9%   SPY 6.0%   IWM 2.0%
split 3 (2023-05 → 2024-08, AI rally):          BTC-USD 25.8%  SPY 19.8%  GLD 17.9%  TLT 16.7%  IWM 9.9%   QQQ 9.9%
split 4 (2024-08 → 2025-08, risk-on):           BTC-USD 30.6%  GLD 15.9%  IWM 15.9%  TLT 13.9%  QQQ 11.9%  SPY 11.9%
```

This is genuinely *regime-aware* selection — picks vary substantially by fold:

- Bonds (TLT) feature heavily in the lower-vol early period.
- GLD comes to the fore in the 2022 bear regime.
- BTC dominates the risk-on tail.

The pattern is different from HGB regression, which consistently leaned to whatever its training period favored. LambdaRank's selections move with the regime instead of being stuck in the train-period bias. This is the *kind* of behavior we want from a cross-asset model — even if the absolute Spearman alignment is too weak to clear null gates.

LambdaRank top-2 in split 2 picks BTC 51.6% (most), GLD 43.7%. BTC's split-2 dominance pulls the top-2 result down because BTC lost 37% in split 2. Top-1 avoids this trap by being more selective; top-2 forces inclusion of a second pick which often becomes BTC.

## Pass criteria evaluation — LambdaRank top-1 (5d, normalized)

| Criterion | Threshold | Result | Pass? |
|---|---|---|---|
| Mean IR vs equal-weight > 0 | > 0 | +0.337 | ✓ |
| Random top-k p-value ≤ 0.05 in majority of folds | ≥ 3/5 | 0/5 | ✗ |
| Per-date Spearman improves materially | > +0.10 ideally; certainly > prior +0.05 | +0.014 (overall) | ✗ |
| Split 2 no longer fails badly | not strongly negative | +0.770 | ✓ |
| Result not BTC-dominated | BTC < 50% | 22.8% | ✓ |
| Turnover/cost reasonable at 5d | turnover < 0.5/day | 0.26/day | ✓ |
| Performance not driven by one fold | drop-best mean > 0 | +0.229 | ✓ |

**5 of 7 pass; 2 hard ones fail.** The two failures are the only ones that constitute *statistical validation* — null-significance and Spearman. Without them, the candidate is interesting behaviorally but not validated.

## Stop / go verdict

```yaml
lambdarank_top1_5d_normed:
  status: behavioral_improvement_over_HGB_but_not_validated
  promising: cautiously_yes_for_top_1
  production_ready: false
  
  what_changed_from_HGB:
    - per-fold IR variance dropped: std 1.49 → ~0.6
    - drop-best-fold mean stays positive (+0.229 vs +0.08 daily HGB)
    - split-2 failure inverted to +0.77
    - asset selection now varies by regime
    - BTC dominance reduced (22.8%)
    - per-date Spearman slightly positive in 4/5 folds
    
  what_did_not_change:
    - null-significance: still 0/5 folds at p ≤ 0.05
    - absolute Spearman remains < 0.02 overall
    - net Sharpe (0.87) still below equal-weight (1.12)
    - top-2 policy still negative
```

## Why the null gate still fails

LambdaRank top-1 produces real, regime-aware behavior but a per-date Spearman of only +0.014 means the model has minimal cross-sectional discrimination on any given day. Its wins come from a few well-timed regime shifts that happen to be right (split 2 GLD, split 3-4 BTC), and over a 252-day fold those wins don't accumulate enough edge to clear the variance of a 500-run random top-1 distribution. The standard error of an IR over ~50 effective decisions (5d rebalance × 252 days) is large; a real signal needs either much larger per-decision edge or many more decisions.

The deeper issue is exposure of a project-level constraint: **6 assets is a small universe for cross-sectional ranking.** Per-date groups are only 6 elements, so the maximum NDCG@1 information per group is binary and the rank-loss gradient is dominated by a few hard pairs. A 12-30 asset universe would give the loss more signal per group and is the natural next experiment in this direction.

## Robustness follow-ups (not for this patch)

1. **Expand the universe.** Add a handful of related ETFs (e.g. EFA, EEM, HYG, DBC, USO, USDU, UUP) and re-run lambdarank. Larger groups give NDCG more to optimize against.
2. **Target/rebalance horizon alignment.** Try forward 5-day target with 5-day rebalance, and forward 20-day with 20-day rebalance (the latter we already know collapses — but it was tested under regression loss, not lambdarank).
3. **Per-date cross-sectional z-score** of features (different invariance than per-asset).
4. **Cash-out option** — already noted in the original feasibility report.
5. **Different `min_child_samples` / `num_leaves`** for LGBMRanker — current values are reasonable but unverified for this small-group regime.

None of these are appropriate for this patch (the brief was specifically about rank-objective with 6 assets and the existing pipeline). They are the natural next experiments if the team wants to keep pushing.

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false` in both new metadata files.
- `--prepare-missing` not set; no yfinance fetch.
- LightGBM 4.6.0 was already an existing dependency; no new dependency added.
- Static-import test continues to enforce no-legacy / no-Optuna / no-deep / no-stacking invariants.
- No champion manifest changes.
- No existing result files overwritten.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | LightGBM availability confirmed | ✓ 4.6.0, `LGBMRanker` imported lazily |
| 2 | Lambdarank model path implemented | ✓ `_score_with_lambdarank` in experiment module |
| 3 | Tests pass | ✓ 7 new tests, 33/33 cross-asset suite, 160/160 collectible suite |
| 4 | Dry-run passes | ✓ |
| 5 | Tiny plumbing run completes | ✓ pipeline confirmed end-to-end |
| 6 | Decision-grade 5d run completes | ✓ |
| 7 | No old workflows / Optuna / deep / stacking / data download | ✓ |
| 8 | Report created | ✓ this file |

## Recommended next step

Given:

- LambdaRank top-1 improved behavior but still fails statistical validation.
- Per-date Spearman is the binding constraint and is still ~0.
- The 6-asset universe constrains the loss's ability to learn ranking.

The single most informative next move is **expanding the universe** before any further model changes. Adding ~10-15 more liquid ETFs gives the rank loss a meaningfully larger NDCG-relevant signal per date and tests whether the framework can find ranking edges at all under more typical cross-asset breadth. If LambdaRank's Spearman still stays near zero with a 15-20 asset universe, the conclusion is structural — these features do not support 20-day forward ranking — and the next direction should be a different target (e.g. forward volatility / drawdown ranking) rather than further model tuning.

Do **not** run seed sensitivity, deeper hyperparameter search, or move to deep models yet — the diagnosis is structural, not stochastic.
