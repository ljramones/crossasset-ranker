# Patch: Cross-Asset Ranking Robustness Test 1 — Rebalance Frequency Sweep

Date: 2026-05-10
Daily reference run: `20260510T231114Z` (decision-grade)
5-day rebalance run: `20260511T000316Z`
20-day rebalance run: `20260511T000803Z`

## TL;DR verdict

```yaml
hgb_top2_at_daily:    PROVISIONAL_PASS  (3/5 folds at p<=0.05)
hgb_top2_at_5d:       FAIL_NULL_GATE    (0/5 folds; mean IR up; turnover 3.3x lower)
hgb_top2_at_20d:      COLLAPSE          (mean IR negative; horizon mismatch)
```

**The daily signal does not survive the random-null gate at 5d rebalance.** The economic signal (mean IR, fold-level positivity, asset diversification) is actually *slightly stronger* at 5d, but random top-k allocations match it once they share the same low-turnover schedule. The daily gate-pass at p≤0.05 was likely an artifact of having ~252 daily decisions per fold to win edge cases against ~252 random sequences — when both shrink to ~50 effective decisions per fold (5d holds), the model's per-fold edge becomes statistically indistinguishable.

**Net status: HGB top-2 is fragile / turnover-dependent.** The split-2 universal failure and feature-normalization concerns from the prior patch remain live. Treat the candidate as **fragile**, not **promising**, until those are investigated.

## Files changed

- `evaluation/cross_asset_ranking.py` — added `apply_rebalance_schedule(allocations, *, rebalance_every, ...)` post-processor: freezes weights between rebalance dates anchored to the first allocation date in the slice. Identity transform when `rebalance_every == 1`.
- `experiments/cross_asset_ranking_experiment.py` — added `rebalance_every: int = 1` to `CrossAssetRankingConfig`; runner applies the rebalance schedule to **both** model allocations and random-null allocations (so the comparison is fair); records `rebalance_every` in metadata; updated `policy` description.
- `scripts/run_cross_asset_ranking_experiment.py` — added `--rebalance-every N` and `--models <subset>` CLI flags; threaded both into `_config_from_args`.
- `tests/test_cross_asset_ranking.py` — added 4 tests: `rebalance_every=1` is identity; weights are held between rebalance dates; turnover decreases as frequency increases; one-bar lag is preserved (deterministic test where only day 1's gross return is constrained to match across schedules — day 2 *must* differ when rebalance frequency > 1, which is the whole point).

## Tests run

```text
uv run python -m pytest tests/test_cross_asset_ranking.py -x -q
... 15 passed in 0.35s

uv run python -m pytest tests/test_cross_asset_ranking.py tests/test_cross_asset_ranking_experiment.py -x -q
... 21 passed in 4.66s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 148 passed in 6.91s
```

No regressions. The single ignored collection (`tests/test_integrity_audit_matched_nulls.py`) is the same pre-existing legacy-import failure documented in `docs/PATCH_DATA_REBUILD_ACTIVE_TRACK_REPORT.md`.

## Dry-run commands and results

```bash
# 5d
uv run python -m scripts.run_cross_asset_ranking_experiment --dry-run \
  --assets SPY QQQ IWM TLT GLD BTC-USD \
  --cache-dir data/multi_asset_cache --output-dir results/cross_asset_ranking_rebalance_5d \
  --forward-horizon 20 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 2 --models hist_gradient_boosting \
  --random-null-runs 500 --random-state 42 --rebalance-every 5 \
  --run-purpose decision_grade --decision-grade

# 20d  (same args, --rebalance-every 20, output dir cross_asset_ranking_rebalance_20d)
```

Both dry-runs printed the resolved config including `rebalance_every: 5/20` and `models: ['hist_gradient_boosting']`. No data load, no model fit, no file writes.

## Execute commands and results

Same arguments with `--dry-run` swapped for `--execute`. Both runs completed end-to-end on the existing 5-split, 14,712-row panel (2015-01-09 → 2026-05-07). 500 random nulls per (split, top_k) = 2,500 nulls per run.

Output files (9 per run) under:
- `results/cross_asset_ranking_rebalance_5d/cross_asset_ranking_*_20260511T000316Z.{csv,json,md}`
- `results/cross_asset_ranking_rebalance_20d/cross_asset_ranking_*_20260511T000803Z.{csv,json,md}`

Both metadata files record `data_downloaded: false`, `decision_grade: true`, `main_py_used: false`, `optuna_used: false`, `deep_models_used: false`, `stacking_used: false`, `prepare_experiment_used: false`, `old_model_zoo_used: false`. No yfinance fetch occurred. All caches were pre-existing.

## Cross-frequency comparison — HGB top-2

| freq | mean IR | mean active ret | net Sharpe | turnover | cost drag | max dd | folds (+IR) | folds p≤0.05 | median p | BTC % |
|---|---|---|---|---|---|---|---|---|---|---|
| daily | +0.408 | +0.077 | 1.385 | 0.706 | 0.049 | -0.213 | 3/5 | **3/5** | **0.042** | 30.2 |
| **5d** | **+0.492** | **+0.087** | **1.362** | **0.212** | **0.014** | **-0.209** | **4/5** | **0/5** | **0.196** | **27.5** |
| 20d | -0.380 | -0.047 | 0.629 | 0.064 | 0.004 | -0.227 | 0/5 | 0/5 | 0.555 | 22.2 |
| equal-weight (k=6) ref | 0.000 | 0.000 | 1.118 | 0.004 | 0.0002 | -0.192 | n/a | n/a | n/a | 100 |

Per-fold IR and p-values (HGB top-2, 5 splits each):

| freq | fold IRs | fold p-values |
|---|---|---|
| daily | [+1.72, −0.02, **−2.32**, +1.23, +1.43] | [0.036, 0.363, 0.974, 0.042, 0.024] |
| 5d | [+0.59, +1.27, **−1.09**, +0.98, +0.71] | [0.261, 0.092, 0.846, 0.102, 0.196] |
| 20d | [−0.16, −0.01, −0.12, −0.40, −1.21] | [0.555, 0.511, 0.537, 0.649, 0.894] |

## What changed between daily and 5d

- **Economic signal got slightly stronger.** Mean IR rose from +0.408 to +0.492. Fold-positive count rose from 3/5 to 4/5. Per-fold IRs at 5d are *less variable* (range 0.59 to 1.27 in positive folds vs 1.23 to 1.72 daily) — the signal is more consistent.
- **Costs collapsed.** Turnover dropped from 0.71/day to 0.21/day (3.3× lower). Cost drag dropped from 4.9% annualized to 1.4%.
- **Net Sharpe is essentially unchanged** (1.385 → 1.362), still beating equal-weight 1.118.
- **Asset diversification is preserved.** BTC selection at 5d is 27.5%, every asset is selected at ≥27% of dates. No dominance.
- **Statistical inference vs random nulls collapsed.** Median p-value rose from 0.042 to 0.196. Zero of five folds clear p ≤ 0.05.

The most important observation: the random-null comparison failed *despite* the underlying economic metric improving. The cause is mechanical — when both the model and the nulls rebalance every 5 days, there are roughly 50 effective independent decisions per 252-day fold instead of 252. The standard error of the per-fold IR for a random allocation grows accordingly, so the model needs a larger absolute IR margin to clear the null distribution. The daily 3/5-fold pass was therefore probably an artifact of granularity, not of a robust cross-asset edge.

## What happened at 20d

- Mean IR went **negative** (−0.380). Active return negative (−0.047).
- Net Sharpe collapsed to 0.629 — **worse than equal-weight (1.118)**.
- All five folds posted negative IR.
- p-values uniformly > 0.5 (worse than random).

This is consistent with **horizon mismatch**. The model's target is `forward_20d_risk_adjusted_return`, so its predictions are calibrated for ~20-day forward windows. Holding the same allocation for the entire 20-day window means the second half of each rebalance period is operating on a stale signal — by trading day 15, the conditions that led to the day-0 prediction may no longer apply, and the model has no opportunity to update. Sub-horizon rebalancing (e.g. 5d) lets the model refresh four times within each forward window; super-horizon rebalancing throws away this corrective ability.

## Asset selection (all three frequencies)

```text
daily:  IWM 39.8% | QQQ 36.3% | TLT 32.1% | GLD 30.9% | SPY 30.7% | BTC-USD 30.2%
5d:     IWM 41.4% | TLT 35.6% | QQQ 34.4% | SPY 31.3% | GLD 29.7% | BTC-USD 27.5%
20d:    IWM 46.3% | TLT 37.5% | GLD 34.9% | QQQ 33.3% | SPY 25.7% | BTC-USD 22.2%
```

No frequency exhibits BTC dominance. IWM is the most-selected at every frequency. Distribution is reasonable across all six.

## Pass criteria evaluation

| Criterion | 5d result | Pass? |
|---|---|---|
| 5d rebalance has positive mean IR | +0.492 | ✓ |
| 5d rebalance passes random nulls in ≥3/5 folds (or comparable aggregate evidence) | 0/5 folds; median p 0.196 | **✗** |
| Turnover materially decreases versus daily | 0.21 vs 0.71 (3.3× lower) | ✓ |
| Result is not BTC-dominated | BTC 27.5% | ✓ |
| 20d does not completely collapse, or collapse is documented as horizon mismatch | Collapsed; documented as horizon mismatch above | ✓ (documented) |

The single hard failure is the random-null gate. Because the spec made null-passing required, the run must be marked as a fail at decision-grade.

## Stop / go verdict

Per the spec's "If 5d fails badly" branch: **mark daily signal as fragile / turnover-dependent.**

The 5d run did not fail "badly" in an economic sense — mean IR is slightly positive, asset diversification is preserved, and net Sharpe still beats equal-weight. But statistical robustness vs random allocations of the same shape did not survive. That, combined with the prior patch's findings (split-2 universal failure, drop-best-fold mean drops 80%, single seed only), is enough to downgrade the candidate.

```yaml
cross_asset_hgb_top2:
  status: fragile
  promising: borderline
  production_ready: false
  cost_robust: yes_at_5d
  null_robust: no_at_5d
  next_gate:
    - split_2_forensics
    - per_asset_feature_normalization
    - then re-run rebalance sweep with normalized features
  biggest_risks:
    - daily null-pass was likely a granularity artifact
    - signal still concentrated in 1-2 strong folds
    - 20d horizon mismatch (expected, but suggests rebalance-horizon coupling)
    - single seed
```

## Limitations

- **Per-fold null comparison only.** A pooled null test (Fisher's combined or stratified permutation) would give a single aggregate p-value across folds and might be a more honest summary. Median or geometric-mean p-value were used here as proxies; both indicate non-significance at 5d.
- **Single random seed.** The random-state for both model fits and null sampling is `42`. Robustness across seeds {1, 7, 13, 99} is the next planned gate.
- **5d rebalance is anchored to the first day of each test slice**, not to a calendar (e.g. weekly-Monday). This is fine for the experiment but means the rebalance days drift across folds. A follow-up could test calendar-anchored rebalance.
- **No per-asset normalization yet.** BTC's volatility scale dominates the feature distribution. The fragility may partly be a feature-scale issue that disappears under cross-sectional z-scoring.

## Recommended next step

Per the spec's "If 5d fails badly" recommendation:

1. **Split-2 forensics** — produce a per-split asset-vs-model comparison for the universal split-2 failure. Identify which assets dominated forward returns in that test window vs which the model selected. If the failure is concentrated in specific asset selections, the issue may be isolatable; if every model failed because the cross-sectional ranking inverted, the failure is a regime-change problem.
2. **Per-asset feature normalization sensitivity** — rebuild the panel with per-asset rolling z-scores for `return_1d/5d/20d/vol_ratio/realized_vol_20`, plus per-date cross-sectional z-scores. Re-run the daily and 5d cases. If linear regression also revives, the previous fragility was scale-driven; if HGB's signal degrades under normalization, the prior signal was leaning on raw scale differences.
3. After (1) and (2), re-run the rebalance sweep on the normalized panel.
4. Only then consider seed sensitivity.

Do **not** pivot away from cross-asset ranking yet. The economic signal at 5d is real (positive mean IR, more consistent fold-level performance, much better cost economics). But it's not yet statistically separable from random allocations of the same turnover profile, and the daily-frequency null pass was probably overstated.

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false` in both runs' metadata.
- `--prepare-missing` not set; no yfinance fetch occurred.
- The static-import test (`test_no_legacy_or_optuna_imports_in_experiment_modules`) continues to enforce the no-legacy-imports invariant.
- No champion manifest was modified.
- No existing result files were overwritten — both runs wrote to fresh per-frequency output directories.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Rebalance-frequency implementation exists | ✓ `apply_rebalance_schedule` + CLI flag |
| 2 | Tests pass | ✓ 4 new tests; 21 cross-asset tests; 148/148 collectible suite |
| 3 | 5d robustness run completes or failure documented | ✓ run completed; null-gate failure documented |
| 4 | 20d robustness run completes or failure documented | ✓ run completed; horizon-mismatch collapse documented |
| 5 | No legacy workflows used | ✓ |
| 6 | No Optuna / deep / stacking / data download | ✓ enforced by metadata + static-import test |
| 7 | Report created | ✓ this file |
