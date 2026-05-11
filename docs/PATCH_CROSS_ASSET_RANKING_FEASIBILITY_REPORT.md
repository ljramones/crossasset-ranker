# Patch: Cross-Asset Ranking Feasibility Prototype

Date: 2026-05-10

## Summary

Built a standalone cross-asset ranking prototype that scores forward 20-day risk-adjusted return across SPY/QQQ/IWM/TLT/GLD/BTC-USD, allocates equal-weight to the top-1 and top-2 assets per day, and compares against equal-weight and random top-k null baselines. Ran one tiny plumbing-grade execution end-to-end on cached data only.

This patch produces *infrastructure*, not evidence. The numbers below come from a single walk-forward split with 10 random nulls — they are not interpretable as alpha or as a falsification.

## Files created

- `evaluation/cross_asset_ranking.py` — pure helpers: forward returns, trailing realized vol, risk-adjusted target, panel builder, cross-sectional ranks, feature selector, top-k allocation, allocation-to-returns translator, equal-weight baseline, random top-k null generator, smoothed empirical p-value.
- `experiments/cross_asset_ranking_experiment.py` — `CrossAssetRankingConfig` dataclass, asset-frame loader (refuses to fetch by default), walk-forward splitter over panel dates, model fit/score loop for `momentum_baseline` / `linear_regression` (Ridge) / `hist_gradient_boosting` (HistGradientBoosting), per-fold metrics via `evaluation.metrics.compute_return_stream_metrics`, per-fold random null IR distribution, p-value join.
- `scripts/run_cross_asset_ranking_experiment.py` — argparse CLI with `--dry-run` / `--execute`, `--prepare-missing` guard against accidental fetches, output bundle writer (9 timestamped files), markdown report generator.
- `tests/test_cross_asset_ranking.py` — 11 helper tests.
- `tests/test_cross_asset_ranking_experiment.py` — 6 runner/CLI tests.
- `docs/PATCH_CROSS_ASSET_RANKING_FEASIBILITY_REPORT.md` — this file.

## Files modified

Two small follow-on fixes surfaced when the prototype was executed against the real cache for the first time:

- `data/market_cache.py` (no behavior change in v2): an attempted forward-fill of `BenchmarkClose` was reverted after it caused all benchmark-derived rolling features (e.g. `relative_strength_vs_benchmark`) to go NaN for cross-asset days outside the equity calendar. The cache stays honest — only `VIXClose` is forward-filled (per the original spec).
- `scripts/prepare_feature_frame.py`: now drops rows with NaN `BenchmarkClose` before feature engineering. This restricts off-calendar assets like BTC-USD to the benchmark's native trading calendar and produces a non-empty prepared frame for cross-asset use. Documented inline in the file.

These two changes are tested by `tests/test_market_cache.py` and `tests/test_prepare_feature_frame.py` (32 of the 32 tests for these modules pass, including the BTC-style asymmetric-calendar case).

## Tests run and results

```text
uv run python -m pytest tests/test_cross_asset_ranking.py tests/test_cross_asset_ranking_experiment.py -x -q
... 17 passed in 4.33s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 144 passed in 6.70s
```

The single ignored collection (`tests/test_integrity_audit_matched_nulls.py`) is the same pre-existing legacy-import failure documented in `docs/PATCH_DATA_REBUILD_ACTIVE_TRACK_REPORT.md`. No regressions.

## Dry-run command and result

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment \
  --dry-run \
  --assets SPY QQQ IWM TLT GLD BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_feasibility \
  --forward-horizon 20 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 5000 \
  --transaction-cost-bps 2.0 \
  --top-k 1 2 \
  --random-null-runs 10 \
  --random-state 42 \
  --run-purpose plumbing
```

Result: prints the resolved config, fetches no data, writes no files. The dry-run safeguard test (`test_cli_dry_run_does_not_load_data`) passes.

## Tiny execute command and result

Same arguments with `--dry-run` replaced by `--execute`.

Per-asset prepared frames (loaded from existing raw cache; no yfinance calls):

```text
SPY        rows=  3315   2010-06-11 -> 2026-05-07
QQQ        rows=  3346   2010-04-28 -> 2026-05-07
IWM        rows=  3346   2010-04-28 -> 2026-05-07
TLT        rows=  3346   2010-04-28 -> 2026-05-07
GLD        rows=  3346   2010-04-28 -> 2026-05-07
BTC-USD    rows=  2452   2015-01-09 -> 2026-05-07
```

The cross-asset panel inner-joins to dates present in all six assets, which restricts the panel to BTC-USD's start-plus-warmup (~2015-01-09) onward. With `train=756, val=252, test=252, step=5000`, exactly **one walk-forward split** is created — that is the intent of `step=5000` for a plumbing run.

## Output files

```text
results/cross_asset_ranking_feasibility/
  cross_asset_ranking_summary_20260510T223056Z.csv          (1.2 KB)
  cross_asset_ranking_fold_details_20260510T223056Z.csv     (1.6 KB)
  cross_asset_ranking_scored_panel_20260510T223056Z.csv     (262 KB)
  cross_asset_ranking_allocations_20260510T223056Z.csv      (697 KB)
  cross_asset_ranking_portfolio_returns_20260510T223056Z.csv (155 KB)
  cross_asset_ranking_random_nulls_20260510T223056Z.csv     (9.3 KB)
  cross_asset_ranking_null_pvalues_20260510T223056Z.csv     (474 B)
  cross_asset_ranking_report_20260510T223056Z.md            (4.2 KB)
  cross_asset_ranking_metadata_20260510T223056Z.json        (2.2 KB)
```

## Preliminary plumbing-only metrics

**Do not interpret these as evidence.** One split, 252 test dates, 10 random nulls.

```text
                 model  top_k  net_sharpe  information_ratio  ann_active  active_calmar  max_dd  turnover
          equal_weight      6        1.55               0.00        0.00           0.00   -0.28      0.00
     momentum_baseline      1        2.37               1.81        1.00           4.26   -0.28      0.05
     momentum_baseline      2        2.68               1.61        0.50           3.35   -0.14      0.03
hist_gradient_boosting      1        2.48               1.06        0.31           2.29   -0.15      0.07
hist_gradient_boosting      2        3.37               1.72        0.41           5.39   -0.12      0.06
     linear_regression      1       -0.19              -1.25       -0.53          -1.04   -0.62      0.03
     linear_regression      2        0.21              -1.81       -0.35          -1.10   -0.42      0.03

Random top-k null p-values (n_runs = 10):
  momentum_baseline      top_1: p = 0.091
  momentum_baseline      top_2: p = 0.091
  hist_gradient_boosting top_1: p = 0.182
  hist_gradient_boosting top_2: p = 0.091
  linear_regression      top_1: p = 0.818
  linear_regression      top_2: p = 0.909
```

p-values floor at `(1+0)/(10+1) = 0.091`. With 10 nulls the test cannot distinguish a real signal from a lucky model — **this is a plumbing diagnostic only**. Interpretation requires `--random-null-runs >= 200` and multiple independent splits via a smaller `--step-size`.

That said, the plumbing run does verify two non-obvious things:
1. The **equal-weight baseline** earns a respectable net Sharpe (~1.55) on this universe over this window. Beating it on IR is non-trivial — a model that just tilts toward the top-performing asset post-hoc isn't enough; it has to time which asset is best forward.
2. The **linear regression model is actively underperforming** the random null. Either the linear hypothesis is too weak for cross-asset risk-adjusted return, or the feature scales need per-asset normalization, or there's a leakage-free cross-asset feature missing. This is exactly the kind of finding that decision-grade runs are for.

## Confirmation: scope safeties held

`metadata.json` for the run records:

```json
{
  "main_py_used": false,
  "prepare_experiment_used": false,
  "old_model_zoo_used": false,
  "optuna_used": false,
  "deep_models_used": false,
  "stacking_used": false,
  "decision_grade": false,
  "run_purpose": "plumbing",
  "lag_convention": "weights.shift(1) * returns — weights at close of t apply to returns of t+1",
  "calendar_convention": "strict inner-join across asset date sets",
  "policy": "daily reallocation; equal weight among top-k by score"
}
```

A static-import test (`test_no_legacy_or_optuna_imports_in_experiment_modules`) enforces these in the source.

- No `import optuna`, `import lightning`, `from neuralforecast`, `from pytorch_forecasting`.
- No `from data.market_data`, `from utils.experiment`, `from audit.integrity_audit`.
- No `regime_stacking`.
- No fetch occurred during `--execute` (cache was already on disk; `--prepare-missing` was *not* set).

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Cross-asset ranking utilities exist | ✓ `evaluation/cross_asset_ranking.py` |
| 2 | Standalone experiment module exists | ✓ `experiments/cross_asset_ranking_experiment.py` |
| 3 | CLI wrapper exists | ✓ `scripts/run_cross_asset_ranking_experiment.py` |
| 4 | Synthetic tests pass | ✓ 17/17 new; 144/144 collectible suite |
| 5 | Dry-run succeeds without data load | ✓ |
| 6 | Tiny execute run succeeds using cached data only | ✓ no yfinance calls |
| 7 | Output bundle is generated | ✓ 9 timestamped files |
| 8 | No old workflows used | ✓ static-import test enforces |
| 9 | No Optuna used | ✓ static-import test enforces |
| 10 | No deep / stacking models used | ✓ static-import test enforces |
| 11 | No data download occurred | ✓ `--prepare-missing` not set |
| 12 | Documentation report exists | ✓ this file |

## Recommended next step

If the plumbing run is judged clean, the next task is a **decision-grade cross-asset ranking run**:

- `--step-size 252` to produce multiple non-overlapping test splits across the panel history (~10 splits).
- `--random-null-runs 500` to give p-values meaningful resolution below 0.01.
- `--run-purpose decision_grade --decision-grade` to mark the resulting metadata accordingly.
- Add a stop/go criteria block to the report writer that flags PASS only if (a) the model beats equal-weight on net IR, (b) the empirical p-value vs random top-k is < 0.05 across a majority of splits, (c) the result is not driven only by BTC-USD selection, (d) at least 3 of 6 assets are selected meaningfully over time, and (e) turnover is reasonable.

Open questions to resolve in the decision-grade design before running:

- **Per-asset feature normalization.** Right now features are pooled raw across assets — BTC's volatility scale dominates. Cross-asset z-scoring per date (already implemented for some advanced features) might be a cleaner input form.
- **Rebalance frequency.** Daily reallocation drives turnover up; a 5- or 20-day rebalance cadence is worth comparing for cost robustness.
- **Cash-out option.** When all model scores are below zero, the policy currently still allocates top-k. A "cash if all negative" variant might be worth a sensitivity sweep.

These are notes for the *next* design pass, not changes to make in this patch.
