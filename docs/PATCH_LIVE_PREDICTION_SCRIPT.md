# Live Prediction Script — Implementation, Validation, Operational Workflow

Created 2026-05-11 as the deployment infrastructure following the empirical close of the cross-asset ranking campaign documented in `RIDGE_BASELINE_RESULTS.md` and `PATCH_CROSS_ASSET_RANKING_20D_HORIZON.md`.

## Scope

`scripts/run_live_prediction.py` is a pure orchestration layer over the existing pipeline. It does not introduce new feature engineering, new model code, or new metrics. Every transformation is imported from:

- `experiments.cross_asset_ranking_experiment` — `_build_panel`, `_score_with_lambdarank`, `load_prepared_asset_frames`, `target_column_for_horizon`, `CrossAssetRankingConfig`
- `evaluation.cross_asset_ranking` — `select_cross_asset_feature_columns`, `normalize_features_per_asset_train_only`, `build_top_k_allocations`

The script provides three modes of use:

1. **Live mode** — predict the top-k pick for today (or any past as-of date) using a 1008-day rolling training window ending strictly before the as-of date.
2. **Replay mode** — emit a sequence of predictions across a date range, stepping by the profile's rebalance horizon. Used to build a paper-trade forward-walk track record.
3. **Dry-run** — print the resolved configuration and exit. No data load, no fit, no writes.

The script does not execute trades, call brokerage APIs, or integrate with any external trading infrastructure. The user reviews stdout and JSON output and executes manually.

## Profile definitions

Four named profiles, all LambdaRank. Ridge profiles are intentionally absent — Ridge underperformed the trivial baseline at both horizons (5d Spearman −0.030, 20d −0.055; ICIR −2.2 and −1.6 respectively), per `RIDGE_BASELINE_RESULTS.md`.

| Profile | Horizon | Top-k | XS features | Regime interactions | Description |
| --- | ---: | ---: | --- | --- | --- |
| `v3_5d_top2` | 5d | 2 | ✓ | ✓ | Economic high-water mark (Sharpe +1.18, IR +0.748) |
| `20d_top1` | 20d | 1 | ✓ | — | Cost-efficient alternative (lowest turnover) |
| `20d_top2` | 20d | 2 | ✓ | — | Concentration hedge of 20d_top1 |
| `v1_5d_top2` | 5d | 2 | ✓ | — | Rank-quality reference |

All profiles share: 18-asset universe (`SPY, QQQ, IWM, DIA, EFA, EEM, TLT, IEF, SHY, LQD, HYG, GLD, SLV, USO, DBA, UUP, VNQ, BTC-USD`), per-asset train z-score normalization, LightGBM `LGBMRanker` with the campaign's pinned hyperparameters (100 estimators, lr=0.05, num_leaves=15, min_child_samples=5, random_state=42), 1008-day training window.

Each profile emits a 12-char `signature` hash of its deployment-relevant fields (name, horizon, top-k, feature flags, normalization, model, universe, training-window length). Any change to those fields produces a new signature so prediction-log analysis can cleanly cohort by configuration.

**Constraint on future model substitution:** Any model replacing LambdaRank must respect per-date grouping (pairwise rank loss constrained within each date's universe). Pooled regression on per-asset-normalized features fails the cross-date sign-coherence problem documented in the Ridge baseline. Acceptable alternatives include other rankers with explicit group masking (e.g. XGBoost `rank:pairwise` with `group` param), neural pairwise rankers with within-date grouping, or any model trained on features that are already cross-sectionally comparable (rank-based or per-date z-score).

## Causal data handling (no-look-ahead)

For any prediction at as-of date `D`:

1. **Frame truncation.** Each asset's prepared feature frame is sliced to rows where `date <= D` before any pipeline call. Done in `_truncate_frames` and applied per replay date in replay mode.
2. **Panel construction.** `build_cross_asset_panel` is called on the truncated frames, so forward returns for `t > D-H` come back as NaN (their forward window includes dates not in the truncated panel).
3. **Feature normalization.** `normalize_features_per_asset_train_only` fits the per-asset (mean, std) on the explicit `train_dates` argument (the 1008 trailing trading days strictly before `D`); the same statistics are applied to all rows including the live date. This is the same operation the campaign experiment runner used for its per-fold normalization, but with the live training window passed explicitly.
4. **Training row filter.** `_fit_lambdarank_on_panel` drops rows where the target is NaN. Combined with frame truncation, this means training rows are exactly those whose forward target was *knowable* at as-of date `D` — i.e. rows where `t + H ≤ D`.
5. **VIX z-score warmup.** `add_vix_zscore_to_panel` uses a 252-day rolling window. The data load is sized to include enough prior history for the warmup to be valid throughout the training window.

The strict-inner-join calendar of `build_cross_asset_panel` means the live date is honored only if every asset has data through `D`. If the user requests an as-of date for which some assets are missing the latest bar (e.g. yfinance cache hasn't yet been refreshed), the script falls back to the most recent date present in *all* asset frames and reports the substitution on stdout.

## Validation against the experiment runner

The script's deployment-time training pattern (1008-day rolling window, panel truncated to live date) intentionally diverges from the campaign experiment runner (756-day static train per fold, panel built over all dates). To catch implementation drift independent of that divergence, the pipeline components are validated by reproducing the runner's behavior using the same imported functions.

**Validation procedure** (one-off, executed 2026-05-11):

Using the same imports the live script uses — `_build_panel`, `select_cross_asset_feature_columns`, `normalize_features_per_asset_train_only`, `_score_with_lambdarank` — replicate the campaign runner's fold-4 setup for the `20d_top1` configuration:
- 756-day training window aligned to the runner's fold-4 train dates (`2020-02-18` → `2023-05-12`)
- Panel built over all available data (no truncation)
- Predict for fold-4's 252 test dates (`2024-08-12` → `2025-08-13`)
- Compare per-date scores to `results/cross_asset_ranking_20d_target_lambdarank/cross_asset_ranking_scored_panel_*.csv`

**Result:**

| Metric | Value |
| --- | --- |
| (date, asset) rows compared | 4,536 |
| Mean absolute score difference | **0.000000** |
| Maximum absolute score difference | **0.000000** |
| Score correlation | **1.000000** |
| Per-date top-1 pick match | **252 / 252 (100.0%)** |

The pipeline components produce numerically identical results to the campaign experiment runner when given the same training setup. Any divergence observed in live-mode deployment use comes purely from the intentional design choices (1008-day rolling vs 756-day static, truncated panel vs full panel), not from implementation drift.

A practical consequence: predicting in **live mode** at the first date of a campaign fold's test window does *not* match the runner's per-fold pick on that date. For example, predicting `20d_top1` at `2024-08-12` (fold-4 start) produces `UUP`, while the campaign runner picked `GLD`. Both are correct under their respective designs; the difference is that live mode has 252 extra training days (the val window the campaign held out) and trains on a panel that doesn't include any post-as-of-date forward targets.

## Output format

Each prediction produces three artifacts:

1. **Human-readable stdout summary** — header line, picks with scores and weights, full 18-asset ranking, top-10 feature-importance entries, training window dates and sample count, cache freshness, NaN-feature diagnostics.

2. **Per-prediction JSON** at `predictions/<YYYY-MM-DD>_<profile_name>.json`. Schema (from `PredictionRecord` dataclass):

   - `as_of_date`, `profile_name`, `profile_signature`, `forward_horizon`, `top_k`
   - `universe`: list of 18 asset tickers
   - `scores`: `{asset: score}` for all 18 assets
   - `top_k_picks`: list of `{asset, score, rank, weight}`
   - `feature_importance`: top-10 list of `{feature, gain, split_count}`
   - `training_window_start`, `training_window_end`, `training_n_samples`, `training_n_dates`
   - `cache_last_bar_date`, `fetched_fresh_data`
   - `nan_feature_diagnostics`: `{feature_name: nan_count}` (omitted features mean clean)
   - `feature_values_at_live_date`: `{asset: {feature: value}}` — normalized values that fed the model on this date
   - `generated_at_utc`

3. **Master log** at `predictions/predictions_log.jsonl`. Each prediction appends one line (same payload as the per-prediction JSON, single-line). Build the forward-walk track record by reading this file over time.

## Usage examples

```bash
# Dry-run — validate config without loading data
uv run python -m scripts.run_live_prediction --dry-run --profile 20d_top1

# Live mode — predict for today (uses latest available cached trading day)
uv run python -m scripts.run_live_prediction --execute --profile v3_5d_top2 --as-of-date today

# Live mode with fresh-data fetch — fill in any missing asset caches via yfinance
uv run python -m scripts.run_live_prediction --execute --profile 20d_top1 --as-of-date today --fetch-fresh-data

# Predict for a specific past date (single prediction)
uv run python -m scripts.run_live_prediction --execute --profile v3_5d_top2 --as-of-date 2026-03-15

# Historical replay — build a forward-walk track record
uv run python -m scripts.run_live_prediction --execute --profile 20d_top1 \
    --mode replay --start-date 2025-01-15 --end-date 2026-04-15
```

## Operational workflow

**When to run.** Run on the rebalance cadence of the chosen profile (5 trading days for `v3_5d_top2` / `v1_5d_top2`, 20 trading days for `20d_top1` / `20d_top2`). Run after the prior bar's close (i.e. after market close on day `D` to get a prediction for day `D` to be acted on day `D+1`).

**What to do with output.** The stdout block names the top-k assets to be held until the next rebalance. The JSON record is a permanent audit trail; the JSONL log is the running forward-walk track. Live execution is manual — the user places the orders themselves at a broker of their choice. The script does not call any brokerage API.

**How to track over time.** Tail `predictions/predictions_log.jsonl` after each run. Compare prediction-vs-realized outcomes by reading the log alongside actual return data once the forward horizon has elapsed. The realized-PnL comparison code is not in scope for this patch; build it as a separate analysis script when there is enough track-record to evaluate.

**Cache freshness.** `--fetch-fresh-data` allows the underlying loader to fetch *missing* asset caches via yfinance (using `prepare_missing=True` on `load_prepared_asset_frames`). It does **not** force a hard refresh of already-cached data. For a full cache refresh, run `scripts/prepare_feature_frame.py --execute --asset <ticker>` for each asset before invoking this script. The script reports the cache's last-bar date as `cache_last_bar_date` in every prediction record so the user can see at a glance whether the data is stale.

## Limitations

- **Prediction service, not trading service.** The script outputs picks; the user places orders. No order routing, no slippage modeling beyond the campaign's static 2 bps cost assumption (which lives in `CrossAssetRankingConfig`, not in the live prediction path — costs only enter the picture when the user computes realized PnL against the predictions).
- **Cache refresh limitation.** `--fetch-fresh-data` triggers fill-in-missing only, not hard refresh. If the cache is out of date, predictions will be made on whatever data is present.
- **Live mode != campaign per-fold.** Live picks intentionally diverge from the campaign runner due to the 1008-vs-756 window difference and the truncated-vs-full panel difference. See the validation section above.
- **Universe is fixed.** Adding or removing an asset breaks comparability with the campaign artifacts and invalidates the profile signatures. Any universe change is a campaign-level decision, not a script flag.
- **Ridge models excluded.** Ridge profiles are not selectable. The campaign showed Ridge actively miscoded cross-sectional structure (Spearman below the trivial baseline at both horizons). Re-enabling Ridge would require new empirical evidence — typically that the features being used are cross-sectionally comparable by construction, which the current per-asset-normalized feature stack is not. See `docs/RIDGE_BASELINE_RESULTS.md` Section "Addendum — mechanism refinement and deferred follow-up" for the deferred xs_rank-only Ridge test that would change this picture.

## Risk surface review

The script's safety properties, mapped to specific guards:

| Risk | Guard | Where |
| --- | --- | --- |
| Future leakage via raw frames | `_truncate_frames` slices each frame to `<= as_of_date` before panel construction | `_truncate_frames`, called in both live and replay paths |
| Future leakage via normalization | `normalize_features_per_asset_train_only` takes explicit `train_dates`; stats fit only on those rows | `_predict_for_as_of` passes the 1008-day window |
| Future leakage via VIX z-score | 252-day rolling window is causal (uses [t−251, t] including current bar, known at close) | Built inside `_build_panel` via `add_vix_zscore_to_panel` |
| Future leakage via target | Forward targets are computed but only used for training; live-date row has NaN target by construction | `_fit_lambdarank_on_panel` drops NaN-target rows |
| Stale cache silently used | `cache_last_bar_date` recorded in every prediction record and printed on stdout | `_predict_for_as_of`, `_format_stdout` |
| Profile drift over time | `profile.signature()` includes universe + training-window length; any change yields new signature | `Profile.signature` |
| Implementation drift from pipeline | Pipeline components validated to match the campaign runner numerically (100% top-1 match over 252 test dates) | Validation section above |
