# Patch 13: First Controlled Regime-Overlay Plumbing Run

## Scope

This was a **controlled plumbing run**, not a claim-worthy research result.

Constraints respected:

- no `main.py`
- no `prepare_experiment(...)`
- no model-zoo workflow
- no Optuna
- no data downloads
- standalone runner only
- offline feature preparation from cached local CSVs only

## Offline Input Preparation

The standalone runner currently expects a **prepared feature CSV**, not raw
market OHLCV data.

To avoid any download path, the input frame was built from:

- `data/spy_daily.csv`
- `data/multi_asset_cache/vix_daily.csv`

Benchmark handling:

- `BenchmarkClose` was taken from the local SPY close series
- `benchmark_return_1d` was computed locally

Feature generation used:

- `features.feature_engineering.build_feature_set(...)`
- `advanced_features=True`
- `vix_features=True`

Temporary prepared input:

- `/private/tmp/regime_overlay_spy_feature_frame.csv`

Prepared frame size:

- `3313` rows

## Controlled Run Command

Executed from repo root:

```bash
python -m scripts.run_regime_overlay_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/regime_overlay_plumbing_run \
  --train-size 756 \
  --val-size 126 \
  --test-size 126 \
  --step-size 5000 \
  --target-vol 0.10 \
  --realized-vol-window 20 \
  --transaction-cost-bps 2.0 \
  --null-runs 10 \
  --asset-name SPY \
  --model-name hmm_regime_overlay_hard_veto_plumbing \
  --date-column date \
  --model-type hmm
```

Notes:

- `step-size 5000` was chosen deliberately to force a **single split**
- `null-runs 10` was chosen deliberately to keep this as a plumbing check

## Output Files

Generated without overwriting existing results:

- `results/regime_overlay_plumbing_run/regime_overlay_experiment_summary_20260508_193537.csv`
- `results/regime_overlay_plumbing_run/regime_overlay_experiment_audit_artifacts_20260508_193537.csv`

## Run Outcome

The standalone runner completed successfully.

Operational checks passed:

- train-only regime fitting executed
- validation/test used the standalone overlay workflow
- timestamped outputs were written
- overlay audit artifacts included explicit:
  - `executed_position`
  - `base_position`
  - `dangerous_regime_id`
  - `dangerous_regime_probability`
  - `overlay_threshold`
  - `overlay_risk_multiplier`
  - `overlay_mode`
  - `is_cut_day`

Observed summary row:

- `dangerous_regime_id = 1`
- `threshold = 0.5`
- `risk_multiplier = 0.5`
- `base_information_ratio = 0.320590`
- `overlay_information_ratio = 0.335894`
- `overlay_vs_base_information_ratio = 0.068381`
- `overlay_vs_base_active_calmar = 0.105660`
- `overlay_vs_base_annualized_active_return = 0.001531`
- `same_average_exposure_p_value = 0.6`
- `same_turnover_p_value = 0.6`
- `same_regime_exposure_p_value = 0.5`

## Interpretation

This run should be interpreted as:

- **plumbing success**
- **not statistical success**

Why:

- only one split
- only 10 null draws
- no robustness claim should be made from this run
- matched-null p-values were not significant

That means the overlay path is now operational, but it has **not** yet passed
the reset-plan success gate.

## Immediate Next Step

Before any broader claim:

1. run a larger controlled overlay experiment with more splits
2. increase null-run budget materially
3. inspect whether HMM live-safe probabilities remain numerically stable across
   more folds
4. judge success only on active metrics and matched-null p-values
