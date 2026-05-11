# Tiny Drawdown Calibration Plumbing Run

## Executive Summary

The standalone drawdown-risk calibration runner executed successfully on the prepared offline SPY feature frame.

This was a plumbing-only run.

It was used to verify:

- the execute path works on real offline input
- the calibration bundle is complete
- artifact columns are present for later analysis
- metadata clearly marks the run as non-decision-grade and classifier-only

This was not a model-validation run and not a trading result.

## Command Used

```bash
python -m scripts.run_drawdown_risk_calibration_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/drawdown_calibration_plumbing_run \
  --base-model-name regularized_linear \
  --calibration-method platt \
  --target-column target_drawdown_event_20d_3pct \
  --asset-name SPY \
  --run-purpose plumbing \
  --train-size 756 \
  --val-size 126 \
  --test-size 126 \
  --step-size 5000 \
  --n-bins 5
```

## Output Bundle

Created files:

- [drawdown_risk_calibration_summary_20260510_195241.csv](/Users/larrym/prediction/results/drawdown_calibration_plumbing_run/drawdown_risk_calibration_summary_20260510_195241.csv)
- [drawdown_risk_calibration_fold_details_20260510_195241.csv](/Users/larrym/prediction/results/drawdown_calibration_plumbing_run/drawdown_risk_calibration_fold_details_20260510_195241.csv)
- [drawdown_risk_calibration_oof_artifacts_20260510_195241.csv](/Users/larrym/prediction/results/drawdown_calibration_plumbing_run/drawdown_risk_calibration_oof_artifacts_20260510_195241.csv)
- [drawdown_risk_calibration_bins_20260510_195241.csv](/Users/larrym/prediction/results/drawdown_calibration_plumbing_run/drawdown_risk_calibration_bins_20260510_195241.csv)
- [drawdown_risk_calibration_report_20260510_195241.md](/Users/larrym/prediction/results/drawdown_calibration_plumbing_run/drawdown_risk_calibration_report_20260510_195241.md)
- [drawdown_risk_calibration_metadata_20260510_195241.json](/Users/larrym/prediction/results/drawdown_calibration_plumbing_run/drawdown_risk_calibration_metadata_20260510_195241.json)

## Output Checks

All expected files were created and parsed successfully.

Observed structure:

- summary CSV
  - `1` row
  - includes raw vs calibrated validation and test metrics
  - includes `mean_test_brier_improvement`

- fold-details CSV
  - `1` row
  - includes:
    - `validation_raw_*`
    - `validation_calibrated_*`
    - `test_raw_*`
    - `test_calibrated_*`

- OOF artifacts CSV
  - `126` rows
  - includes:
    - `prediction_probability`
    - `raw_prediction_probability`
    - `calibrated_prediction_probability`
    - `target`
    - `asset_return`
    - `benchmark_return`
    - `base_model_name`
    - `calibration_method`

- calibration bins CSV
  - `5` rows
  - includes:
    - `bin_id`
    - `count`
    - `mean_predicted_probability`
    - `observed_event_rate`
    - `signed_gap`
    - `abs_gap`

- markdown report
  - clearly states:
    - `Run purpose: plumbing`
    - `Decision grade: False`
    - `Classification only: True`
    - `Economic overlay used: False`

- metadata JSON
  - clearly records:
    - `decision_grade: false`
    - `run_purpose: plumbing`
    - `classification_only: true`
    - `economic_overlay_used: false`
    - `trading_strategy_validated: false`

## Plumbing Result

The execute path is operational.

The standalone calibration runner:

- loaded offline prepared input
- built walk-forward splits
- fit a simple base classifier
- fit a validation-only calibrator
- scored calibrated test probabilities
- wrote a complete timestamped bundle

## Safety Boundaries Confirmed

This run did not use:

- `main.py`
- `prepare_experiment(...)`
- Optuna
- old model-zoo workflows
- ensemble workflows
- HMM/regime workflows
- volatility/HMM overlay workflows
- economic overlays

## Interpretation Boundary

This run should not be interpreted as evidence that calibration works.

It only shows that:

- the standalone calibration experiment executes successfully
- the artifact bundle is complete
- the metadata and report labeling are appropriate for a non-decision-grade plumbing run
