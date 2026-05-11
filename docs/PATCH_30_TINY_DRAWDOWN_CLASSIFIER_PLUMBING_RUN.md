# Tiny Drawdown-Risk Classifier Plumbing Run

## Executive Summary

The standalone drawdown-risk classifier runner executed successfully on the offline prepared SPY feature frame.

This was a plumbing run only.

It verified:

- the standalone execute path works on real offline input
- the target `target_drawdown_event_20d_3pct` can be appended and consumed correctly
- the expected output bundle is created
- the CSV, markdown, and JSON outputs parse cleanly

This was not a decision-grade experiment.

No old workflows were triggered.

## Input Check

Confirmed input existed:

- `/private/tmp/regime_overlay_spy_feature_frame.csv`

## Command Used

```bash
python -m scripts.run_drawdown_risk_classifier_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/drawdown_classifier_plumbing_run \
  --date-column date \
  --price-column "Adj Close" \
  --target-column target_drawdown_event_20d_3pct \
  --model-name logistic \
  --asset-name SPY \
  --train-size 756 \
  --val-size 126 \
  --test-size 126 \
  --step-size 5000
```

## Output Bundle

Created under [drawdown_classifier_plumbing_run](/Users/larrym/prediction/results/drawdown_classifier_plumbing_run):

- [drawdown_risk_classifier_summary_20260510_185308.csv](/Users/larrym/prediction/results/drawdown_classifier_plumbing_run/drawdown_risk_classifier_summary_20260510_185308.csv)
- [drawdown_risk_classifier_fold_details_20260510_185308.csv](/Users/larrym/prediction/results/drawdown_classifier_plumbing_run/drawdown_risk_classifier_fold_details_20260510_185308.csv)
- [drawdown_risk_classifier_oof_artifacts_20260510_185308.csv](/Users/larrym/prediction/results/drawdown_classifier_plumbing_run/drawdown_risk_classifier_oof_artifacts_20260510_185308.csv)
- [drawdown_risk_classifier_report_20260510_185308.md](/Users/larrym/prediction/results/drawdown_classifier_plumbing_run/drawdown_risk_classifier_report_20260510_185308.md)
- [drawdown_risk_classifier_metadata_20260510_185308.json](/Users/larrym/prediction/results/drawdown_classifier_plumbing_run/drawdown_risk_classifier_metadata_20260510_185308.json)

## Parse and Schema Checks

### Summary CSV

- parsed successfully
- row count: `1`
- contained expected aggregate classifier fields, including:
  - `model_name`
  - `target_column`
  - `n_splits`
  - `mean_test_auc_roc`
  - `mean_validation_auc_roc`
  - `mean_test_brier_score`
  - `positive_test_auc_folds`
  - `std_test_auc_roc`

### Fold Details CSV

- parsed successfully
- row count: `1`
- contained separate validation and test metrics:
  - `validation_auc_roc`
  - `validation_brier_score`
  - `validation_positive_prediction_rate`
  - `test_auc_roc`
  - `test_brier_score`
  - `test_positive_prediction_rate`

### OOF Artifacts CSV

- parsed successfully
- row count: `126`
- contained the required future-analysis fields:
  - `date`
  - `split_id`
  - `model_name`
  - `asset`
  - `asset_return`
  - `benchmark_return`
  - `raw_signal`
  - `prediction_probability`
  - `target`
  - `executed_position`
  - `strategy_gross_return`
  - `strategy_net_return`
  - `turnover`
  - `transaction_cost`

Notes:

- this is still a classifier-only experiment
- `executed_position` is present because the standardized audit schema includes it, but no economic overlay evaluation was run

### Markdown Report

- parsed successfully
- clearly identifies:
  - model: `logistic`
  - target: `target_drawdown_event_20d_3pct`
  - splits: `1`
- includes notes:
  - `Standalone simple classifier workflow only.`
  - `No economic overlay evaluation was run.`

### Metadata JSON

- parsed successfully
- includes:
  - `model_name`
  - `target_column`
  - `num_splits`
  - `feature_count`
  - notes confirming this is classifier-only and not an overlay run

## Safety Boundary Check

The generated report and metadata did not indicate use of:

- `main.py`
- `prepare_experiment(...)`
- Optuna
- ensemble workflows
- regime workflows
- overlay workflows

This is consistent with the intended isolation of the standalone classifier runner.

## Reporting Gap

The outputs are clearly classifier-only, but they do not yet explicitly label the run as:

- plumbing
- non-decision-grade

This is a reporting gap, not an execution failure.

It does not affect the purpose of this run, which was only bundle verification.

## Tiny Diagnostic Snapshot

These values are included only to confirm the runner populated the outputs correctly. They are not a research conclusion:

- `mean_test_auc_roc = 0.570292`
- `mean_validation_auc_roc = 0.353086`
- `mean_test_brier_score = 0.259034`
- `mean_test_positive_prediction_rate = 0.626984`

## Conclusion

The standalone drawdown-risk classifier execute path is operational on real offline SPY input.

The output bundle is complete and parseable.

No old workflows were triggered.

This was a plumbing success only, not a model-validation result.
