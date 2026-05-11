# Tiny Volatility-Quantile Overlay Bundle Verification Run

## Executive Summary

This task executed one tiny offline standalone volatility-quantile overlay run on the prepared cached SPY feature frame to verify bundle creation only.

This was not a decision-grade experiment.

The run succeeded and produced the full normalized output bundle:

- summary CSV
- fold-details CSV
- audit-artifacts CSV
- matched-nulls CSV
- markdown report
- metadata JSON

The generated outputs parsed successfully, contained the expected columns, and clearly labeled the run as diagnostic only:

- `Null runs: 5`
- `Decision grade: False`

No old workflows were triggered. This run did not call:

- `main.py`
- `prepare_experiment(...)`
- Optuna
- regime-stacking workflows
- HMM/GMM regime-overlay workflows

## Offline Input Check

Confirmed input file existed before execution:

- `/private/tmp/regime_overlay_spy_feature_frame.csv`

## Command Used

```bash
python -m scripts.run_vol_quantile_overlay_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/vol_quantile_bundle_verification_run \
  --train-size 756 \
  --val-size 126 \
  --test-size 126 \
  --step-size 5000 \
  --target-vol 0.10 \
  --realized-vol-window 20 \
  --transaction-cost-bps 2.0 \
  --null-runs 5 \
  --asset-name SPY \
  --model-name vol_quantile_overlay_bundle_check \
  --date-column date
```

## Output Bundle

Timestamped files created under [results/vol_quantile_bundle_verification_run](/Users/larrym/prediction/results/vol_quantile_bundle_verification_run):

- [vol_quantile_overlay_experiment_summary_20260510_151215.csv](/Users/larrym/prediction/results/vol_quantile_bundle_verification_run/vol_quantile_overlay_experiment_summary_20260510_151215.csv)
- [vol_quantile_overlay_experiment_fold_details_20260510_151215.csv](/Users/larrym/prediction/results/vol_quantile_bundle_verification_run/vol_quantile_overlay_experiment_fold_details_20260510_151215.csv)
- [vol_quantile_overlay_experiment_audit_artifacts_20260510_151215.csv](/Users/larrym/prediction/results/vol_quantile_bundle_verification_run/vol_quantile_overlay_experiment_audit_artifacts_20260510_151215.csv)
- [vol_quantile_overlay_experiment_matched_nulls_20260510_151215.csv](/Users/larrym/prediction/results/vol_quantile_bundle_verification_run/vol_quantile_overlay_experiment_matched_nulls_20260510_151215.csv)
- [vol_quantile_overlay_experiment_report_20260510_151215.md](/Users/larrym/prediction/results/vol_quantile_bundle_verification_run/vol_quantile_overlay_experiment_report_20260510_151215.md)
- [vol_quantile_overlay_experiment_metadata_20260510_151215.json](/Users/larrym/prediction/results/vol_quantile_bundle_verification_run/vol_quantile_overlay_experiment_metadata_20260510_151215.json)

## Parse and Schema Checks

### Summary CSV

- parsed successfully
- row count: `1`
- contained the expected top-level diagnostics, including:
  - `base_information_ratio`
  - `overlay_information_ratio`
  - `overlay_vs_base_information_ratio`
  - `overlay_vs_base_active_calmar`
  - `overlay_vs_base_annualized_active_return`
  - `same_average_exposure_p_value`
  - `same_turnover_p_value`
  - `same_vol_state_exposure_p_value`

### Fold Details CSV

- parsed successfully
- row count: `1`
- contained validation-selection fields and full test metrics
- included explicit overlay-vs-base fields and position-change diagnostics

### Audit Artifacts CSV

- parsed successfully
- row count: `126`
- contained required overlay artifact columns:
  - `date`
  - `split_id`
  - `model_name`
  - `asset`
  - `asset_return`
  - `benchmark_return`
  - `executed_position`
  - `strategy_gross_return`
  - `strategy_net_return`
  - `turnover`
  - `transaction_cost`
  - `base_position`
  - `vol_state`
  - `high_vol_indicator`
  - `vol_quantile`
  - `vol_cutoff_value`
  - `overlay_threshold_state`
  - `overlay_risk_multiplier`
  - `overlay_mode`
  - `is_cut_day`

Observed run-specific values:

- `overlay_mode = vol_quantile_hard_veto`
- `model_name = vol_quantile_overlay_bundle_check`
- `split_id = 0`

### Matched Nulls CSV

- parsed successfully
- row count: `5`
- contained one row for each null family:
  - `same_average_exposure_random`
  - `same_turnover_random`
  - `same_exposure_and_turnover_random`
  - `same_vol_state_exposure_random`
  - `block_bootstrap_same_exposure_random`

### Markdown Report

- parsed successfully
- clearly labeled:
  - `Null runs: 5`
  - `Decision grade: False`
- included sections for:
  - summary
  - fold details
  - matched null diagnostics
  - notes

### Metadata JSON

- parsed successfully
- contained:
  - `decision_grade: false`
  - `null_runs: 5`
  - `mode: execute`
  - `n_splits: 1`
- notes explicitly stated:
  - standalone volatility-quantile overlay workflow only
  - do not treat results as validated unless matched-null gates pass at higher null-run budgets

## Safety Boundary Check

The generated report and metadata did not reference old workflow entry points. A text scan found no matches for:

- `prepare_experiment`
- `main.py`
- `Optuna`
- `stacking_ensemble`
- `regime_stacking`

This is consistent with the intended isolation of the standalone volatility-quantile runner.

## Tiny Diagnostic Snapshot

These values are included only to confirm the output bundle populated correctly. They are not a research conclusion:

- `overlay_vs_base_information_ratio = -0.040076`
- `same_average_exposure_p_value = 0.6`
- `same_turnover_p_value = 0.6`
- `same_vol_state_exposure_p_value = 0.4`

## Conclusion

The standalone volatility-quantile overlay runner executed successfully on real offline prepared input and produced the expected normalized output bundle.

This was a plumbing verification success, not a strategy validation result.

The next safe step, if desired, is a larger offline decision-grade volatility-quantile overlay run with a materially higher null budget and multiple splits.
