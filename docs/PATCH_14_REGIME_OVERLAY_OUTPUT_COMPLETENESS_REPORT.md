# Patch 14: Regime-Overlay Output Completeness Report

## Inspection of Existing Plumbing Outputs

### `results/regime_overlay_plumbing_run/regime_overlay_experiment_summary_20260508_193537.csv`

```text
file:
  exists: true
  rows: 1
  columns:
    - split_id
    - dangerous_regime_id
    - threshold
    - risk_multiplier
    - base_information_ratio
    - overlay_information_ratio
    - overlay_vs_base_information_ratio
    - overlay_vs_base_active_calmar
    - overlay_vs_base_annualized_active_return
    - fraction_in_market
    - daily_turnover
    - same_average_exposure_p_value
    - same_turnover_p_value
    - same_regime_exposure_p_value
  contains_fold_details: false
  contains_matched_null_rows: true
  contains_stop_go_decision: false
  notes:
    - contains only one summary row
    - matched-null information is compressed into p-value columns only
    - does not expose validation selection details or full null distribution summaries
```

### `results/regime_overlay_plumbing_run/regime_overlay_experiment_audit_artifacts_20260508_193537.csv`

```text
file:
  exists: true
  rows: 126
  columns:
    - date
    - split_id
    - model_name
    - asset
    - asset_return
    - benchmark_return
    - raw_signal
    - prediction_probability
    - target
    - executed_position
    - strategy_gross_return
    - strategy_net_return
    - turnover
    - transaction_cost
    - regime_id
    - base_position
    - dangerous_regime_id
    - dangerous_regime_probability
    - overlay_threshold
    - overlay_risk_multiplier
    - overlay_mode
    - is_cut_day
    - regime_prob_0
    - regime_prob_1
    - regime_prob_2
  contains_fold_details: false
  contains_matched_null_rows: false
  contains_stop_go_decision: false
  notes:
    - good per-row audit artifact
    - not a substitute for split-level evaluation or null-summary tables
```

### `docs/PATCH_13_FIRST_CONTROLLED_REGIME_OVERLAY_PLUMBING_RUN.md`

```text
file:
  exists: true
  rows: n/a
  columns: n/a
  contains_fold_details: true
  contains_matched_null_rows: true
  contains_stop_go_decision: false
  notes:
    - human-readable run note only
    - not machine-readable
    - useful context, but not a standardized experiment artifact bundle
```

## Gap Summary

The initial plumbing run produced:

- summary CSV
- audit artifact CSV
- manual markdown note

It did **not** produce separate:

- fold-details CSV
- matched-null results CSV
- standardized experiment markdown report
- machine-readable run metadata JSON

## Patch Outcome

The standalone runner now emits a complete timestamped bundle on each execute run:

- `regime_overlay_experiment_summary_<timestamp>.csv`
- `regime_overlay_experiment_fold_details_<timestamp>.csv`
- `regime_overlay_experiment_audit_artifacts_<timestamp>.csv`
- `regime_overlay_experiment_matched_nulls_<timestamp>.csv`
- `regime_overlay_experiment_report_<timestamp>.md`
- `regime_overlay_experiment_metadata_<timestamp>.json`

## Additional Notes

- output paths remain timestamped and non-overwriting
- markdown rendering no longer depends on optional `tabulate`
- synthetic tests now cover:
  - fold-details frame generation
  - matched-null frame generation
  - markdown report assembly
  - full output-path bundle naming
