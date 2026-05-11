# Patch 15: Tiny Regime-Overlay Bundle Verification Run

## Purpose

This was a **tiny offline bundle-verification rerun**, not a decision-grade
research run.

It was used only to confirm that the normalized standalone regime-overlay
output bundle is produced correctly on real cached SPY data.

## Offline Input Check

Prepared input was already available:

- `/private/tmp/regime_overlay_spy_feature_frame.csv`

No feature rebuild or data download was required for this task.

## Command Used

Executed from repo root:

```bash
python -m scripts.run_regime_overlay_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/regime_overlay_bundle_verification_run \
  --train-size 756 \
  --val-size 126 \
  --test-size 126 \
  --step-size 5000 \
  --target-vol 0.10 \
  --realized-vol-window 20 \
  --transaction-cost-bps 2.0 \
  --null-runs 5 \
  --asset-name SPY \
  --model-name hmm_regime_overlay_hard_veto_bundle_check \
  --date-column date \
  --model-type hmm
```

Notes:

- `step-size 5000` forced a single-split plumbing run
- `null-runs 5` kept the run explicitly diagnostic only

## Bundle Verification Result

The standalone runner executed successfully and produced all six expected files:

1. `results/regime_overlay_bundle_verification_run/regime_overlay_experiment_summary_20260508_194742.csv`
2. `results/regime_overlay_bundle_verification_run/regime_overlay_experiment_fold_details_20260508_194742.csv`
3. `results/regime_overlay_bundle_verification_run/regime_overlay_experiment_audit_artifacts_20260508_194742.csv`
4. `results/regime_overlay_bundle_verification_run/regime_overlay_experiment_matched_nulls_20260508_194742.csv`
5. `results/regime_overlay_bundle_verification_run/regime_overlay_experiment_report_20260508_194742.md`
6. `results/regime_overlay_bundle_verification_run/regime_overlay_experiment_metadata_20260508_194742.json`

## Parsing / Completeness Checks

### Summary CSV

- parsed successfully
- rows: `1`
- contains expected summary columns including:
  - `overlay_vs_base_information_ratio`
  - `same_average_exposure_p_value`
  - `same_turnover_p_value`
  - `same_regime_exposure_p_value`

### Fold Details CSV

- parsed successfully
- rows: `1`
- contains detailed validation/test fields including:
  - `decision_metric`
  - `validation_score`
  - `test_overlay_vs_base_information_ratio`
  - `position_change_*`

### Audit Artifact CSV

- parsed successfully
- rows: `126`
- contains required overlay audit columns including:
  - `executed_position`
  - `base_position`
  - `dangerous_regime_id`
  - `dangerous_regime_probability`
  - `overlay_threshold`
  - `overlay_risk_multiplier`
  - `overlay_mode`
  - `is_cut_day`

### Matched Nulls CSV

- parsed successfully
- rows: `5`
- contains one row for each null family:
  - `same_average_exposure_random`
  - `same_turnover_random`
  - `same_exposure_and_turnover_random`
  - `same_regime_exposure_random`
  - `block_bootstrap_same_exposure_random`
- contains:
  - `decision_metric`
  - `canonical_value`
  - `mean_null_value`
  - `percentile_95_null_value`
  - `p_value`
  - `n_runs`
  - `passes_p_value_gate`

### Markdown Report

- parsed successfully as text
- contains:
  - summary section
  - fold details section
  - matched null diagnostics section
- clearly states:
  - `Null runs: 5`
  - `Decision grade: False`

### Metadata JSON

- parsed successfully
- contains:
  - `mode: "execute"`
  - `null_runs: 5`
  - `n_splits: 1`
  - `decision_grade: false`
  - warning notes that results are not validated at this budget

## Tiny Diagnostic Output Snapshot

Observed summary values:

- `overlay_vs_base_information_ratio = 0.068381`
- `same_average_exposure_p_value = 0.6`
- `same_turnover_p_value = 0.6`
- `same_regime_exposure_p_value = 0.6`

These values are **not** decision-grade and should not be interpreted as a
research conclusion.

## Safety / Workflow Boundary Check

No evidence of old workflows being triggered was found during this run:

- no `main.py`
- no `prepare_experiment(...)`
- no Optuna
- no regime-stacking ensemble path
- no model-zoo training workflow

This was a standalone overlay execution only.

## Conclusion

The normalized output bundle is now verified on a real offline cached-data run.

The standalone regime-overlay runner is operational and artifact-complete for
future controlled experiments, but this tiny rerun does **not** validate the
overlay as a research result.
