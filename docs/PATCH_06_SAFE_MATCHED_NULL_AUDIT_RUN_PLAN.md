# Patch 06: Safe Matched-Null Audit Run Plan

## Purpose

Determine whether the current repository can generate a real matched-null comparative audit report from existing saved artifacts only, without triggering:

- model training
- walk-forward evaluation
- Optuna tuning
- data downloads

## Short Answer

No, not for the full current comparative audit workflow.

The existing saved artifacts are sufficient for a **partial artifact-only matched-null audit** of the saved `regime_stacking_ensemble` OOF outputs, but they are **not sufficient for the full comparative report** currently produced by `--audit --comparative-null-test`.

The current audit entry point is not safe for artifact-only execution because it rebuilds experiments, reloads data, and reruns evaluation logic.

## Audit Entry Points

```text
audit_entry_point:
  file: main.py
  function_or_command: main() -> _run_audit_workflow() via `--audit`
  does_it_train_models: true
  does_it_run_walk_forward: true
  does_it_download_data: unclear
  does_it_overwrite_results: true
  safe_to_run_now: false
  notes:
    - `main.py:74-79` dispatches `--audit` to `_run_audit_workflow(...)`.
    - `_run_audit_workflow(...)` in `main.py:131-147` calls `run_integrity_audit(...)`.
    - This path writes `results/integrity_audit_report.md` and, when enabled, `results/comparative_null_test_report.md`.
```

```text
audit_entry_point:
  file: audit/integrity_audit.py
  function_or_command: run_integrity_audit(...)
  does_it_train_models: true
  does_it_run_walk_forward: true
  does_it_download_data: unclear
  does_it_overwrite_results: true
  safe_to_run_now: false
  notes:
    - `run_integrity_audit(...)` at `audit/integrity_audit.py:53-90` begins with `prepare_experiment(...)`.
    - It writes markdown reports to the results directory.
    - Comparative mode invokes `_run_comparative_null_test(...)`, which rebuilds OOF frames and single-model artifacts.
```

```text
audit_entry_point:
  file: audit/integrity_audit.py
  function_or_command: _run_comparative_null_test(...)
  does_it_train_models: true
  does_it_run_walk_forward: true
  does_it_download_data: false
  does_it_overwrite_results: false
  safe_to_run_now: false
  notes:
    - `audit/integrity_audit.py:494-534`
    - Calls `_augment_regime_splits(...)`, `_load_best_params(...)`, and `_build_ensemble_oof_frame_local(...)`.
    - `_build_ensemble_oof_frame_local(...)` at `audit/integrity_audit.py:1097-1138` calls `evaluate_model(...)` for each base model.
    - `_run_one_model_comparative_null(...)` at `audit/integrity_audit.py:537-703` also evaluates single models through `_evaluate_single_model_artifacts(...)`.
```

```text
audit_entry_point:
  file: audit/integrity_audit.py
  function_or_command: _run_matched_null_audit(...)
  does_it_train_models: false
  does_it_run_walk_forward: false
  does_it_download_data: false
  does_it_overwrite_results: false
  safe_to_run_now: true
  notes:
    - `audit/integrity_audit.py:1037-1088`
    - Consumes executed positions, returns, benchmark returns, and optional regime labels.
    - Safe only when these inputs are sourced from existing artifacts rather than rebuilt experiments.
```

```text
audit_entry_point:
  file: evaluation/null_baselines.py
  function_or_command: run_matched_null_suite(...)
  does_it_train_models: false
  does_it_run_walk_forward: false
  does_it_download_data: false
  does_it_overwrite_results: false
  safe_to_run_now: true
  notes:
    - `evaluation/null_baselines.py:265-331`
    - Pure artifact-level utility that evaluates canonical executed positions and Monte Carlo nulls.
```

## Current Audit Architecture and Why It Is Unsafe

The current audit architecture is experiment-driven, not artifact-driven.

- `main.py:74-79` routes `--audit` to `_run_audit_workflow(...)`.
- `main.py:131-147` calls `run_integrity_audit(...)`.
- `audit/integrity_audit.py:62` immediately calls `prepare_experiment(...)`.
- `audit/integrity_audit.py:516-518` rebuilds regime-augmented splits and local OOF frames.
- `audit/integrity_audit.py:1097-1138` evaluates each base model again to rebuild the ensemble OOF frame.
- `audit/integrity_audit.py:596-647` evaluates saved single-model candidates again through `evaluate_model(...)`.

This means the current CLI audit path is not a safe artifact-only report generation path.

## Artifact Feasibility Check

### Artifacts That Exist and Are Sufficient

The following artifact is sufficient for a **single-model artifact-only matched-null audit** of the saved regime-stacking ensemble:

```text
results/ensembles/regime_stacking_oof.csv
```

Observed columns:

- `date`
- `split_id`
- `regime_id`
- `target_direction`
- `forward_simple_return_1d`
- `benchmark_return_1d`
- `probability`
- `prediction`

This is enough to derive:

- executed positions using split-local shift of `prediction`
- active returns via `forward_simple_return_1d`
- benchmark-relative metrics via `benchmark_return_1d`
- regime-matched nulls via `regime_id`

The same is also true for:

```text
results/ensembles/regime_stacking_oof_interactions.csv
```

Multi-asset cost-specific regime-stacking OOF files also exist, for example:

- `results/multi_asset/spy/cost_2_0bps/ensembles/regime_stacking_oof.csv`
- `results/multi_asset/qqq/cost_2_0bps/ensembles/regime_stacking_oof.csv`
- similar paths under `results/multi_asset/*/cost_*bps/ensembles/`

These are also plausible inputs for future artifact-only matched-null analysis.

### Artifacts That Exist but Are Not Sufficient

These files are **not sufficient** for matched-null comparative audit by themselves:

```text
results/ensembles/equity_curves/stacking_ensemble_per_split.csv
results/ensembles/equity_curves/regime_stacking_ensemble_per_split.csv
results/equity_curves/itransformer_per_split.csv
results/optimization/equity_curves/itransformer_tuned_per_split.csv
results/equity_curves/lstm_per_split.csv
results/equity_curves/buy_and_hold_spy_per_split.csv
```

Observed columns in these per-split equity files:

- `date`
- `split_id`
- `strategy_return`
- `equity_curve`

What is missing:

- raw predictions
- executed positions
- benchmark return series
- regime labels

Without those fields, we cannot safely run:

- same-average-exposure nulls
- same-turnover nulls
- same-regime-exposure nulls
- executed-position-aware benchmark-relative matched-null metrics

We also cannot safely infer executed positions from `strategy_return` alone because:

- costs are already embedded
- return zeroes are ambiguous
- sign and gross-return reconstruction may be lossy

### Comparative Report Coverage Gap

The current comparative audit targets four models:

- `regime_stacking_ensemble_regime`
- `stacking_ensemble_baseline`
- `itransformer_tuned_regime`
- `lstm_regime`

Artifact coverage today:

```text
regime_stacking_ensemble_regime:
  artifact_only_feasible: yes
  source: results/ensembles/regime_stacking_oof.csv

stacking_ensemble_baseline:
  artifact_only_feasible: no
  reason:
    - no saved OOF prediction file
    - only per-split strategy-return/equity files are present

itransformer_tuned_regime:
  artifact_only_feasible: no
  reason:
    - tuned per-split equity curve exists
    - no saved OOF prediction or executed-position artifact found
    - no saved benchmark-return-aligned OOF file found

lstm_regime:
  artifact_only_feasible: no
  reason:
    - only per-split strategy-return/equity files are present
    - no saved OOF prediction/executed-position artifact found
```

## Feasibility Conclusion

### Full Current Comparative Matched-Null Audit

Can we generate the current full comparative matched-null audit report using existing artifacts only?

**No.**

Reasons:

1. The current CLI audit path rebuilds experiments and reruns evaluation.
2. Saved artifact coverage is incomplete for three of the four comparative models.
3. The available per-split equity files do not contain enough information to reconstruct executed positions and benchmark-relative null inputs safely.

### Partial Artifact-Only Matched-Null Audit

Can we generate a real matched-null audit for the saved regime-stacking OOF artifact only?

**Yes, in principle.**

The saved OOF file contains the minimum required fields:

- test-slice returns
- benchmark returns
- split IDs
- raw predictions
- regime labels

However, there is **no existing safe CLI command** that performs this artifact-only report generation today.

## Safe Command Status

There is currently **no safe built-in command** to run now that would generate a real matched-null comparative audit using existing artifacts only.

Do **not** run:

```bash
uv run python main.py --audit --comparative-null-test
```

Why not:

- it calls `prepare_experiment(...)`
- it can reload data
- it reruns walk-forward model evaluation
- it overwrites existing markdown reports in `results/`

## What Would Be Needed for a Safe Artifact-Only Audit Path

### Minimal New Helper Needed

A future safe helper should:

1. Read an existing saved OOF artifact such as:
   - `results/ensembles/regime_stacking_oof.csv`
2. Reconstruct split-local executed positions from `prediction`:
   - `executed_position = prediction.shift(1)` within each `split_id`
3. Call:
   - `audit.integrity_audit._run_matched_null_audit(...)`
   - or directly `evaluation.null_baselines.run_matched_null_suite(...)`
4. Render a standalone markdown report to a new path, not overwrite current reports.

That helper would be safe because:

- no model objects are instantiated
- no data is downloaded
- no walk-forward evaluation is recomputed
- only saved artifacts are read

### Artifacts That Must Be Saved in Future Runs

For every model that may appear in a comparative audit, future runs should save a standardized OOF artifact with at least:

- `date`
- `split_id`
- `prediction`
- `probability` if available
- `forward_simple_return_1d`
- `benchmark_return_1d`
- `regime_id` if regime-aware
- optional `model`
- optional `cost_bps`

If the strategy uses non-binary or sized positions, also save:

- `executed_position`

Saving `executed_position` directly would remove ambiguity and make artifact-only audit paths simpler and safer.

## Recommended Next Step

Do not run the existing audit CLI.

Instead, the next safe implementation step should be:

1. add a small artifact-only helper or script
2. limit it initially to:
   - `results/ensembles/regime_stacking_oof.csv`
3. write output to a new path such as:
   - `results/comparative_null_test_report_artifact_only.md`
   - or `results/matched_null_regime_stacking_artifact_only.md`
4. only after that, expand saved OOF coverage for:
   - `stacking_ensemble_baseline`
   - `itransformer_tuned_regime`
   - `lstm_regime`

## Final Answer

```text
Can we generate a real matched-null comparative audit report for the current saved model outputs using existing artifacts only?

Full comparative report:
  No.

Artifact-only regime-stacking matched-null report:
  Yes, in principle, from saved OOF artifacts.

Safe built-in command available right now:
  No.
```
