# Patch 20: Volatility-Quantile Overlay Dry-Run Safety Report

## Summary

The standalone volatility-quantile overlay runner passed a dry-run safety check.

Validated properties:

- `--dry-run` exits before loading data
- `--dry-run` exits before generating walk-forward splits
- `--dry-run` exits before writing files
- the execute path remains explicit via `--execute`
- execute mode requires explicit runtime arguments
- output paths are timestamped for execute mode
- the runner is isolated from `main.py`, `prepare_experiment(...)`, Optuna, model-zoo workflows, and regime-stacking workflows

## Files Inspected

### `scripts/run_vol_quantile_overlay_experiment.py`

Observed:

- explicit mutually exclusive modes:
  - `--dry-run`
  - `--execute`
- dry-run branch returns before:
  - `pd.read_csv(...)`
  - `generate_walk_forward_splits(...)`
  - any output writes
- execute path validates required runtime args
- execute path writes timestamped output bundle names

### `experiments/vol_quantile_overlay_experiment.py`

Observed:

- no import from `main.py`
- no import or call to `prepare_experiment(...)`
- no HMM/regime detector import
- no model-zoo or ensemble dependency
- no Optuna dependency
- uses:
  - repaired metric layer
  - standalone null-baseline utilities
  - standalone vol-target / overlay evaluation path

## Dry-Run Command

Executed from repo root:

```bash
python -m scripts.run_vol_quantile_overlay_experiment --dry-run --output-dir /private/tmp/vol_quantile_dry_run_check
```

Observed output:

- `Volatility-quantile overlay runner dry run only.`
- `No data will be loaded. No outputs will be written.`

No output directory was created by the dry-run invocation.

## Output-Path Guardrails

Execute-mode output paths are timestamped:

- `vol_quantile_overlay_experiment_summary_<timestamp>.csv`
- `vol_quantile_overlay_experiment_fold_details_<timestamp>.csv`
- `vol_quantile_overlay_experiment_audit_artifacts_<timestamp>.csv`
- `vol_quantile_overlay_experiment_matched_nulls_<timestamp>.csv`
- `vol_quantile_overlay_experiment_report_<timestamp>.md`
- `vol_quantile_overlay_experiment_metadata_<timestamp>.json`

This avoids overwriting prior runs by default.

## Test Coverage

Verified with targeted tests:

```bash
python -m py_compile scripts/run_vol_quantile_overlay_experiment.py tests/test_vol_quantile_overlay_experiment.py
python -m pytest tests/test_volatility_overlay.py tests/test_vol_quantile_overlay_experiment.py tests/test_null_baselines.py tests/test_metrics.py
```

Result:

- `33 passed`

Synthetic test coverage confirms:

- dry-run does not call `pd.read_csv`
- execute mode refuses missing required runtime arguments
- output bundle naming is timestamped and complete
- same-vol-state matched-null path is wired in

## Safety Conclusion

The standalone volatility-quantile overlay runner is safe to invoke in dry-run mode and is appropriately isolated from the old workflows.

## Next Safe Step

The next safe step is:

```text
A tiny offline bundle-verification execute run using a prepared cached feature CSV
and a very small null budget.
```

That run should verify:

- execute path works on real offline input
- full output bundle is emitted
- audit artifacts contain the required vol-state columns

It should not be interpreted as a research result.
