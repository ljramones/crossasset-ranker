# Patch 12A: Regime-Overlay Runner Dry-Run Report

## Summary

The standalone regime-overlay runner passed a dry-run safety check after adding
explicit mode guards and non-overwriting output paths.

Validated properties:

- `--dry-run` performs no data load
- `--dry-run` performs no detector fit
- `--dry-run` performs no walk-forward split generation
- `--dry-run` performs no result-file writes
- the real run path is explicit via `--execute`
- real output paths are timestamped, so future runs do not overwrite prior
  artifacts by default
- the runner remains isolated from `main.py` and `prepare_experiment(...)`
- matched-null evaluation is wired against overlay positions with base
  net-return stream as the benchmark
- audit artifacts store explicit `executed_position` and overlay-specific fields

## CLI Safety Notes

### Safe dry-run command

Run the wrapper as a module from the repository root:

```bash
python -m scripts.run_regime_overlay_experiment --dry-run --output-dir /private/tmp/regime_overlay_dry_run_check --model-type hmm
```

Observed dry-run output:

- `Regime overlay runner dry run only.`
- `No data will be loaded. No detector will be fitted. No outputs will be written.`

### Important invocation nuance

This wrapper should be invoked as:

```bash
python -m scripts.run_regime_overlay_experiment ...
```

not:

```bash
python scripts/run_regime_overlay_experiment.py ...
```

because direct script-path execution from the repo root does not reliably place
the repository root on `sys.path` for `evaluation.*` imports.

## Files Checked

### `scripts/run_regime_overlay_experiment.py`

- no `main.py` import
- no `prepare_experiment(...)` import or call
- explicit mutually-exclusive mode guard:
  - `--dry-run`
  - `--execute`
- `--execute` validates required runtime arguments
- output files are timestamped:
  - `regime_overlay_experiment_summary_<timestamp>.csv`
  - `regime_overlay_experiment_audit_artifacts_<timestamp>.csv`

### `experiments/regime_overlay_experiment.py`

- validation/test regime inference uses `predict_live_safe(...)`
- matched nulls are evaluated with:
  - canonical positions = overlay executed positions
  - benchmark for overlay-vs-base nulls = base net-return stream
- audit artifact frame includes:
  - `executed_position`
  - `base_position`
  - `dangerous_regime_id`
  - `dangerous_regime_probability`
  - `overlay_threshold`
  - `overlay_risk_multiplier`
  - `overlay_mode`
  - `is_cut_day`

## Tests Run

```bash
python -m py_compile scripts/run_regime_overlay_experiment.py tests/test_regime_overlay_experiment.py
python -m pytest tests/test_regime_overlay_experiment.py tests/test_regime_overlay.py tests/test_null_baselines.py tests/test_metrics.py
```

Result:

- `34 passed`

## New/Updated Safety Coverage

`tests/test_regime_overlay_experiment.py` now verifies:

- dry-run does not call `pd.read_csv`
- execute mode refuses missing required runtime arguments
- output path generation is timestamped and non-overlapping
- train fit / validation live-safe / test live-safe call separation
- overlay audit artifacts use explicit executed positions without double shift

## Conclusion

The standalone regime-overlay runner is ready for a first controlled market-data
run later, but only via explicit `--execute` and only through the module-style
invocation path.
