# Drawdown Calibration Dry-Run Safety Report

## Executive Summary

The standalone drawdown calibration runner passed dry-run safety validation.

Confirmed:

- dry-run exits before reading CSV input
- dry-run exits before generating walk-forward splits
- dry-run exits before fitting base classifiers
- dry-run exits before fitting calibrators
- dry-run exits before writing any output files

The calibration runner remains isolated from:

- `main.py`
- `prepare_experiment(...)`
- Optuna
- old model-zoo workflows
- ensemble workflows
- HMM/regime workflows
- volatility/HMM overlay workflows
- economic overlays

## Inspected Entry Point

Primary CLI wrapper:

- [run_drawdown_risk_calibration_experiment.py](/Users/larrym/prediction/scripts/run_drawdown_risk_calibration_experiment.py)

Primary experiment module:

- [drawdown_risk_calibration_experiment.py](/Users/larrym/prediction/experiments/drawdown_risk_calibration_experiment.py)

Primary calibration utilities:

- [probability_calibration.py](/Users/larrym/prediction/evaluation/probability_calibration.py)

## Dry-Run Behavior

Safe dry-run command:

```bash
python -m scripts.run_drawdown_risk_calibration_experiment --dry-run
```

Observed output:

```text
Drawdown-risk calibration runner dry run only.
No data will be loaded. No outputs will be written.
```

The early return occurs before:

- `pd.read_csv(...)`
- `append_drawdown_label_grid(...)`
- `generate_walk_forward_splits(...)`
- `run_drawdown_risk_calibration_experiment(...)`
- any CSV, markdown, or JSON writes

## Execute-Mode Guardrails

Execute mode requires explicit runtime arguments:

- `--input-csv`
- `--output-dir`
- `--target-column`
- `--base-model-name`
- `--calibration-method`

Future execute outputs are timestamped and non-overwriting:

- `drawdown_risk_calibration_summary_<timestamp>.csv`
- `drawdown_risk_calibration_fold_details_<timestamp>.csv`
- `drawdown_risk_calibration_oof_artifacts_<timestamp>.csv`
- `drawdown_risk_calibration_bins_<timestamp>.csv`
- `drawdown_risk_calibration_report_<timestamp>.md`
- `drawdown_risk_calibration_metadata_<timestamp>.json`

## Implemented Calibration Methods

Currently implemented:

- `identity`
- `platt`
- `isotonic`

Deferred:

- beta calibration
- temperature scaling
- any pooled cross-fold calibrator

## Verification

Dry-run command executed:

```bash
python -m scripts.run_drawdown_risk_calibration_experiment --dry-run
```

Focused tests executed:

```bash
python -m pytest tests/test_probability_calibration.py tests/test_drawdown_risk_calibration_experiment.py tests/test_calibration_diagnostics.py
```

Result:

```text
19 passed
```

## Next Safe Step

The next safe step is a tiny offline plumbing execute run on the prepared SPY feature frame, still classifier-only and still without any economic overlay.
