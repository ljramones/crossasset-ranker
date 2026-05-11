# Drawdown Classifier Dry-Run Safety Report

## Executive Summary

The standalone drawdown-risk classifier CLI passed dry-run safety validation.

Safe dry-run command:

```bash
python -m scripts.run_drawdown_risk_classifier_experiment --dry-run
```

Observed output:

```text
Drawdown-risk classifier runner dry run only.
No data will be loaded. No outputs will be written.
```

This confirms the dry-run path exits before:

- reading any CSV input
- generating walk-forward splits
- fitting any classifier
- writing any output files

The runner remains isolated from:

- `main.py`
- `prepare_experiment(...)`
- Optuna
- old model-zoo workflows
- ensemble workflows
- HMM/regime workflows
- economic overlay workflows

## Dry-Run Boundary Check

Inspected file:

- [run_drawdown_risk_classifier_experiment.py](/Users/larrym/prediction/scripts/run_drawdown_risk_classifier_experiment.py)

Dry-run boundary behavior:

- `--dry-run` is in a mutually exclusive mode group with `--execute`
- the script returns immediately after printing the dry-run message
- `pd.read_csv(...)` is only called after the dry-run early return
- `generate_walk_forward_splits(...)` is only called after data loading
- classifier fitting happens only inside the standalone experiment path, which is only reached after execute-mode argument validation and data loading
- file writes are timestamped and only happen in execute mode

## Execute-Mode Guardrails

Execute mode currently requires explicit runtime arguments for:

- `input_csv`
- `output_dir`
- `target_column`
- `model_name`

Timestamped non-overwriting output naming is present via:

- `strftime("%Y%m%d_%H%M%S", gmtime())`

Expected execute outputs are timestamped:

- `drawdown_risk_classifier_summary_<timestamp>.csv`
- `drawdown_risk_classifier_fold_details_<timestamp>.csv`
- `drawdown_risk_classifier_oof_artifacts_<timestamp>.csv`
- `drawdown_risk_classifier_report_<timestamp>.md`
- `drawdown_risk_classifier_metadata_<timestamp>.json`

## Isolation Check

A source scan confirmed no dependency on:

- `prepare_experiment`
- `main.py`
- Optuna imports
- ensemble paths
- overlay execution paths

The standalone runner depends only on:

- drawdown-label utilities
- walk-forward split generation
- standalone classifier experiment module

## Rolling Event-Rate Baseline Status

Current status:

- `rolling_event_rate` exists only as a builder alias
- it currently resolves to the same `ConstantProbabilityClassifier` implementation used by `event_rate` / `historical_event_rate`

That means:

- a distinct rolling event-rate baseline is not yet implemented
- it should be treated as deferred, not as a truly separate baseline

This is acceptable for the dry-run safety stage.

Before the first real classifier comparison run, the baseline list should be described precisely as:

- always-negative
- constant historical event-rate
- logistic regression
- regularized linear classifier
- HistGradientBoosting

And the rolling event-rate baseline should either:

1. be implemented distinctly, or
2. be omitted from the first real run

## Verification

Commands run:

```bash
python -m scripts.run_drawdown_risk_classifier_experiment --dry-run
python -m pytest tests/test_drawdown_classification.py tests/test_drawdown_risk_classifier_experiment.py tests/test_drawdown_labels.py
```

Result:

- dry-run behaved safely
- targeted tests passed: `20 passed`

## Next Safe Step

The next safe step is:

```text
a tiny offline execute plumbing run on the prepared SPY feature frame
```

That run should:

- stay inside the standalone classifier path
- use the viable target `target_drawdown_event_20d_3pct`
- produce the normalized output bundle
- remain clearly non-decision-grade
