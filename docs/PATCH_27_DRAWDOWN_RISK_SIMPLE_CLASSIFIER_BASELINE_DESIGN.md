# Drawdown-Risk Simple Classifier Baseline Design

## 1. Executive Summary

The first classifier experiment in the target-reset track should be a simple supervised baseline for:

```text
target_drawdown_event_20d_3pct
```

This experiment should answer a narrower question than the failed daily-timing work:

```text
Can simple models detect elevated 20-day drawdown risk better than trivial baselines?
```

This is a classifier-first experiment.

It is not yet an economic overlay experiment.

The proposed sequence is:

1. build the viable drawdown-risk label into the prepared feature frame
2. run simple walk-forward classifiers only
3. rank them by event-detection and calibration quality
4. only if classifier skill is stable, convert the best model into an exposure rule later

This design does not use:

- the failed regime-stacking ensemble
- the failed HMM hard-veto overlay
- the failed volatility-quantile overlay
- Optuna
- deep sequence models

## 2. Why `target_drawdown_event_20d_3pct`

The offline label viability report showed:

- all six candidate labels were viable across all `9` walk-forward splits
- `target_drawdown_event_20d_3pct` had:
  - overall positive rate `0.291667`
  - overall positive count `966`
  - minimum test positive count `41`
  - full train / validation / test viability across all splits

This makes it strong enough for classifier training while still representing a meaningful risk event.

Why not start with `target_drawdown_event_20d_2pct`:

- it has stronger count support
- but it is broader and less specific as a risk event
- it is better kept as a sensitivity target

Why not start with `target_drawdown_event_10d_5pct`:

- it is viable
- but it is materially sparser and closer to fold fragility

Recommended target order:

1. `target_drawdown_event_20d_3pct`
2. `target_drawdown_event_20d_2pct`
3. `target_drawdown_event_10d_3pct`

## 3. Experiment Question

The first simple baseline experiment should test:

```text
Can a simple supervised classifier predict 20-day, -3% drawdown events better than naive baselines, with stable walk-forward performance?
```

This is deliberately narrower than:

- predicting daily SPY direction
- timing exposure directly
- claiming economic value immediately

The classifier must earn the right to be turned into a policy.

## 4. Files and Functions Inspected

### `features/feature_engineering.py`

Relevant paths:

- `FeatureSet`
- `build_feature_set(...)`

Current limitation:

- the current build path is hardwired to `target_direction`

Design implication:

- the first classifier experiment needs a target-aware feature-preparation branch
- drawdown labels should coexist with the current features, not replace the feature blocks themselves

### `utils/experiment.py`

Relevant paths:

- `prepare_experiment(...)`
- `prepare_experiment_from_market_data(...)`
- `evaluate_model(...)`

Current limitation:

- `evaluate_model(...)` still hardcodes:
  - `target_column="target_direction"`
  - `return_column="forward_simple_return_1d"`
  - `benchmark_column="benchmark_return_1d"`

Design implication:

- the first classifier baseline should likely use a dedicated standalone experiment path or a small generalization of `evaluate_model(...)`
- do not jam drawdown-risk into `target_direction`

### `evaluation/walk_forward.py`

Relevant path:

- `generate_walk_forward_splits(...)`

Design implication:

- the walk-forward split logic is already correct for this experiment
- selection must remain validation-only
- reporting must remain strictly test-only

### `evaluation/drawdown_labels.py`

Relevant paths:

- `append_drawdown_label_grid(...)`
- `get_drawdown_label_columns(...)`
- `evaluate_candidate_drawdown_labels(...)`

Design implication:

- the target utilities already support the label family and viability checks
- the next step is classifier evaluation, not more label invention

### `evaluation/audit_artifacts.py`

Relevant path:

- `build_standard_audit_artifact_frame(...)`

Design implication:

- even the classifier-only experiment should save standardized OOF artifacts
- especially:
  - `target`
  - `prediction_probability`
  - `model_name`
  - `asset_return`
  - `benchmark_return`

### `evaluation/metrics.py`

Relevant paths:

- `compute_classification_metrics(...)`
- active-return diagnostics

Design implication:

- the classifier stage needs classification metrics first
- active-return metrics are for the later policy stage, not the first gate

## 5. Proposed Baseline Models

Start with simple models only.

Required baselines:

1. always-negative baseline
2. rolling historical event-rate baseline
3. logistic regression
4. regularized linear classifier
5. HistGradientBoosting

Optional later, but still “simple enough”:

6. LightGBM
7. XGBoost

Do not include:

- LSTM
- PatchTST
- iTransformer
- Mamba
- TFT

The new rule remains:

```text
No complex model until a simple drawdown-risk model proves itself.
```

## 6. Input Features

The first classifier should reuse the existing stationary feature backbone.

Core current features:

- `return_1d`
- `return_5d`
- `return_20d`
- `vol_ratio`
- `momentum_norm`
- `volume_zscore`
- `range_norm`
- `sma_ratio`
- `realized_vol_20`
- `close_to_open_gap`
- `price_acceleration`
- `downside_vol_ratio`

Recommended advanced features to include in at least one baseline variant:

- `relative_strength_vs_benchmark`
- `overnight_gap_zscore`
- `volume_trend_strength`
- `autocorrelation_zscore`
- `volatility_regime`
- `asset_return_vs_spy`
- `relative_vol_ratio`
- VIX-derived features when available

Initial experiment structure:

- baseline feature set
- baseline + advanced + VIX feature set

This allows the first classifier experiment to test both a compact and richer stationary feature space without touching model complexity.

## 7. Walk-Forward Design

Use the same time-ordered structure already validated in the repo:

- train: `756`
- validation: `252`
- test: `252`
- step: `252`

Selection rules:

- model hyperparameters must be fixed or minimal
- no Optuna
- any threshold choice must use validation only
- final classifier scoring must use test only

This first classifier experiment should focus on stable out-of-fold probability forecasts, not parameter mining.

## 8. Metrics

### 8.1 Primary Classifier Metrics

Recommended primary metrics:

- ROC AUC
- precision
- recall
- balanced accuracy
- Brier score

Add:

- positive prediction rate
- base event rate by fold
- simple calibration diagnostics

The key question is:

```text
Does the model separate drawdown-risk events from non-events better than trivial baselines across folds?
```

### 8.2 Ranking Order

Rank simple classifiers primarily by:

1. mean test ROC AUC
2. fold stability of ROC AUC
3. Brier score
4. balanced accuracy
5. recall at a reasonable risk threshold

Do not rank by:

- Sharpe
- information ratio
- any economic metric yet

This stage is about event detection quality.

## 9. Baseline Comparisons

The classifier should be compared against:

### Naive event baselines

- always-negative classifier
- historical unconditional event-rate classifier
- rolling event-rate classifier

### Model baselines

- logistic regression
- regularized linear classifier
- HistGradientBoosting

The baseline to beat first is not “buy and hold.”

It is:

```text
simple event-detection baselines on the same target
```

## 10. OOF Artifact Requirements

Even though this is a classifier-only stage, the experiment should save standardized OOF artifacts so later steps do not need reconstruction.

Minimum fields:

- `date`
- `split_id`
- `model_name`
- `asset_return`
- `benchmark_return`
- `target`
- `raw_signal` or predicted class if emitted
- `prediction_probability`

No executed-position field is required yet unless the classifier is explicitly mapped to a policy.

## 11. Stop / Go Gates

### Gate 1: Label Viability

Already passed for `target_drawdown_event_20d_3pct`.

### Gate 2: Classifier Skill

Proceed to policy conversion only if at least one simple model:

- beats naive event baselines on mean test ROC AUC
- is not wildly unstable across folds
- has usable calibration
- does not collapse to trivial class predictions

### Gate 3: Complexity Escalation

Only consider stronger tree models or anything more complex if:

- the simple models show real classifier skill first

If simple models fail here, do not jump to sequence architectures.

## 12. Recommended First Implementation Shape

The safest implementation path is a standalone classifier experiment runner rather than modifying `main.py` first.

Suggested shape:

- standalone script or experiment module for drawdown-risk classification
- consumes prepared feature frame
- appends drawdown label grid
- selects `target_drawdown_event_20d_3pct`
- runs walk-forward classifiers
- writes:
  - fold metrics CSV
  - summary CSV
  - OOF predictions CSV
  - report markdown
  - metadata JSON

This mirrors the safer standalone pattern used in the overlay work.

## 13. What To Inspect Before Implementation

Before patching code for the classifier experiment, verify:

- where `target_column` is still hardcoded to `target_direction`
- whether `BaseSignalModel` implementations assume directional labels semantically
- how probability outputs are standardized across simple models
- how `compute_classification_metrics(...)` is currently used
- how OOF predictions are currently assembled and saved

Do not assume the current model interface is target-agnostic until verified.

## 14. Recommended First Experiment Matrix

Primary target:

- `target_drawdown_event_20d_3pct`

First classifier set:

- always-negative
- rolling event-rate baseline
- logistic regression
- regularized linear classifier
- HistGradientBoosting

Feature variants:

1. baseline stationary features only
2. baseline + advanced + VIX features

This is enough to answer the first research question without exploding scope.

## 15. Explicit Non-Goals

This first classifier baseline experiment should not:

- optimize trading performance
- build an overlay policy yet
- run matched-null economic tests yet
- use Optuna
- compare to old ensemble champions
- use HMM or regime state logic

Those belong only after classifier skill is established.

## 16. Final Recommendation

The next concrete experiment should be:

```text
A standalone walk-forward simple-classifier baseline on target_drawdown_event_20d_3pct.
```

That is the cleanest next test because:

- the target is viable
- the label is meaningful
- the experiment is simpler than the failed timing tracks
- the result will decisively tell us whether the feature set contains useful forward drawdown information

If this simple classifier baseline fails, the project should become much more skeptical of single-asset SPY predictive modeling in its current form.
