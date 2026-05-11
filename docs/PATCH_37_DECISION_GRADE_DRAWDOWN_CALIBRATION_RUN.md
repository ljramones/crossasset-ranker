# Decision-Grade Drawdown Calibration Run

## Executive Summary

The first decision-grade standalone drawdown calibration experiment completed successfully on the offline prepared SPY feature frame.

This was a classifier-only calibration run.

No economic overlay was evaluated.

Overall verdict:

```text
FAIL
```

Reason:

- isotonic calibration improved Brier score and calibration quality for both `logistic` and `regularized_linear`
- but no calibrated model beat the constant `event_rate` baseline on mean test Brier score
- Platt scaling materially damaged ranking quality
- isotonic preserved ranking better, but still did not clear the full calibration gate

So the weak ranking signal was partially calibratable, but not enough to justify moving into an economic overlay stage.

## Baseline Reference Point

Uncalibrated decision-grade classifier baselines:

- `event_rate`
  - mean test ROC AUC: `0.500000`
  - mean test Brier: `0.205322`

- `rolling_event_rate`
  - mean test ROC AUC: `0.500000`
  - mean test Brier: `0.246417`

- `logistic`
  - mean test ROC AUC: `0.548617`
  - mean test Brier: `0.251957`

- `regularized_linear`
  - mean test ROC AUC: `0.552627`
  - mean test Brier: `0.251150`

## Decision-Grade Calibration Runs

Four standalone calibration candidates were run:

- `logistic + platt`
- `logistic + isotonic`
- `regularized_linear + platt`
- `regularized_linear + isotonic`

Output roots:

- [logistic_platt](/Users/larrym/prediction/results/drawdown_calibration_decision_grade_run/logistic_platt)
- [logistic_isotonic](/Users/larrym/prediction/results/drawdown_calibration_decision_grade_run/logistic_isotonic)
- [regularized_linear_platt](/Users/larrym/prediction/results/drawdown_calibration_decision_grade_run/regularized_linear_platt)
- [regularized_linear_isotonic](/Users/larrym/prediction/results/drawdown_calibration_decision_grade_run/regularized_linear_isotonic)

## Aggregate Results

### `logistic + platt`

- mean test calibrated ROC AUC: `0.464733`
- mean test calibrated Brier: `0.245809`
- mean test Brier improvement vs raw logistic: `0.006148`
- pooled calibrated ECE: `0.177735`
- beats `event_rate` on Brier: `no`
- beats `rolling_event_rate` on Brier: `yes`

### `logistic + isotonic`

- mean test calibrated ROC AUC: `0.533090`
- mean test calibrated Brier: `0.235252`
- mean test Brier improvement vs raw logistic: `0.016706`
- pooled calibrated ECE: `0.150555`
- beats `event_rate` on Brier: `no`
- beats `rolling_event_rate` on Brier: `yes`

### `regularized_linear + platt`

- mean test calibrated ROC AUC: `0.472100`
- mean test calibrated Brier: `0.244304`
- mean test Brier improvement vs raw regularized linear: `0.006845`
- pooled calibrated ECE: `0.177944`
- beats `event_rate` on Brier: `no`
- beats `rolling_event_rate` on Brier: `yes`

### `regularized_linear + isotonic`

- mean test calibrated ROC AUC: `0.528105`
- mean test calibrated Brier: `0.235154`
- mean test Brier improvement vs raw regularized linear: `0.015996`
- pooled calibrated ECE: `0.140728`
- beats `event_rate` on Brier: `no`
- beats `rolling_event_rate` on Brier: `yes`

## Best Candidate

The best calibration candidates were the isotonic runs:

- `logistic + isotonic`
- `regularized_linear + isotonic`

Why:

- both materially improved Brier score versus their raw models
- both materially reduced pooled ECE
- both moved mean predicted probability closer to the realized event rate
- both avoided the severe AUC collapse seen under Platt scaling

Important limits:

- `logistic + isotonic` mean test ROC AUC was `0.533090`
- `regularized_linear + isotonic` mean test ROC AUC was `0.528105`
- neither beat the `event_rate` baseline on mean test Brier

That means the calibration rescue improved probability quality, but not enough to clear the full gate.

## Fold-Level Stability

Fold-level Brier improvement counts:

- `logistic + platt`: improved in `6 / 9` folds
- `logistic + isotonic`: improved in `6 / 9` folds
- `regularized_linear + platt`: improved in `6 / 9` folds
- `regularized_linear + isotonic`: improved in `6 / 9` folds

Fold-level AUC behavior:

- Platt scaling was unstable and often destructive
  - `logistic + platt`: calibrated AUC below `0.5` in `5 / 9` folds
  - `regularized_linear + platt`: calibrated AUC below `0.5` in `5 / 9` folds

- isotonic was materially safer
  - `logistic + isotonic`: calibrated AUC below `0.5` in `1 / 9` folds
  - `regularized_linear + isotonic`: calibrated AUC below `0.5` in `1 / 9` folds

This means:

- Platt scaling should not be the preferred next path here
- isotonic is the only method that looked plausibly useful
- even isotonic was not strong enough to clear the full classifier calibration gate

## Gate Decision

Primary calibration gate:

- beat `event_rate` on mean test Brier
- beat `rolling_event_rate` on mean test Brier
- improve Brier versus the uncalibrated same model
- improve ECE versus the uncalibrated same model
- preserve ranking signal without material AUC deterioration

Result:

- beat `event_rate` on Brier: `no`
- beat `rolling_event_rate` on Brier: `yes` for all four calibrated runs
- improved Brier vs same raw model: `yes`
- improved ECE vs same raw model: `yes`
- preserved ranking adequately:
  - `platt`: `no`
  - `isotonic`: `partially`

Therefore:

```text
The calibration gate fails because the calibrated models still do not beat the constant event-rate baseline on mean test Brier score.
```

## Interpretation

This is not a total null.

The result suggests:

- the weak ranking signal was not entirely noise
- isotonic calibration did improve the probability layer
- but the underlying signal remains too weak to produce decision-grade probabilities

So the correct interpretation is:

```text
Calibration helped, but not enough.
```

That is not sufficient to justify any economic overlay stage.

## Decision

Current status:

- no economic overlay should be built from these calibrated classifier outputs
- no trading claim should be made
- no model should be promoted

The safest next move is another documentation and design step:

- record that isotonic was the only non-destructive calibrator
- record that the classifier calibration gate still failed
- decide whether to:
  - stop work on `target_drawdown_event_20d_3pct`, or
  - run one narrower follow-up around isotonic only with stricter feature or target simplification

## Boundary Conditions

This task did not:

- run `main.py`
- run `prepare_experiment(...)`
- run Optuna
- run deep or sequence models
- run stacking or ensemble workflows
- run HMM or volatility overlay workflows
- run any economic overlay

This was a standalone classifier calibration decision run only.
