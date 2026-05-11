# Drawdown Calibration Gate Failure and Target Sensitivity Plan

## 1. Executive Summary

The decision-grade standalone drawdown calibration run completed successfully.

The artifact bundle was valid.

The calibration gate failed.

`isotonic` was the best calibration method.

Calibration improved Brier score and ECE versus the raw linear models, but no calibrated model beat the constant `event_rate` baseline on mean test Brier score.

No economic overlay should be built.

Calibration improved the weak classifiers, but did not make them decision-grade.

The next step should be target sensitivity, not model complexity.

## 2. What Was Run

The calibration stage evaluated four standalone classifier-only candidates on the offline prepared SPY feature frame:

- `logistic + platt`
- `logistic + isotonic`
- `regularized_linear + platt`
- `regularized_linear + isotonic`

This was done through the standalone calibration runner only.

It did not use:

- `main.py`
- `prepare_experiment(...)`
- old model-zoo workflows
- Optuna
- deep models
- sequence models
- stacking ensembles
- economic overlays

## 3. Best Results

Best calibrated candidates:

### `logistic + isotonic`

- mean test ROC AUC: `0.533090`
- mean test Brier: `0.235252`
- pooled ECE improved from `0.184193` to `0.150555`

### `regularized_linear + isotonic`

- mean test ROC AUC: `0.528105`
- mean test Brier: `0.235154`
- pooled ECE improved from `0.188082` to `0.140728`

Reference baseline:

### `event_rate`

- mean test Brier: `0.205322`

## 4. Why the Calibration Gate Failed

The calibration experiment did improve the weak linear classifiers.

What improved:

- Brier score versus the raw linear models
- ECE versus the raw linear models
- mean predicted probability moved closer to the realized event rate
- `isotonic` preserved ranking materially better than `platt`

What did not improve enough:

- no calibrated model beat the constant `event_rate` baseline on mean test Brier score

That is the decisive failure.

This means:

- the weak ranking signal was not entirely noise
- but the signal was not strong enough to produce decision-grade probabilities in the current target setup

## 5. Why This Was Not a Plumbing Failure

The failure should not be blamed on broken infrastructure.

What worked:

- the standalone calibration runner executed successfully
- all expected output files were created
- the report and metadata correctly marked the runs as classifier-only
- no economic overlay logic was triggered
- no old workflow was triggered

This was a research result, not an execution failure.

## 6. Calibration Method Decision

Current method ranking:

1. `isotonic`
2. `platt`

Interpretation:

- `isotonic` was the only calibration method that improved probability quality without broadly destroying ranking
- `platt` scaling was too unstable and often degraded AUC materially

Practical conclusion:

```text
If any further calibration work happens on this target family, isotonic is the only method worth carrying forward by default.
```

## 7. Economic Overlay Decision

Economic overlay remains blocked.

Reasons:

- the classifier gate already failed before overlay conversion
- the calibration gate also failed
- a probability stream that still loses to `event_rate` on Brier score is not ready for policy conversion

Therefore:

```text
No economic overlay should be built from the current target_drawdown_event_20d_3pct classifier outputs.
```

## 8. What This Result Means

This is not a total null.

The calibrated isotonic results suggest:

- some weak ordering information may exist
- probability quality can be improved somewhat
- but the current target-definition and feature setup are still not strong enough to clear the baseline gate

That changes the next question.

The next question is not:

```text
Can we use a more complex model?
```

The next question should be:

```text
Is the current target definition the problem?
```

## 9. Why Target Sensitivity Is the Right Next Step

Target sensitivity is the correct next stage because:

- the model family already showed only weak ranking signal
- calibration improved that weak signal only partially
- the constant event-rate baseline remains hard to beat on probability quality
- adding more model complexity now would likely be noise mining

The project needs to test whether a nearby drawdown target is more learnable.

## 10. Target Sensitivity Plan

The next experiment family should stay classifier-only and test nearby target variants before any new model complexity.

Recommended target grid:

### Horizon sensitivity

- `target_drawdown_event_10d_2pct`
- `target_drawdown_event_10d_3pct`
- `target_drawdown_event_10d_5pct`
- `target_drawdown_event_20d_2pct`
- `target_drawdown_event_20d_3pct`
- `target_drawdown_event_20d_5pct`

### Evaluation approach

For each target:

1. keep the simple classifier set small
   - `event_rate`
   - `rolling_event_rate`
   - `logistic`
   - `regularized_linear`
2. run classifier-only walk-forward evaluation
3. if raw ranking signal is weak, stop
4. if raw ranking signal is modest but promising, test isotonic calibration only
5. compare against:
   - `event_rate` on Brier
   - `rolling_event_rate` on Brier
   - ROC AUC stability across folds

### Success criterion for a target variant

A target should only be considered promising if:

- raw or calibrated linear models beat `event_rate` on mean test Brier
- the AUC edge is stable across folds
- calibration does not collapse ranking
- the result is not driven by one fold

## 11. What Not To Do Next

Do not:

- build an overlay from the current target
- jump to deep models
- jump to ensemble complexity
- run Optuna
- tune endlessly around `target_drawdown_event_20d_3pct`

The right next move is to test whether the target itself is the weak link.

## 12. Current Project Decision

Current status of `target_drawdown_event_20d_3pct`:

- not a total null
- not decision-grade
- not eligible for economic overlay
- not strong enough to justify model-complexity escalation

Therefore:

```text
Move to target sensitivity before any additional model complexity or policy conversion work.
```
