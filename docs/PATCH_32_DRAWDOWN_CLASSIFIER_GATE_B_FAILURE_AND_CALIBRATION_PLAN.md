# Drawdown Classifier Gate B Failure and Calibration Plan

## 1. Executive Summary

The decision-grade classifier-only run for `target_drawdown_event_20d_3pct` completed successfully.

The artifact bundle was valid.

Gate B failed.

The strongest simple models:

- `regularized_linear`
- `logistic`

showed weak positive ranking signal on ROC AUC, but failed on Brier score relative to the constant event-rate baseline.

This means:

- there may be some ranking information in the current feature set
- but the predicted probabilities are not well calibrated enough to justify an economic overlay

This is not a validated classifier, but it is not a total null either.
The next question is whether the ranking signal can be calibrated or whether it is noise.

No economic overlay should be built yet.

## 2. What Happened

The standalone classifier-only Gate B run evaluated five simple baselines on the offline prepared SPY feature frame:

- `event_rate`
- `rolling_event_rate`
- `logistic`
- `regularized_linear`
- `histgb`

Primary target:

```text
target_drawdown_event_20d_3pct
```

Best simple models:

### `regularized_linear`

- mean test ROC AUC: `0.552627`
- mean test Brier: `0.251150`
- positive AUC folds: `7 / 9`

### `logistic`

- mean test ROC AUC: `0.548617`
- mean test Brier: `0.251957`
- positive AUC folds: `7 / 9`

Key baselines:

### `event_rate`

- mean test ROC AUC: `0.500000`
- mean test Brier: `0.205322`

### `rolling_event_rate`

- mean test ROC AUC: `0.500000`
- mean test Brier: `0.246417`

## 3. Why Gate B Failed

Gate B required more than a weak ranking edge.

The decisive failure was:

- no simple model beat the constant `event_rate` baseline on mean test Brier score

That matters because Brier score directly reflects probability quality.

The current read is:

- the linear models may rank some risky periods above calmer periods
- but the probability outputs are not trustworthy enough yet
- a classifier with weak discrimination and worse-than-trivial calibration is not ready for policy conversion

Strict Gate B result:

```text
FAIL
```

## 4. Why This Is Not a Plumbing Failure

The failure should not be blamed on broken infrastructure.

What worked:

- the standalone classifier runner executed successfully
- all five model bundles were generated
- metadata correctly marked the runs as classifier-only and decision-grade
- no old workflows were triggered
- OOF artifacts, fold details, and summary files were saved for each model

This was a research result, not an execution failure.

## 5. What the Result Does Suggest

This is not a total null.

Reasons:

- `regularized_linear` reached mean test ROC AUC above `0.55`
- `logistic` was just below that level
- both linear models had positive AUC in `7 / 9` folds
- average precision was modestly above base event rate

That suggests there may be some ranking signal.

But the signal is weak, and weak ranking signal is not enough by itself.

The project should not confuse:

```text
slightly-better ranking
```

with:

```text
usable calibrated event probability
```

## 6. Why Calibration Is Now the Right Next Step

The next question is no longer:

```text
Can we add more model complexity?
```

The next question should be:

```text
Are the linear models miscalibrated but still informative, or is the apparent ranking edge mostly noise?
```

That is the correct next step because:

- ROC AUC alone is insufficient
- Brier failure is the main blocker
- policy conversion would depend on thresholded probabilities
- thresholded probabilities are only meaningful if calibration is acceptable

## 7. What Not To Do Next

Do not:

- build an economic overlay yet
- run matched-null overlay tests yet
- jump to deep models
- run Optuna
- add complexity before understanding calibration

The current classifier is not eligible for economic conversion.

## 8. Calibration Diagnostics Plan

The next diagnostics pass should remain classifier-only.

It should focus on:

1. probability calibration by fold
2. reliability / calibration curves
3. bin-level predicted probability vs realized event frequency
4. Brier decomposition if practical:
   - reliability
   - resolution
   - uncertainty
5. threshold sensitivity:
   - precision / recall at multiple cutoffs
   - false negative burden in high-risk bins
6. ranking-vs-calibration comparison:
   - does a monotonic transformation help?
   - or is the edge too weak to matter?

## 9. Recommended Diagnostic Outputs

The next calibration-focused task should produce, at minimum:

- per-model fold calibration CSV
- pooled OOF calibration table
- decile or quantile bucket table:
  - mean predicted probability
  - realized event rate
  - count
- reliability plot data
- Brier comparison table against:
  - `event_rate`
  - `rolling_event_rate`
  - `logistic`
  - `regularized_linear`

Optional but useful:

- isotonic or Platt-style post-hoc calibration comparison on validation-only then tested on held-out folds

## 10. Decision

Current classifier status:

- not a total null
- not validated
- not ready for overlay conversion

Therefore:

```text
No economic overlay should be built from target_drawdown_event_20d_3pct yet.
```

## 11. Final Conclusion

The simple drawdown-risk classifier track has produced a mixed result:

- weak ranking evidence
- poor probability quality versus the simplest baseline

That means the right next step is calibration and probability diagnostics, not model complexity and not economic deployment.
