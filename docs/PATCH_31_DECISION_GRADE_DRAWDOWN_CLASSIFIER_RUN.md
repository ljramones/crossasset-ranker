# Decision-Grade Drawdown-Risk Classifier Run

## Executive Summary

The first decision-grade classifier-only Gate B evaluation completed successfully on the offline prepared SPY feature frame for:

```text
target_drawdown_event_20d_3pct
```

Only the standalone drawdown-risk classifier runner was used.

No economic overlay was evaluated.

Five simple baselines were run:

- `event_rate`
- `rolling_event_rate`
- `logistic`
- `regularized_linear`
- `histgb`

Main result:

- the linear models showed modest discrimination signal
- `regularized_linear` was the strongest simple baseline on mean test ROC AUC
- but no model beat the constant event-rate baseline on mean test Brier score
- therefore the result is not yet a clean Gate B pass

Strict Gate B verdict:

```text
FAIL / borderline-promising, but not sufficient to move to economic overlay work yet
```

This is not a trading result.

## Commands Used

Each model was run in the standalone classifier path with decision-grade metadata.

Example command pattern:

```bash
python -m scripts.run_drawdown_risk_classifier_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/drawdown_classifier_decision_grade_run/<model_name> \
  --date-column date \
  --price-column "Adj Close" \
  --target-column target_drawdown_event_20d_3pct \
  --model-name <model_name> \
  --asset-name SPY \
  --run-purpose decision_grade \
  --decision-grade \
  --train-size 756 \
  --val-size 252 \
  --test-size 252 \
  --step-size 252
```

Models executed:

- `event_rate`
- `rolling_event_rate`
- `logistic`
- `regularized_linear`
- `histgb`

## Output Bundles

Each model wrote the full standalone classifier bundle under:

- [event_rate](/Users/larrym/prediction/results/drawdown_classifier_decision_grade_run/event_rate)
- [rolling_event_rate](/Users/larrym/prediction/results/drawdown_classifier_decision_grade_run/rolling_event_rate)
- [logistic](/Users/larrym/prediction/results/drawdown_classifier_decision_grade_run/logistic)
- [regularized_linear](/Users/larrym/prediction/results/drawdown_classifier_decision_grade_run/regularized_linear)
- [histgb](/Users/larrym/prediction/results/drawdown_classifier_decision_grade_run/histgb)

All bundles included:

- summary CSV
- fold-details CSV
- OOF artifact CSV
- markdown report
- metadata JSON

Metadata correctly marked:

- `decision_grade: true`
- `run_purpose: decision_grade`
- `classification_only: true`
- `economic_overlay_used: false`
- `trading_strategy_validated: false`

## Model Comparison

### Aggregate Comparison

| Model | Mean Test ROC AUC | Mean Test Brier | Positive AUC Folds | Std Test AUC | AP | AP - Base Rate | Mean Test Positive Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `regularized_linear` | `0.552627` | `0.251150` | `7 / 9` | `0.130057` | `0.321762` | `0.040898` | `0.414903` |
| `logistic` | `0.548617` | `0.251957` | `7 / 9` | `0.125031` | `0.321776` | `0.040911` | `0.407407` |
| `event_rate` | `0.500000` | `0.205322` | `0 / 9` | `0.000000` | `0.280989` | `0.000125` | `0.000000` |
| `rolling_event_rate` | `0.500000` | `0.246417` | `0 / 9` | `0.000000` | `0.288744` | `0.007879` | `0.333333` |
| `histgb` | `0.488414` | `0.244230` | `3 / 9` | `0.107250` | `0.293503` | `0.012639` | `0.188713` |

Observed base event rate across OOF test rows:

- `0.280864`

### Interpretation

What looks encouraging:

- `regularized_linear` and `logistic` both beat the trivial baselines on mean test ROC AUC
- both linear models had positive test AUC in `7 / 9` folds
- both achieved average precision about `0.041` above the base event rate
- positive prediction rates were not degenerate

What is still weak:

- neither linear model beat the constant event-rate baseline on mean test Brier score
- both linear models were only modestly above the `0.55` AUC threshold, and only one cleared it
- `histgb` underperformed the linear models

## Gate B Assessment Against Criteria

### Criterion 1

```text
simple model beats constant event-rate baseline on mean test ROC AUC
```

Result:

- `logistic`: yes
- `regularized_linear`: yes
- `histgb`: no

### Criterion 2

```text
simple model beats constant event-rate baseline on mean test Brier score
```

Result:

- no

The constant event-rate baseline had the best Brier score:

- `event_rate = 0.205322`

All learned models were worse on Brier.

### Criterion 3

```text
simple model beats rolling event-rate baseline on mean test ROC AUC or Brier score
```

Result:

- `logistic`: yes on ROC AUC, no on Brier
- `regularized_linear`: yes on ROC AUC, no on Brier
- `histgb`: no on ROC AUC, yes on Brier

### Criterion 4

```text
mean test ROC AUC > 0.55
```

Result:

- `regularized_linear = 0.552627`: yes, barely
- `logistic = 0.548617`: no, narrowly below

### Criterion 5

```text
average precision exceeds base event rate by meaningful margin
```

Result:

- `logistic`: `+0.040911`
- `regularized_linear`: `+0.040898`

This is positive, but modest rather than decisive.

### Criterion 6

```text
performance is not driven by one fold only
```

Result:

- `logistic`: `7 / 9` positive-AUC folds
- `regularized_linear`: `7 / 9` positive-AUC folds

That is materially better than a one-fold fluke.

### Criterion 7

```text
predicted positive rate is not degenerate
```

Result:

- `logistic`: `0.407407`
- `regularized_linear`: `0.414903`

These are not degenerate.

## Fold Stability Notes

The linear baselines were not uniformly strong.

Examples:

- `logistic` had weak folds at:
  - split `5`: test AUC `0.428162`
  - split `6`: test AUC `0.279514`
- `regularized_linear` had weak folds at:
  - split `5`: test AUC `0.413696`
  - split `6`: test AUC `0.274816`

So although performance was not driven by only one fold, it also was not uniformly robust.

## Decision

Strict reading:

- Gate B does not pass yet

Why:

1. no model beat the constant event-rate baseline on mean test Brier score
2. the ROC AUC edge is modest
3. the best model only barely cleared the `0.55` threshold

More nuanced read:

- the linear models may contain real but modest event-discrimination signal
- that signal is not yet strong enough to justify overlay conversion or economic claims

## What This Means Next

Do not:

- convert this classifier into an economic overlay yet
- claim that drawdown prediction is proven
- jump to complex models yet

Reasonable next steps, still within the classifier-only stage:

1. inspect probability calibration more directly
2. compare baseline-only features vs advanced+VIX features explicitly if not already separated
3. consider a second target sensitivity run on:
   - `target_drawdown_event_20d_2pct`
   - or `target_drawdown_event_10d_3pct`
4. only if classifier evidence strengthens should the project move to a policy-conversion stage

## Final Conclusion

The classifier-only Gate B run was operationally successful and scientifically informative.

The best simple models:

- `regularized_linear`
- `logistic`

showed modest predictive signal for `target_drawdown_event_20d_3pct`, but not enough to pass a strict classifier Gate B.

Therefore:

```text
The project has not yet earned the right to convert drawdown-risk classification into an economic exposure rule.
```
