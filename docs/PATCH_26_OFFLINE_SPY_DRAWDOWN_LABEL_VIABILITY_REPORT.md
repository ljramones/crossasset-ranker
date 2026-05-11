# Offline SPY Drawdown Label Viability Report

## Executive Summary

An offline drawdown-label viability diagnostic was run on the prepared SPY feature frame:

- input: `/private/tmp/regime_overlay_spy_feature_frame.csv`
- candidate horizons: `10`, `20`
- candidate thresholds: `-2%`, `-3%`, `-5%`
- walk-forward structure:
  - train: `756`
  - validation: `252`
  - test: `252`
  - step: `252`

This was a label diagnostics task only.

No models were trained.

Main conclusion:

- all six candidate drawdown-event labels are statistically viable under the current split sizes
- the originally proposed `target_drawdown_event_20d_3pct` is clearly viable
- the mechanically strongest label by event-count support is `target_drawdown_event_20d_2pct`
- the recommended first classifier target is still `target_drawdown_event_20d_3pct`

Reason for the recommendation:

- `20d_3pct` has strong fold-level viability
- it is materially more “risk-event-like” than `20d_2pct`
- it still has ample positive counts in train, validation, and test
- it better matches the intended research question: elevated drawdown risk, not merely mild weakness

## Command Used

```bash
python -m scripts.run_drawdown_label_viability_report \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/drawdown_label_viability \
  --date-column date \
  --price-column "Adj Close" \
  --horizons 10,20 \
  --thresholds=-0.02,-0.03,-0.05 \
  --train-size 756 \
  --val-size 252 \
  --test-size 252 \
  --step-size 252
```

## Output Files

Created under [drawdown_label_viability](/Users/larrym/prediction/results/drawdown_label_viability):

- [drawdown_label_prevalence_20260510_160222.csv](/Users/larrym/prediction/results/drawdown_label_viability/drawdown_label_prevalence_20260510_160222.csv)
- [drawdown_label_viability_20260510_160222.csv](/Users/larrym/prediction/results/drawdown_label_viability/drawdown_label_viability_20260510_160222.csv)
- [drawdown_label_candidate_summary_20260510_160222.csv](/Users/larrym/prediction/results/drawdown_label_viability/drawdown_label_candidate_summary_20260510_160222.csv)
- [drawdown_label_viability_report_20260510_160222.md](/Users/larrym/prediction/results/drawdown_label_viability/drawdown_label_viability_report_20260510_160222.md)
- [drawdown_label_viability_metadata_20260510_160222.json](/Users/larrym/prediction/results/drawdown_label_viability/drawdown_label_viability_metadata_20260510_160222.json)

## Candidate Label Results

### Overall Prevalence

| Label | Positive Rate | Positive Count | Notes |
| --- | ---: | ---: | --- |
| `target_drawdown_event_10d_5pct` | `0.0743` | `246` | Viable but relatively sparse |
| `target_drawdown_event_20d_5pct` | `0.1525` | `505` | Viable, lower-frequency severe event |
| `target_drawdown_event_10d_3pct` | `0.1842` | `610` | Viable, moderate event frequency |
| `target_drawdown_event_10d_2pct` | `0.2841` | `941` | Viable, broader event |
| `target_drawdown_event_20d_3pct` | `0.2917` | `966` | Viable and well balanced |
| `target_drawdown_event_20d_2pct` | `0.4052` | `1342` | Most common and best-supported mechanically |

None of the candidate labels were too rare or degenerate on the full offline sample.

## Fold-Level Viability

All six labels passed the current viability checks across all `9` walk-forward splits for:

- train slices
- validation slices
- test slices

That means:

- `fraction_viable_splits = 1.0` for every candidate label
- no candidate failed because of insufficient positive or negative counts
- no candidate failed because of pathological positive-rate bounds

## Specific Check: `target_drawdown_event_20d_3pct`

The originally proposed target is viable.

Observed diagnostics:

- overall positive rate: `0.291667`
- overall positive count: `966`
- test-slice mean positive rate: `0.280864`
- test-slice min positive rate: `0.162698`
- test-slice max positive rate: `0.591270`
- test-slice mean positive count: `70.777778`
- test-slice minimum positive count across folds: `41`
- all train, validation, and test splits viable: `True`

This is strong enough to support the first classifier experiment.

## Which Labels Are Less Attractive

### `target_drawdown_event_10d_5pct`

This label is still viable, but it is the weakest of the set.

Why:

- overall positive rate is only `7.43%`
- minimum test positive count is `5`
- it is closest to becoming sparse in some folds

This makes it more fragile as a first supervised target.

### `target_drawdown_event_20d_2pct`

This label is mechanically the easiest to work with.

Why:

- overall positive rate is `40.52%`
- minimum test positive count is `44`
- all splits are comfortably populated

However, it may be too broad for the first research question.

Interpretation:

- it likely captures many mild-to-moderate drawdown episodes
- that is useful for robustness
- but it is slightly less aligned with the intended “elevated drawdown risk” concept than `20d_3pct`

## Recommended Primary Target

Recommended primary target for the first simple classifier experiment:

```text
target_drawdown_event_20d_3pct
```

Recommended secondary sensitivity targets:

1. `target_drawdown_event_20d_2pct`
2. `target_drawdown_event_10d_3pct`

Reasoning:

- `20d_3pct` is severe enough to represent a meaningful risk event
- it remains statistically comfortable across all folds
- it avoids the sparsity risk of `10d_5pct`
- it avoids becoming too broad and common like `20d_2pct`

## Decision

Proceed to the next target-reset step:

- implement a simple-model classifier experiment on `target_drawdown_event_20d_3pct`

Do not:

- jump directly to complex models
- reinterpret this as predictive success
- skip classifier-only evaluation and go straight to an economic overlay

The correct next step is:

1. simple classifier baselines
2. fold-stable event-detection evaluation
3. only then an exposure-rule conversion if classification skill is real

## Final Conclusion

The drawdown-label viability diagnostic was successful.

The label space is not the bottleneck.

The strongest conclusion is:

```text
target_drawdown_event_20d_3pct is viable and should be the primary target for the first simple drawdown-risk classifier experiment.
```
