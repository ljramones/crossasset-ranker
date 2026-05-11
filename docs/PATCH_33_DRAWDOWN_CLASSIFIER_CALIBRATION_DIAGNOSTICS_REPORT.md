# Drawdown Classifier Calibration Diagnostics

## Executive Summary

Calibration diagnostics were run on the existing decision-grade drawdown-classifier OOF prediction artifacts only. No models were retrained, refit, or tuned.

The main result is:

- `regularized_linear` and `logistic` still show weak ranking signal
- but both models are materially miscalibrated
- both models systematically overpredict drawdown-event probability versus the realized event rate
- this supports the Gate B interpretation: there may be some weak ranking information, but the current probabilities are not trustworthy enough for any economic overlay stage

This is not a trading result. It is a classifier-diagnostics result only.

## Inputs

Diagnostics were computed from the existing decision-grade classifier artifacts under:

- [drawdown_classifier_decision_grade_run](/Users/larrym/prediction/results/drawdown_classifier_decision_grade_run)

New calibration outputs were written to:

- [drawdown_classifier_calibration_diagnostics](/Users/larrym/prediction/results/drawdown_classifier_calibration_diagnostics)

Primary generated files:

- [drawdown_classifier_calibration_summary_20260510_191936.csv](/Users/larrym/prediction/results/drawdown_classifier_calibration_diagnostics/drawdown_classifier_calibration_summary_20260510_191936.csv)
- [drawdown_classifier_calibration_by_fold_20260510_191936.csv](/Users/larrym/prediction/results/drawdown_classifier_calibration_diagnostics/drawdown_classifier_calibration_by_fold_20260510_191936.csv)
- [drawdown_classifier_calibration_bins_20260510_191936.csv](/Users/larrym/prediction/results/drawdown_classifier_calibration_diagnostics/drawdown_classifier_calibration_bins_20260510_191936.csv)
- [drawdown_classifier_calibration_report_20260510_191936.md](/Users/larrym/prediction/results/drawdown_classifier_calibration_diagnostics/drawdown_classifier_calibration_report_20260510_191936.md)
- [drawdown_classifier_calibration_metadata_20260510_191936.json](/Users/larrym/prediction/results/drawdown_classifier_calibration_diagnostics/drawdown_classifier_calibration_metadata_20260510_191936.json)

## Pooled Results

Overall base event rate:

- realized event rate: `0.280864`

Pooled model diagnostics:

- `logistic`
  - ROC AUC: `0.546673`
  - average precision: `0.321776`
  - Brier score: `0.251957`
  - mean predicted probability: `0.457988`
  - expected calibration error: `0.184193`
  - maximum calibration error: `0.414683`

- `regularized_linear`
  - ROC AUC: `0.543432`
  - average precision: `0.321762`
  - Brier score: `0.251150`
  - mean predicted probability: `0.463605`
  - expected calibration error: `0.188082`
  - maximum calibration error: `0.381251`

- `event_rate`
  - ROC AUC: `0.476715`
  - Brier score: `0.205322`
  - mean predicted probability: `0.281893`
  - expected calibration error: `0.099794`

- `rolling_event_rate`
  - ROC AUC: `0.504123`
  - Brier score: `0.246417`
  - mean predicted probability: `0.333333`
  - expected calibration error: `0.217372`

Interpretation:

- the linear models have a small ranking edge over trivial baselines
- but they are badly miscalibrated in absolute probability space
- the average predicted probability for both linear models is about `0.46`, far above the realized event rate of about `0.28`
- this overprediction is large enough to explain why Brier score remains worse than the constant event-rate baseline

## Bin-Level Calibration Pattern

The pooled calibration tables show a clear pattern for both `logistic` and `regularized_linear`:

- low-probability bins are only slightly underconfident or near fair
- middle and high-probability bins are systematically too high
- the top decile probabilities are especially overstated

Examples:

- `logistic` top bin:
  - mean predicted probability: `0.758295`
  - observed event rate: `0.343612`
  - signed gap: `-0.414683`

- `regularized_linear` top bin:
  - mean predicted probability: `0.738079`
  - observed event rate: `0.356828`
  - signed gap: `-0.381251`

This is not a subtle calibration miss. It is a strong overconfidence pattern.

## Fold-Level Stability

The fold-level picture is mixed:

- both `logistic` and `regularized_linear` had several folds with AUC above `0.55`
- but both also had clearly weak or negative folds

Notable weak folds:

- `logistic`
  - split `5`: AUC `0.428162`
  - split `6`: AUC `0.279514`

- `regularized_linear`
  - split `5`: AUC `0.413696`
  - split `6`: AUC `0.274816`

Calibration was also poor across many folds:

- `logistic` fold ECE ranged up to `0.395448`
- `regularized_linear` fold ECE ranged up to `0.394491`

This means the result is not just one bad pooled calibration statistic. The probability quality is unstable at the fold level too.

## Decision

Current decision:

- do not build an economic overlay from these probabilities
- do not treat the current classifier outputs as decision-grade risk probabilities
- do not respond by adding model complexity first

The right next question is:

```text
Is the weak ranking signal calibratable, or is it mostly fold-level noise?
```

## Recommended Next Diagnostics

The next calibration-focused work should stay strictly in the classifier layer:

1. Reliability and lift analysis by fold and pooled OOF predictions.
2. Probability calibration methods on top of OOF predictions or fold-local validation predictions only:
   - Platt scaling
   - isotonic regression
3. Threshold-free ranking diagnostics:
   - lift in top risk deciles
   - recall captured in top predicted-risk buckets
4. Compare calibrated linear models against:
   - constant event-rate baseline
   - rolling event-rate baseline
5. Re-run Gate B only after calibration diagnostics show a meaningful improvement in Brier score without destroying AUC.

## Boundary Conditions

This task did not:

- retrain models
- refit models
- run economic overlays
- run `main.py`
- run `prepare_experiment(...)`
- run Optuna
- touch old model-zoo or ensemble workflows

This was an artifact-only calibration diagnostics pass.
