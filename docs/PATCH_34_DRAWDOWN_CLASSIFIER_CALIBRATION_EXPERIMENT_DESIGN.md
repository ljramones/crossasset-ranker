# Drawdown Classifier Calibration Experiment Design

## 1. Executive Summary

Gate B failed because probability quality was poor even though the best linear models showed weak positive ranking signal.

The next experiment should test calibration methods on top of the current linear classifiers, not economic overlays and not more complex models.

The goal is:

- improve probability calibration
- improve Brier score and log loss relative to trivial baselines
- preserve as much ranking signal as possible

This is a calibration rescue attempt for a weak ranking signal, not a strategy validation.

No economic overlay should be built unless calibrated probabilities pass the classifier gate first.

## 2. Why This Is the Right Next Step

The current state is:

- `regularized_linear` and `logistic` showed weak AUC signal
- both models materially overpredicted drawdown-event probability
- pooled mean predicted probability was around `0.46`
- realized event rate was around `0.28`
- pooled ECE was around `0.18` to `0.19`
- Brier score remained worse than the constant `event_rate` baseline

That means the immediate bottleneck is not model expressiveness. It is probability reliability.

The next question is:

```text
Can the weak ranking signal be converted into usable calibrated probabilities?
```

If the answer is no, then the current target-feature setup is probably not strong enough to justify any downstream policy layer.

## 3. Files and Outputs Inspected

Key inputs to this design:

```text
docs/PATCH_31_DECISION_GRADE_DRAWDOWN_CLASSIFIER_RUN.md
docs/PATCH_32_DRAWDOWN_CLASSIFIER_GATE_B_FAILURE_AND_CALIBRATION_PLAN.md
docs/PATCH_33_DRAWDOWN_CLASSIFIER_CALIBRATION_DIAGNOSTICS_REPORT.md

evaluation/drawdown_classification.py
evaluation/calibration_diagnostics.py
experiments/drawdown_risk_classifier_experiment.py

results/drawdown_classifier_decision_grade_run/
results/drawdown_classifier_calibration_diagnostics/
```

Important observations from those artifacts:

- the useful candidates are `logistic` and `regularized_linear`
- `histgb` did not provide a strong enough ranking advantage to justify calibration-first priority
- the current failure is strongly consistent with overconfidence, not just random threshold misspecification

## 4. Scope of the Calibration Experiment

This experiment should remain classifier-only.

It should not:

- build a trading overlay
- evaluate executed positions
- run matched-null overlay tests
- introduce deep models
- introduce sequence models
- introduce Optuna
- touch `main.py` or old model-zoo workflows

It should only answer whether post-hoc calibration can improve probability quality for the weak linear classifiers.

## 5. Candidate Base Models

Calibrate these base models only:

- `logistic`
- `regularized_linear`

Keep these as reference baselines:

- `event_rate`
- `rolling_event_rate`

Do not expand the model set yet.

## 6. Calibration Methods To Test

Test a small, explicit calibration set:

- identity / uncalibrated baseline
- Platt scaling
- isotonic regression

Optional later extension only if needed:

- beta calibration

The first calibration experiment should stop at Platt and isotonic.

## 7. Fold-Safe Calibration Protocol

Calibration must respect the walk-forward structure.

For each walk-forward split:

1. Fit the base classifier on train only.
2. Generate raw probabilities on validation.
3. Fit the calibrator on validation predictions and validation labels only.
4. Apply the calibrator to test probabilities only.
5. Score the calibrated test probabilities.

This means:

- no test labels may be used in calibration fitting
- no pooled OOF calibration fit across all folds
- no leakage from future folds

The calibration experiment must remain fold-local and live-style.

## 8. Primary Metrics

Primary success metrics:

- mean test Brier score
- mean test log loss
- mean test ROC AUC
- mean test average precision

Calibration-specific diagnostics:

- expected calibration error
- maximum calibration error
- calibration slope, if practical
- calibration intercept, if practical
- mean predicted probability versus realized event rate

Secondary stability metrics:

- number of positive-AUC folds
- fold-to-fold standard deviation of AUC
- fold-to-fold standard deviation of Brier

## 9. Gate Definition

The calibration experiment should use a stricter gate than “AUC improved slightly.”

Recommended Gate C:

1. A calibrated linear model must beat `event_rate` on mean test Brier score.
2. A calibrated linear model must beat `rolling_event_rate` on mean test Brier score.
3. A calibrated linear model must preserve useful ranking:
   - mean test ROC AUC should remain at or above roughly the uncalibrated level
   - a large AUC collapse should count as failure
4. Calibration improvement should not be driven by one fold only.
5. Mean predicted probability should move materially closer to the realized event rate unless the target is genuinely time-varying in a fold-consistent way.

Recommended interpretation:

- pass: better Brier and acceptable AUC retention
- fail: Brier still worse than `event_rate`, or AUC collapses materially

## 10. Expected Output Bundle

The calibration experiment should write a standalone bundle similar to the classifier runner.

Recommended files:

- `drawdown_classifier_calibration_experiment_summary_<timestamp>.csv`
- `drawdown_classifier_calibration_experiment_fold_details_<timestamp>.csv`
- `drawdown_classifier_calibration_experiment_oof_artifacts_<timestamp>.csv`
- `drawdown_classifier_calibration_experiment_bins_<timestamp>.csv`
- `drawdown_classifier_calibration_experiment_report_<timestamp>.md`
- `drawdown_classifier_calibration_experiment_metadata_<timestamp>.json`

Metadata should include:

- `classification_only: true`
- `economic_overlay_used: false`
- `decision_grade: true/false`
- `base_model_name`
- `calibration_method`
- `target_column`

## 11. Recommended OOF Artifact Fields

The calibration experiment artifacts should include at least:

- `date`
- `split_id`
- `model_name`
- `base_model_name`
- `calibration_method`
- `target`
- `raw_prediction_probability`
- `calibrated_prediction_probability`
- `prediction`
- `asset_return`
- `benchmark_return`

This preserves auditability without implying any trading policy.

## 12. Implementation Notes

The cleanest implementation path is a standalone experiment path adjacent to the current classifier runner.

Suggested modules:

```text
evaluation/probability_calibration.py
experiments/drawdown_classifier_calibration_experiment.py
scripts/run_drawdown_classifier_calibration_experiment.py
```

That path should reuse:

- `evaluation/drawdown_classification.py`
- `evaluation/calibration_diagnostics.py`
- existing walk-forward split definitions

Avoid trying to retrofit this into `main.py` or `prepare_experiment(...)`.

## 13. What Not To Do

Do not:

- jump to deep models
- jump to stacking
- run Optuna
- add economic overlays
- claim success from OOF recalibration that leaks validation or test information
- pool all folds into one global calibrator

The experiment should stay narrow and falsifiable.

## 14. Decision Logic After the Calibration Experiment

If calibration succeeds:

- proceed to one more decision-grade classifier-only run with calibrated probabilities
- only then reconsider whether any policy conversion is justified

If calibration fails:

- conclude that the current drawdown target may contain weak ranking noise but not usable probability signal in this setup
- do not rescue it with immediate complexity
- consider a second target reset:
  - different drawdown horizon
  - different threshold
  - forward risk-adjusted return target
  - cross-asset ranking target

## 15. Current Recommendation

Recommended next implementation:

1. Build fold-safe Platt and isotonic calibration utilities.
2. Run a calibration-only experiment on `logistic` and `regularized_linear`.
3. Compare against `event_rate` and `rolling_event_rate`.
4. Re-evaluate Gate C from Brier/log loss first, then AUC retention.

Until that is done:

```text
No economic overlay should be built from the current drawdown classifier outputs.
```
