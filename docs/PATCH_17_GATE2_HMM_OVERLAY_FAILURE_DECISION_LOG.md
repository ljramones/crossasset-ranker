# Patch 17: Gate 2 HMM Overlay Failure Decision Log

## 1. Executive Summary

The decision-grade standalone HMM regime-overlay Gate 2 evaluation completed successfully.

The experiment produced valid standalone artifacts, including:

- summary CSV
- fold-details CSV
- audit-artifact CSV
- matched-null CSV
- markdown report
- metadata JSON

The overlay **failed Gate 2**.

The failure was broad across:

- active performance versus the volatility-targeted SPY baseline
- fold-level consistency
- matched-null tests

This is a **research failure, not an infrastructure failure**.

The standalone overlay runner worked.

The matched-null framework worked.

The current HMM hard-veto overlay hypothesis did **not** work.

The current HMM hard-veto overlay should **not** be promoted, and it should **not** be tuned further without a new hypothesis.

## 2. Decision Record

### Evaluated command

```bash
python -m scripts.run_regime_overlay_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/regime_overlay_decision_grade_run \
  --train-size 756 \
  --val-size 252 \
  --test-size 252 \
  --step-size 252 \
  --target-vol 0.10 \
  --realized-vol-window 20 \
  --transaction-cost-bps 2.0 \
  --null-runs 500 \
  --asset-name SPY \
  --model-name hmm_regime_overlay_hard_veto_gate2 \
  --date-column date \
  --model-type hmm
```

### Output bundle

- [regime_overlay_experiment_summary_20260508_195523.csv](/Users/larrym/prediction/results/regime_overlay_decision_grade_run/regime_overlay_experiment_summary_20260508_195523.csv)
- [regime_overlay_experiment_fold_details_20260508_195523.csv](/Users/larrym/prediction/results/regime_overlay_decision_grade_run/regime_overlay_experiment_fold_details_20260508_195523.csv)
- [regime_overlay_experiment_audit_artifacts_20260508_195523.csv](/Users/larrym/prediction/results/regime_overlay_decision_grade_run/regime_overlay_experiment_audit_artifacts_20260508_195523.csv)
- [regime_overlay_experiment_matched_nulls_20260508_195523.csv](/Users/larrym/prediction/results/regime_overlay_decision_grade_run/regime_overlay_experiment_matched_nulls_20260508_195523.csv)
- [regime_overlay_experiment_report_20260508_195523.md](/Users/larrym/prediction/results/regime_overlay_decision_grade_run/regime_overlay_experiment_report_20260508_195523.md)
- [regime_overlay_experiment_metadata_20260508_195523.json](/Users/larrym/prediction/results/regime_overlay_decision_grade_run/regime_overlay_experiment_metadata_20260508_195523.json)

### Final decision

```text
Gate 2 result: FAIL
```

## 3. Why It Failed

### Active metrics failed

Observed aggregate results:

- mean `overlay_vs_base_information_ratio = -1.0145`
- median `overlay_vs_base_information_ratio = -1.0008`
- positive IR folds = `1 / 9`
- mean `overlay_vs_base_annualized_active_return = -0.03245`
- positive active-return folds = `1 / 9`
- mean `overlay_vs_base_active_calmar = -0.5778`

This is not a marginal miss. The overlay underperformed the volatility-targeted baseline on the primary benchmark-relative metrics.

### Matched-null tests failed

Pass counts:

- `same_average_exposure_random`: `0 / 9`
- `same_turnover_random`: `0 / 9`
- `same_exposure_and_turnover_random`: `0 / 9`
- `same_regime_exposure_random`: `0 / 9`
- `block_bootstrap_same_exposure_random`: `0 / 9`

This means the overlay did not beat matched random de-risking under any null family.

### The failure was broad, not concentrated

The failure was not driven by one bad fold overwhelming a mostly good result:

- only `1 / 9` folds had positive overlay-vs-base information ratio
- only `1 / 9` folds had positive overlay-vs-base annualized active return

That matters because it rules out the easy excuse that the overlay “mostly works but had one unlucky period.”

### The overlay reduced exposure, but in an economically unhelpful way

Observed exposure and turnover effects:

- mean base position = `0.7609`
- mean overlay position = `0.6654`
- mean fraction cut days = `0.2690`
- mean base daily turnover = `0.0191`
- mean overlay daily turnover = `0.0844`

So the overlay did cut risk exposure, but it paid for that with much higher turnover and still failed the benchmark-relative and matched-null tests.

## 4. What This Failure Means

The current HMM hard-veto overlay hypothesis should be treated as falsified for this configuration.

More precisely:

- the standalone overlay plumbing is valid
- the fold-local dangerous-regime mapping worked
- the validation-only parameter selection worked
- the live-safe inference path worked
- the matched-null framework worked
- the hypothesis that this HMM hard-veto overlay adds useful active skill versus a volatility-targeted baseline did **not** work

This should change project behavior.

## 5. What The Project Must Stop Doing

Effective immediately, stop doing the following on this track:

1. Do not promote the current HMM hard-veto overlay.
2. Do not tune the current HMM hard-veto overlay blindly.
3. Do not add more overlay complexity just to rescue this failure.
4. Do not interpret exposure reduction by itself as evidence of skill.
5. Do not treat matched-null failure as a minor detail.
6. Do not bypass the matched-null gate for future regime-overlay claims.

## 6. Why This Is Not An Infrastructure Failure

Important clarification:

- the experiment ran end-to-end
- the output bundle was complete
- the audit artifacts were saved
- the matched-null CSV was saved
- the markdown and metadata outputs were saved
- the dangerous regime varied by fold, confirming no hardcoded regime id
- the parameter selection varied by fold, confirming the selection path was live

So the correct interpretation is:

```text
The experiment infrastructure is usable.
The current research hypothesis failed.
```

That distinction must be preserved to avoid future context drift.

## 7. Pivot Guidance

The project should not continue iterating this exact HMM hard-veto idea without a deliberate new hypothesis.

The next track should be chosen from simpler or reframed alternatives, for example:

1. Compare HMM versus GMM under the same standalone overlay framework.
2. Test whether the hard-veto structure is too blunt relative to the vol-target baseline.
3. Inspect fold-level dangerous-regime summaries to understand whether the detector is cutting at the wrong times.
4. Consider simpler regime-risk rules that can also be matched-null tested cleanly.
5. Prefer reframed hypotheses over additional complexity.

What should not happen next:

- “try a more complex overlay”
- “add more thresholds and multipliers”
- “stack another model on top”
- “keep tuning until one run passes”

Those are exactly the behaviors the reset plan was designed to prevent.

## 8. Recommended Next Decision Frame

The next candidate hypothesis should be stated explicitly before implementation.

A good next-step hypothesis should look like:

```text
Given the HMM hard-veto failure, does an alternative regime classifier or a simpler
exposure-reduction rule improve active performance versus volatility-targeted SPY
and beat matched random de-risking?
```

That keeps the research disciplined:

- new hypothesis
- same gate
- same matched-null discipline

## 9. Status Update

Current project status on the regime-overlay branch:

```text
standalone_overlay_runner: working
artifact_bundle: complete
matched_null_gate: working
hmm_hard_veto_overlay: failed Gate 2
promotion_status: rejected
next_step: choose a new hypothesis deliberately
```

## 10. Bottom Line

The first decision-grade standalone HMM regime-overlay Gate 2 run should be recorded as a clear failure.

Not because the system could not run.

Because it ran correctly and the hypothesis did not survive benchmark-relative and matched-null testing.
