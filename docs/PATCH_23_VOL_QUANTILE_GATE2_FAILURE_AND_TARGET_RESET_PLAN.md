# Volatility-Quantile Gate 2 Failure and Target Reset Plan

## 1. Executive Summary

The decision-grade standalone simple volatility-quantile overlay run completed successfully.

The artifact bundle was valid and complete.

The overlay failed Gate 2.

The failure was broad across active metrics and matched-null tests:

- mean `overlay_vs_base_information_ratio = -0.286533`
- median `overlay_vs_base_information_ratio = 0.000000`
- positive IR folds: `4 / 9`
- mean `overlay_vs_base_annualized_active_return = -0.014435`
- positive active-return folds: `4 / 9`
- mean `overlay_vs_base_active_calmar = -0.183084`

Matched-null passes were effectively absent:

- `same_average_exposure_random`: `0 / 9`
- `same_turnover_random`: `0 / 9`
- `same_exposure_and_turnover_random`: `0 / 9`
- `same_vol_state_exposure_random`: `1 / 9`
- `block_bootstrap_same_exposure_random`: `0 / 9`

This is a research failure, not an infrastructure failure.

Combined with the prior HMM hard-veto overlay failure, SPY daily overlay timing is not currently supported in this framework.

This result falsifies the simple volatility-state overlay hypothesis in the current setup.

## 2. What Happened

The formal standalone Gate 2 command used the offline prepared SPY feature frame and the standalone volatility-quantile runner only:

```bash
python -m scripts.run_vol_quantile_overlay_experiment \
  --execute \
  --input-csv /private/tmp/regime_overlay_spy_feature_frame.csv \
  --output-dir results/vol_quantile_decision_grade_run \
  --train-size 756 \
  --val-size 252 \
  --test-size 252 \
  --step-size 252 \
  --target-vol 0.10 \
  --realized-vol-window 20 \
  --transaction-cost-bps 2.0 \
  --null-runs 500 \
  --asset-name SPY \
  --model-name vol_quantile_overlay_gate2 \
  --date-column date
```

The run produced a complete artifact bundle:

- [vol_quantile_overlay_experiment_summary_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_summary_20260510_152512.csv)
- [vol_quantile_overlay_experiment_fold_details_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_fold_details_20260510_152512.csv)
- [vol_quantile_overlay_experiment_audit_artifacts_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_audit_artifacts_20260510_152512.csv)
- [vol_quantile_overlay_experiment_matched_nulls_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_matched_nulls_20260510_152512.csv)
- [vol_quantile_overlay_experiment_report_20260510_152512.md](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_report_20260510_152512.md)
- [vol_quantile_overlay_experiment_metadata_20260510_152512.json](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_metadata_20260510_152512.json)

This confirms the evaluation path worked end to end.

## 3. Why This Is Not a Plumbing Failure

The failure should not be blamed on missing artifacts or broken reporting.

What worked correctly:

- the standalone runner executed successfully
- the multi-split evaluation completed
- the output bundle was complete and parseable
- matched-null evaluation ran with `500` Monte Carlo draws
- fold-level and summary-level metrics were saved
- artifact-only audit fields were present

There was one minor reporting inconsistency:

- the metadata/report still labeled the run as `Decision grade: False`

That is a reporting bug, not a research-saving issue. It does not alter the negative Gate 2 result.

## 4. Why the Overlay Failed

The overlay reduced exposure, but not intelligently enough to beat matched random de-risking.

Observed exposure and turnover effects:

- mean base position: `0.760905`
- mean overlay position: `0.651648`
- mean fraction cut days: `0.266755`
- mean base daily turnover: `0.019128`
- mean overlay daily turnover: `0.024017`

Interpretation:

- the overlay clearly did something
- the result was not “no effect”
- but the effect was not skillful in active-return terms

The overlay acted like a blunt de-risking rule rather than a robust source of timing skill.

## 5. Matched-Null Interpretation

The matched-null framework did exactly what it was supposed to do.

The central validation question was:

```text
Does the overlay beat dumb random de-risking with the same exposure, turnover, or state-exposure profile?
```

The answer was no.

This is the critical result. Even where fold-level active performance was mildly positive, the overlay did not survive matched-null comparison in a repeatable way.

That means the volatility-state rule did not add enough information beyond simple random reductions in exposure.

## 6. Combined Interpretation with the HMM Failure

There are now two formal Gate 2 failures:

1. HMM hard-veto overlay failure
2. simple volatility-quantile overlay failure

This matters because the two approaches span both:

- a more complex latent-state overlay
- a simpler observable-state overlay

Both failed.

This weakens the broader idea that daily SPY risk-overlay timing is the right formulation in the current framework.

The correct conclusion is not:

```text
Try a slightly different overlay threshold.
```

The correct conclusion is:

```text
The current daily SPY overlay timing research track is not earning the right to continue as the primary path.
```

## 7. Decision

Do not:

- promote the volatility-quantile overlay
- rescue it with incremental threshold tweaking
- add more overlay complexity on the same formulation
- interpret exposure reduction itself as evidence of skill
- treat either overlay path as a production candidate

SPY daily risk-overlay timing should be deprioritized in this setup.

## 8. What This Means for the Reset Plan

The project should now move to target reset.

That means shifting away from:

- daily SPY direction prediction
- daily SPY overlay timing as the main alpha path

And toward targets where “always long” or “cut exposure sometimes” are not easy cheat codes.

Priority reset directions already identified in the first-principles plan:

1. future drawdown risk
2. forward risk-adjusted return
3. cross-asset relative ranking
4. future volatility forecasting

## 9. Recommended Next Research Order

The next track should be deliberately simpler and better aligned to the failure evidence.

Recommended order:

1. Formal target-reset design note for the next label family.
2. Choose one primary new target for implementation.
3. Start with simple models only.
4. Require matched-null success before any complex model is reintroduced.

The best current candidates are:

- future drawdown risk
- cross-asset relative ranking

Those are the most consistent with the evidence that the system may be better suited to risk or allocation decisions than daily SPY timing.

## 10. New Guardrail

Adopt this rule going forward:

```text
No further daily SPY overlay rescue experiments until a new target hypothesis is documented first.
```

This prevents more local tuning on a track that has already failed two formal Gate 2 evaluations.

## 11. Final Status Statement

Current status:

- original regime-stacking ensemble: failed matched-null validation
- HMM hard-veto overlay: failed Gate 2
- simple volatility-quantile overlay: failed Gate 2

Therefore:

```text
The project should stop trying to prove daily SPY timing skill with overlay variants and move to target reset.
```
