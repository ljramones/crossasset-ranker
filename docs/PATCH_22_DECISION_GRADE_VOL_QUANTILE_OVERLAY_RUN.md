# Decision-Grade Volatility-Quantile Overlay Gate 2 Run

## Executive Summary

The first formal Gate 2 evaluation for the standalone simple volatility-quantile overlay completed successfully on the prepared offline SPY feature frame.

This run used:

- the standalone volatility-quantile overlay runner only
- offline cached/prepared input only
- a multi-split walk-forward structure
- `null_runs=500`

The overlay failed Gate 2.

The failure was broad enough that the conclusion is not ambiguous:

- aggregate overlay-vs-base active performance was negative
- fold-level wins were inconsistent
- matched-null passes were effectively absent
- the overlay reduced exposure but did not demonstrate skill beyond matched random de-risking

This does not support promotion, rescue, or blind tuning of the simple volatility-quantile overlay.

## Command Used

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

## Artifacts

Generated bundle:

- [vol_quantile_overlay_experiment_summary_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_summary_20260510_152512.csv)
- [vol_quantile_overlay_experiment_fold_details_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_fold_details_20260510_152512.csv)
- [vol_quantile_overlay_experiment_audit_artifacts_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_audit_artifacts_20260510_152512.csv)
- [vol_quantile_overlay_experiment_matched_nulls_20260510_152512.csv](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_matched_nulls_20260510_152512.csv)
- [vol_quantile_overlay_experiment_report_20260510_152512.md](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_report_20260510_152512.md)
- [vol_quantile_overlay_experiment_metadata_20260510_152512.json](/Users/larrym/prediction/results/vol_quantile_decision_grade_run/vol_quantile_overlay_experiment_metadata_20260510_152512.json)

## Gate 2 Outcome

### Aggregate Active Performance

- mean `overlay_vs_base_information_ratio = -0.286533`
- median `overlay_vs_base_information_ratio = 0.000000`
- positive IR folds: `4 / 9`
- mean `overlay_vs_base_annualized_active_return = -0.014435`
- positive active-return folds: `4 / 9`
- mean `overlay_vs_base_active_calmar = -0.183084`

Interpretation:

- the overlay did not improve aggregate benchmark-relative performance
- the result was not concentrated in a single catastrophic fold only, but it also was not consistently positive
- the fold distribution is too weak to support the hypothesis

### Exposure and Turnover Effect

- mean base position: `0.760905`
- mean overlay position: `0.651648`
- mean fraction cut days: `0.266755`
- mean base daily turnover: `0.019128`
- mean overlay daily turnover: `0.024017`

Interpretation:

- the overlay cut exposure meaningfully
- turnover increased, though not catastrophically
- exposure reduction alone did not translate into positive active skill

## Matched-Null Results

Pass counts by null family:

- `same_average_exposure_random`: `0 / 9`
- `same_turnover_random`: `0 / 9`
- `same_exposure_and_turnover_random`: `0 / 9`
- `same_vol_state_exposure_random`: `1 / 9`
- `block_bootstrap_same_exposure_random`: `0 / 9`

This is the key failure.

The simple volatility-state overlay did not beat matched random de-risking in a repeatable way. One fold passed `same_vol_state_exposure_random`, but the broader pattern remained decisively negative.

## Gate 2 Verdict

Gate 2 status: `FAIL`

Why it failed:

1. aggregate `overlay_vs_base_information_ratio` was negative
2. aggregate `overlay_vs_base_annualized_active_return` was negative
3. aggregate `overlay_vs_base_active_calmar` was negative
4. matched-null pass rates were effectively zero
5. exposure reduction did not translate into evidence of skill

## Important Reporting Note

There is one reporting inconsistency in the generated artifacts:

- the markdown report and metadata still label the run as `Decision grade: False`

That labeling is inconsistent with the intended use of this run:

- `9` splits
- `null_runs = 500`

This should be treated as a reporting bug, not as evidence in favor of the strategy.

It does not change the research conclusion because the observed Gate 2 metrics and matched-null failures are already clearly negative.

## Comparison to Failed HMM Hard-Veto Overlay

This simpler overlay was the correct next baseline after the HMM hard-veto failure.

Compared with the failed HMM hard-veto overlay:

- complexity was reduced
- no HMM/GMM inference was involved
- observable volatility state was used instead

That was the right scientific control.

However, the simpler overlay also failed Gate 2.

This weakens the broader hypothesis that daily SPY risk-overlay timing based on a simple state-cut rule is currently producing robust active skill in this framework.

## Decision

Do not:

- promote the volatility-quantile overlay
- tune it blindly
- treat it as a champion candidate
- claim production readiness

The result should be recorded as a research failure, not an infrastructure failure.

## Next Implication

The next research step should not be another ad hoc overlay rescue attempt.

The project now has two failed Gate 2 overlay tracks:

- HMM hard-veto overlay
- simple volatility-quantile overlay

That suggests the next hypothesis should be reframed more fundamentally, rather than adding overlay complexity on the same daily SPY timing setup.
