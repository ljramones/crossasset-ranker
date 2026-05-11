# Patch 16: Decision-Grade HMM Regime-Overlay Run

## Scope

This was the first **decision-grade Gate 2 evaluation** for the standalone HMM
regime-risk overlay on SPY using the standalone overlay runner only.

Constraints respected:

- standalone runner only
- no `main.py`
- no `prepare_experiment(...)`
- no model-zoo workflow
- no Optuna
- no regime-stacking ensemble path
- offline prepared input only

## Prepared Input Check

Confirmed existing offline prepared input:

- `/private/tmp/regime_overlay_spy_feature_frame.csv`

## Command Used

Executed from repo root:

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

Configuration notes:

- splits: `9`
- null runs per null family per split: `500`
- model type: `hmm`
- baseline: volatility-targeted SPY
- overlay: hard veto

## Output Files

Generated bundle:

- `results/regime_overlay_decision_grade_run/regime_overlay_experiment_summary_20260508_195523.csv`
- `results/regime_overlay_decision_grade_run/regime_overlay_experiment_fold_details_20260508_195523.csv`
- `results/regime_overlay_decision_grade_run/regime_overlay_experiment_audit_artifacts_20260508_195523.csv`
- `results/regime_overlay_decision_grade_run/regime_overlay_experiment_matched_nulls_20260508_195523.csv`
- `results/regime_overlay_decision_grade_run/regime_overlay_experiment_report_20260508_195523.md`
- `results/regime_overlay_decision_grade_run/regime_overlay_experiment_metadata_20260508_195523.json`

## Gate 2 Result

### Verdict

**FAIL**

The fold-local HMM overlay did **not** pass the first Gate 2 evaluation.

## Why It Failed

### 1. Aggregate active performance versus the volatility-targeted baseline was negative

Across the 9 folds:

- mean `overlay_vs_base_information_ratio = -1.0145`
- median `overlay_vs_base_information_ratio = -1.0008`
- positive IR folds: `1 / 9`

- mean `overlay_vs_base_annualized_active_return = -0.03245`
- positive active-return folds: `1 / 9`

- mean `overlay_vs_base_active_calmar = -0.5778`

This is a strong negative result, not a borderline miss.

### 2. Matched-null tests failed across every null family

No matched-null family passed in any fold.

Pass counts:

- `same_average_exposure_random`: `0 / 9`
- `same_turnover_random`: `0 / 9`
- `same_exposure_and_turnover_random`: `0 / 9`
- `same_regime_exposure_random`: `0 / 9`
- `block_bootstrap_same_exposure_random`: `0 / 9`

Average p-values were all far above significance:

- `same_average_exposure_random`: `0.742`
- `same_turnover_random`: `0.800`
- `same_exposure_and_turnover_random`: `0.800`
- `same_regime_exposure_random`: `0.635`
- `block_bootstrap_same_exposure_random`: `0.796`

So the overlay did not beat matched random de-risking.

### 3. The result was not concentrated in one good fold

The failure is broad, not localized:

- 8 of 9 folds had negative overlay-vs-base information ratio
- 8 of 9 folds had negative overlay-vs-base annualized active return

That reduces the chance that this is merely one bad period overwhelming several good ones.

### 4. The overlay reduced exposure, but paid for it with materially higher turnover

Average position change summary:

- mean base position: `0.7609`
- mean overlay position: `0.6654`
- mean position delta: `-0.0955`
- mean fraction of cut days: `0.2690`

Turnover effect:

- mean base daily turnover: `0.0191`
- mean overlay daily turnover: `0.0844`
- mean turnover delta: `+0.0652`

So the overlay did de-risk, but it did so in a way that materially increased
turnover and still failed benchmark-relative matched-null evaluation.

## Additional Observations

### Dangerous regime mapping was fold-local

Observed dangerous regimes by fold:

- `[1, 0, 2, 1, 0, 1, 1, 0, 1]`

This is good operationally because it confirms the experiment did **not**
hardcode one fixed regime id as “bad”.

### Validation-selected parameters varied across folds

Observed threshold selection:

- `[0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.6]`

Observed risk multipliers:

- `[0.5, 0.5, 0.25, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5]`

This is also operationally good: the runner is performing validation-only
selection rather than using a fixed global hardcoded pair.

## Interpretation

This run answers the Gate 2 question clearly:

> Does the fold-local HMM regime-risk overlay improve active performance versus
> a volatility-targeted SPY baseline and pass matched-null tests?

Answer:

**No.**

The overlay reduced exposure, but the reduction did not create useful active
skill relative to matched random de-risking. In this setup, the HMM overlay is
not yet a valid risk classifier for the target objective.

## Practical Conclusion

Do **not** promote the HMM overlay.

Do **not** treat this as a production candidate.

Current status:

- standalone runner: operational
- artifact bundle: complete
- HMM overlay hypothesis on this configuration: **failed**

## Recommended Next Step

The next step should be analytical, not promotional:

1. inspect fold-level dangerous-regime summaries to understand why the overlay
   is cutting at unhelpful times
2. compare HMM versus GMM backend under the same standalone overlay framework
3. consider whether the hard-veto structure is too blunt relative to the
   realized vol-target baseline
4. keep matched-null testing as a mandatory gate for any variant
