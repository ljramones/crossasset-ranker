# Patch 18: Volatility-Quantile Overlay Baseline Design

## 1. Executive Summary

The next experiment should test a **simple volatility-quantile overlay baseline** on top of the existing **volatility-targeted SPY** base strategy.

Plain-language design:

- base strategy: volatility-targeted SPY
- overlay: reduce exposure when trailing realized volatility enters a high-volatility state
- state definition: volatility quantiles estimated from the **training fold only**
- parameter selection: **validation slice only**
- final evaluation: **test slice only**
- null testing: matched exposure, matched turnover, same-vol-state exposure, and block-bootstrap nulls
- success gate: positive active performance versus the vol-targeted baseline **and** significant matched-null results

This experiment does **not** use:

- the failed regime-stacking ensemble
- the failed HMM hard-veto overlay
- model-zoo prediction models
- Optuna
- hardcoded test-period thresholds

The purpose is not to prove alpha immediately. The purpose is to create a clean,
observable, low-complexity baseline against which future HMM or GMM overlays can
be judged.

## 2. Motivation

This is the correct next step because the previous HMM overlay failed clearly:

- it underperformed the volatility-targeted baseline on active metrics
- it failed all matched-null families
- it reduced exposure but increased turnover in an economically unhelpful way

That means:

- the standalone runner worked
- the matched-null framework worked
- the HMM hard-veto hypothesis did not work

A simpler volatility-state baseline should now be tested before trying another
unsupervised regime model.

Why this matters:

1. If a simple volatility-quantile overlay also fails, daily SPY risk-overlay timing may not be promising in this form.
2. If the simple overlay passes, then future HMM/GMM overlays must beat it rather than merely beat raw buy-and-hold or a weak benchmark.
3. Observable trailing volatility is easier to audit than a latent regime state, so it is the right next baseline.

## 3. Files and Functions to Inspect

### `evaluation/regime_overlay.py`

- `build_vol_target_positions(...)`
  - already provides the correct volatility-targeted base position logic using past returns only
- `evaluate_overlay_vs_baseline(...)`
  - already computes overlay-versus-base active metrics
- `summarize_overlay_position_change(...)`
  - already reports exposure and turnover deltas from an overlay

Relevance:

- these functions can be reused directly for the volatility-quantile overlay
- the new overlay should plug into the same base-position and evaluation logic

### `evaluation/null_baselines.py`

- `run_matched_null_suite(...)`
- `same_average_exposure_random(...)`
- `same_turnover_random(...)`
- `same_exposure_and_turnover_random(...)`
- `block_bootstrap_same_exposure_random(...)`
- `same_regime_exposure_random(...)`

Relevance:

- `run_matched_null_suite(...)` already provides the correct evaluation pattern
- the new overlay needs one additional null family conceptually:
  - **same-vol-state exposure random**
- this can likely reuse the same bucketed logic pattern as `same_regime_exposure_random(...)`, but keyed on volatility-state labels instead of HMM regime labels

### `evaluation/audit_artifacts.py`

- `build_standard_audit_artifact_frame(...)`

Relevance:

- future volatility-quantile overlay runs should save audit-ready artifacts in the same standardized style
- overlay-specific per-row fields will still need to be added alongside the standard base columns, as the standalone overlay runner already does

### `experiments/regime_overlay_experiment.py`

- standalone runner structure
- report writer
- artifact writer
- matched-null handling

Relevance:

- the volatility-quantile baseline should reuse this experiment shape:
  - train-only state construction
  - validation-only parameter selection
  - test-only final scoring
  - timestamped artifact bundle

The key difference:

- no HMM fitting
- no latent regime inference
- direct state construction from trailing realized volatility

### `evaluation/metrics.py`

- active-return metrics
- turnover metrics

Relevance:

- active metrics remain the primary decision metrics
- turnover semantics are already repaired and should be used as-is

## 4. Core Hypothesis

### Null hypothesis

```text
A simple volatility-quantile overlay is no better than random de-risking with the same exposure, turnover, and volatility-state profile.
```

### Alternative hypothesis

```text
Observable high-volatility states identify periods where reducing exposure improves active performance versus a volatility-targeted SPY baseline.
```

This is intentionally narrower and simpler than the failed HMM claim.

## 5. Experiment Shape

### Base strategy

Use the existing volatility-targeted SPY baseline:

```text
base_position_t = clip(target_vol / realized_vol_{t-1}, min_position, max_position)
```

Implementation anchor:

- `evaluation.regime_overlay.build_vol_target_positions(...)`

### Overlay idea

Reduce the base position when trailing realized volatility is in a high-volatility state.

Example hard overlay:

```text
if vol_state_t in high_vol_states:
    final_position_t = base_position_t * risk_multiplier
else:
    final_position_t = base_position_t
```

This keeps the experiment directly comparable to the failed HMM hard-veto overlay.

## 6. Volatility-State Definition

The state definition must be **fold-local** and **train-only**.

### Suggested construction

For each split:

1. Compute trailing realized volatility from returns using past data only.
2. On the **training slice only**, compute quantile cutoffs for realized volatility.
3. Map each train/validation/test row into a volatility state using those train-derived cutoffs.

### Example quantile schemes

Preferred initial scheme:

```text
3 states:
  low_vol
  mid_vol
  high_vol
```

Using train-only cutoffs:

- 33rd percentile
- 67th percentile

Alternative scheme:

```text
2 states:
  normal_vol
  high_vol
```

Using a train-only cutoff:

- 80th percentile

Recommendation:

- start with the simpler 2-state version
- then test the 3-state version only if needed

Reason:

- the purpose is to create a simple benchmark, not another flexible state machine

## 7. Parameter Selection

Parameter selection must remain **validation-only**.

### Candidate parameters

1. volatility-state threshold definition
   - top 20% volatility
   - top 25%
   - top 33%
2. risk multiplier
   - `0.0`
   - `0.25`
   - `0.50`

Optional later extension:

- minimum consecutive high-vol days before cutting

Do **not** add this initially.

The first clean benchmark should stay simple.

### Selection rule

Choose parameters on the validation slice using:

- primary metric: `information_ratio` of overlay versus vol-targeted baseline
- secondary diagnostics:
  - `active_calmar`
  - `annualized_active_return`
  - `daily_turnover`

## 8. Test-Time Evaluation

After validation selection, freeze parameters and evaluate on the **test slice only**.

Primary outputs per fold:

- overlay metrics versus market benchmark
- overlay-versus-base active metrics
- exposure/turnover changes
- matched-null summaries

These should mirror the current standalone overlay output bundle so the
volatility-quantile baseline is directly comparable to the failed HMM run.

## 9. Matched Null Framework

The volatility-quantile overlay should be judged against matched nulls, not raw Sharpe.

### Required nulls

1. `same_average_exposure_random`
2. `same_turnover_random`
3. `same_exposure_and_turnover_random`
4. `block_bootstrap_same_exposure_random`
5. **same_vol_state_exposure_random**

### Same-vol-state exposure null

This is the key additional null.

Logic:

- if the overlay is cut or de-risked a certain fraction of the time inside the high-vol state
- then random baselines should match that de-risking rate **within that same volatility state**
- but randomly choose which exact dates are cut

This is the volatility-state analogue of:

- `same_regime_exposure_random(...)`

If the simple overlay cannot beat this null, then it is not timing high-vol periods intelligently. It is just reducing exposure in a broad state bucket.

## 10. Success Gate

The volatility-quantile overlay should pass only if all of the following are reasonably satisfied:

1. aggregate `overlay_vs_base_information_ratio > 0`
2. aggregate `overlay_vs_base_annualized_active_return > 0`
3. `overlay_vs_base_active_calmar` improves or is at least acceptable
4. same-vol-state null p-value `<= 0.05`
5. same-turnover or same-exposure-and-turnover p-value `<= 0.05`
6. turnover does not become economically unreasonable
7. performance is not concentrated in one fold only

Failure conditions:

- raw-looking improvement without matched-null significance
- improvement driven only by de-risking in an obvious volatility bucket
- excessive turnover that destroys economic usefulness
- one-fold concentration

## 11. Why This Is A Better Next Baseline

This baseline is useful regardless of whether it passes:

### If it fails

- that is strong evidence that daily SPY risk-overlay timing may not justify additional regime-model complexity
- it becomes harder to justify another round of HMM/GMM experimentation

### If it passes

- it becomes the minimum benchmark for future regime overlays
- HMM/GMM overlays must beat it, not merely beat a weaker comparison

That makes it a valuable baseline in either outcome.

## 12. Suggested Implementation Shape

The safest implementation path is to reuse the standalone overlay runner architecture with a new state-construction branch.

High-level runner steps:

1. Build vol-targeted base positions.
2. Compute trailing realized volatility.
3. Derive train-only volatility quantile cutoffs.
4. Assign vol-state labels to train/validation/test using those cutoffs.
5. Select validation-only overlay parameters.
6. Evaluate on test only.
7. Run matched nulls including same-vol-state exposure null.
8. Save the same standardized output bundle:
   - summary CSV
   - fold details CSV
   - audit artifact CSV
   - matched-null CSV
   - markdown report
   - metadata JSON

## 13. What This Experiment Must Not Do

Do not let this baseline mutate into another complex research branch prematurely.

Specifically:

1. Do not add HMM/GMM at the same time.
2. Do not add soft overlays initially.
3. Do not add Optuna.
4. Do not add many interacting parameters.
5. Do not compare it primarily on raw Sharpe.
6. Do not skip matched-null testing.

## 14. Decision Frame After This Baseline

Once this experiment is implemented and run, the next decision should be:

### If it fails

```text
SPY daily risk-overlay timing looks weak even with a simple observable volatility-state baseline.
Pause this branch or materially reframe the problem.
```

### If it passes

```text
The simple volatility-state overlay becomes the benchmark.
Any future HMM/GMM overlay must beat it on active metrics and matched-null tests.
```

## 15. Bottom Line

The simple volatility-quantile overlay is the correct next baseline because it is:

- observable
- auditable
- fold-local
- train-only in its state construction
- comparable to the failed HMM overlay
- strict enough to support matched-null validation

It is the right next test before spending more research budget on another latent regime model.
