# Patch 10: Regime Overlay Integration Design

## 1. Executive Summary

The next experiment should test a **fold-local HMM/regime risk overlay** on top of a **volatility-targeted SPY baseline**.

Plain-language design:

- baseline: build a volatility-targeted SPY position series using past returns only
- regime detector: fit the existing regime detector on the **training window only**
- overlay: reduce baseline exposure when the **fold-local dangerous regime** is active
- parameter selection: choose overlay parameters on the **validation slice only**
- final evaluation: score the locked overlay on the **test slice only**
- success gate: require positive active performance versus the volatility-targeted baseline **and** significant matched-null results

This design does **not** use the failed `RegimeStackingEnsemble`.

This design does **not** claim alpha until matched-null tests pass.

This design does **not** hardcode any regime ID such as `regime_id == 1`.

Research hypothesis:

```text
Null hypothesis:
  The HMM/regime overlay is no better than random de-risking with the same exposure, turnover, and regime-exposure profile.

Alternative hypothesis:
  The regime layer identifies risk states where reducing exposure improves active performance versus a volatility-targeted SPY baseline.
```

## 2. Files and Functions Inspected

### `regime/regime_detection.py`

- `MarketRegimeDetector.fit` at `regime/regime_detection.py:63`
  - fits scaler and detector on training-window features only
- `MarketRegimeDetector.predict` at `regime/regime_detection.py:86`
  - transforms a full frame and emits labels/probabilities
  - fills missing values by forward/backward filling probabilities and labels
- `MarketRegimeDetector.identify_best_regime` at `regime/regime_detection.py:112`
  - ranks regimes by regime-conditional `net_sharpe`
  - useful precedent for train-only regime scoring, but not the right objective for the overlay track
- `MarketRegimeDetector._predict_raw` at `regime/regime_detection.py:168`
  - uses backend `predict` / `predict_proba`
- `MarketRegimeDetector._build_canonical_mapping` at `regime/regime_detection.py:185`
  - maps raw states into fixed ids via train-window average return / realized vol
  - current semantics assume bull/bear/high-vol style canonicalization

### `main.py`

- CLI flags at `main.py:54-68`
  - no overlay-specific flag exists yet
- `_run_regime_workflow` at `main.py:199`
  - current regime experiments are stacker-centric, not overlay-centric
- `_build_regime_aware_experiment` at `main.py:984`
  - augments each split with train-fitted regime labels and probabilities
  - demonstrates the right place to attach fold-local regime annotation
- `_print_regime_weighted_summary` at `main.py:1595`
  - current reporting remains ensemble-oriented
- `_build_regime_summary` at `main.py:1692`
  - current summary logic is also ensemble-oriented

### `evaluation/walk_forward.py`

- `WalkForwardSplit` at `evaluation/walk_forward.py:11`
- `generate_walk_forward_splits` at `evaluation/walk_forward.py:20`
  - creates strictly ordered train / validation / test windows
  - this structure is exactly what the overlay experiment should preserve

### `utils/experiment.py`

- `prepare_experiment` at `utils/experiment.py:114`
  - loads config, data, features, splits
- `prepare_experiment_from_market_data` at `utils/experiment.py:141`
  - constructs feature set and walk-forward splits
- `evaluate_model` at `utils/experiment.py:203`
  - current evaluation path is model-centric
  - not the right abstraction for the overlay, because the overlay is a position transform, not a predictive model
- standardized OOF artifact creation after the recent patch
  - now uses `build_standard_audit_artifact_frame(...)` when constructing `oof_predictions`

### `evaluation/regime_overlay.py`

- `build_vol_target_positions` at `evaluation/regime_overlay.py:42`
- `characterize_regimes` at `evaluation/regime_overlay.py:87`
- `compute_regime_danger_scores` at `evaluation/regime_overlay.py:137`
- `identify_dangerous_regime` at `evaluation/regime_overlay.py:155`
- `extract_dangerous_regime_probability` at `evaluation/regime_overlay.py:164`
- `apply_hard_regime_overlay` at `evaluation/regime_overlay.py:186`
- `build_hard_overlay_parameter_grid` at `evaluation/regime_overlay.py:206`
- `apply_soft_regime_overlay` at `evaluation/regime_overlay.py:225`
- `evaluate_overlay_strategy` at `evaluation/regime_overlay.py:268`
- `evaluate_overlay_vs_baseline` at `evaluation/regime_overlay.py:284`

### `evaluation/null_baselines.py`

- `run_matched_null_suite` at `evaluation/null_baselines.py:265`
  - usable for overlay null testing once canonical executed positions are built

### `evaluation/audit_artifacts.py`

- `build_standard_audit_artifact_frame` at `evaluation/audit_artifacts.py:43`
  - important for future overlay artifact persistence

## 3. Key Findings From Inspection

### 3.1 What is already good

- Walk-forward splits are strictly ordered.
- Regime detector fitting is already train-only.
- Overlay utility layer is already available and model-agnostic.
- Active metrics and matched-null utilities are already available.
- Standardized artifact schema now exists for future artifact-only audits.

### 3.2 What cannot be reused as-is

#### `identify_best_regime(...)`

Current behavior:

- chooses the “best” regime by **raw regime-conditional net Sharpe**
- assumes the objective is to find the strongest regime to trade

Overlay requirement:

- we need to identify the **most dangerous** regime
- objective must be risk-centric, not raw Sharpe-centric

Conclusion:

- do **not** reuse `identify_best_regime(...)` directly for the overlay experiment

#### `_build_canonical_mapping(...)`

Current behavior:

- maps raw states into stable ids using return/vol heuristics
- imposes semantic labels like bull / bear / high-vol

Overlay requirement:

- no hardcoded or semantically assumed regime IDs
- dangerous regime must be identified from **fold-local training statistics**

Conclusion:

- overlay logic should not depend on a global canonical state meaning
- it may still consume `regime_id`, but dangerous-regime mapping must be recomputed per fold from train data only

### 3.3 Most important realism risk: regime probabilities

`MarketRegimeDetector.predict(...)` currently calls backend `predict_proba(...)` on the full provided frame.

For the GMM backend, this is effectively pointwise conditional on the current observation and is likely acceptable.

For the HMM backend, `predict_proba(...)` is not obviously guaranteed to be **filtered-only**. It may use full-sequence information inside the provided window.

That means:

- train-window probabilities are fine for characterization
- validation/test probabilities may not yet be live-safe for a true overlay experiment if the HMM path is smoothing across the whole slice

This is the most important unresolved design issue before implementation.

## 4. Safest Experiment Shape

The implementation should be a new, explicit overlay experiment path. It should not be bolted into the failed ensemble flow.

### 4.1 Baseline

Per split:

1. Use `split.train`, `split.validation`, `split.test`.
2. Build a volatility-targeted SPY baseline position series separately for each slice, using only past returns within that slice boundary convention.
3. The first usable positions in each slice should respect lookback warmup and start flat until enough history exists.

Baseline constructor:

- `evaluation.regime_overlay.build_vol_target_positions(...)`

Target baseline concept:

```text
base_position_t = clip(target_vol / realized_vol_{t-1}, min_position, max_position)
```

### 4.2 Regime annotation

Per split:

1. Fit `MarketRegimeDetector` on `split.train` only.
2. Predict regime labels/probabilities for:
   - train
   - validation
   - test
3. Use train-period regimes only to characterize dangerous regimes.

Important:

- no validation or test information may influence dangerous-regime selection

### 4.3 Dangerous regime selection

Per split:

1. Compute training-only regime summary:
   - `characterize_regimes(train_returns, train_regime_labels)`
2. Compute training-only danger score:
   - `compute_regime_danger_scores(...)`
3. Select dangerous regime:
   - `identify_dangerous_regime(...)`

This avoids:

- hardcoded regime ids
- assuming regime `1` is always “bad”

### 4.4 Overlay parameter selection

Initial hard-veto grid:

```text
thresholds = [0.50, 0.60, 0.70]
risk_multipliers = [0.00, 0.25, 0.50]
```

Grid helper:

- `build_hard_overlay_parameter_grid(...)`

Per split:

1. Apply each parameter pair to validation dangerous-regime probabilities:
   - `apply_hard_regime_overlay(...)`
2. Evaluate overlay directly versus the volatility-targeted baseline:
   - `evaluate_overlay_vs_baseline(...)`
3. Rank candidates by:
   - `information_ratio`
   - `active_calmar`
   - drawdown improvement
4. Lock the best validation parameter pair.

No test feedback is allowed during parameter selection.

### 4.5 Final test evaluation

Per split:

1. Apply the locked dangerous-regime mapping and locked overlay parameters to the test slice only.
2. Evaluate:
   - overlay vs market benchmark
   - overlay vs volatility-targeted baseline
3. Save:
   - executed base positions
   - executed overlay positions
   - dangerous-regime probability
   - overlay-specific audit artifacts

## 5. Recommended New Experiment Abstraction

The safest implementation is a new experiment utility, not a modification of existing model evaluation.

Recommended new module:

```text
evaluation/regime_overlay_experiment.py
```

Recommended responsibilities:

- build volatility-targeted base positions
- fit train-only regime detector
- characterize train-only dangerous regime
- run validation-only overlay parameter search
- evaluate test-only overlay
- run matched-null suite on executed overlay positions
- save standardized overlay audit artifacts

Recommended high-level function:

```text
run_regime_overlay_experiment(
    experiment,
    target_vol,
    realized_vol_window,
    thresholds,
    risk_multipliers,
    transaction_cost_bps,
)
```

This should return a structure similar in spirit to `ModelArtifacts`, but for overlay experiments.

## 6. Proposed Per-Split Flow

Per `WalkForwardSplit`:

```text
train / validation / test
    -> fit regime detector on train only
    -> annotate train / validation / test with regime labels + probabilities
    -> build vol-target baseline positions
    -> characterize train regimes only
    -> identify dangerous regime from train only
    -> search hard-overlay params on validation only
    -> lock best params
    -> apply overlay to test only
    -> evaluate overlay vs baseline and benchmark
    -> run matched nulls on test overlay positions
    -> save audit artifact rows
```

## 7. Probability Safety Decision

Before implementation, the project must explicitly choose one of these paths.

### Option A: GMM-only overlay experiment

Safest near-term option.

Reason:

- current GMM probabilities are observation-local
- avoids ambiguity around HMM smoothing

Tradeoff:

- not a true HMM overlay yet

### Option B: HMM with verified filtered probabilities

Preferred long-term option.

Requirement:

- add a live-safe filtered-probability prediction path
- confirm it does not use future observations inside validation/test slices

Tradeoff:

- more implementation work

Recommended decision:

- start with **GMM-only overlay experiment** unless the HMM probability path is upgraded first

## 8. Null-Test Design For Overlay

The overlay must be evaluated against matched nulls, not just raw performance.

Canonical executed positions:

- the final **overlay positions on the test slice**

Null suite:

- `same_average_exposure_random`
- `same_turnover_random`
- `same_exposure_and_turnover_random`
- `same_regime_exposure_random`
- `block_bootstrap_same_exposure_random`

Null helper:

- `run_matched_null_suite(...)`

Most important null:

- `same_regime_exposure_random`

Interpretation:

- if the overlay cannot beat random de-risking with the same per-regime exposure rates, then the regime layer is not timing intelligently

## 9. Artifact Requirements For The Overlay

The overlay experiment should save a standardized OOF-like artifact per split/test row with at least:

- `date`
- `split_id`
- `model_name`
- `asset`
- `asset_return`
- `benchmark_return`
- `raw_signal`
- `prediction_probability`
- `target` if applicable
- `executed_position`
- `strategy_gross_return`
- `strategy_net_return`
- `turnover`
- `transaction_cost`
- `regime_id`
- `regime_prob_0`
- `regime_prob_1`
- `regime_prob_2`

Overlay-specific additions should also be saved:

- `base_position`
- `dangerous_regime_id`
- `dangerous_regime_probability`
- `overlay_threshold`
- `overlay_risk_multiplier`
- `overlay_mode`
- `is_cut_day`

The existing `build_standard_audit_artifact_frame(...)` should be reused or lightly extended, not bypassed.

## 10. Reporting Requirements

Primary ranking metrics must be:

- `information_ratio`
- `active_calmar`
- `annualized_active_return`
- `active_max_drawdown`
- matched-null `p_value`

Secondary diagnostics:

- `fraction_in_market`
- `daily_turnover`
- `annualized_turnover`
- `position_flip_count`
- `average_holding_period_days`
- `cost_drag`

The overlay should be reported against:

1. market benchmark
2. volatility-targeted baseline
3. matched nulls

## 11. Suggested CLI / Workflow Integration

Do not reuse `--regime` for this.

Recommended future CLI flag:

```text
--regime-overlay
```

Recommended behavior:

- build standard experiment
- run the dedicated overlay experiment path
- save a dedicated comparison file
- save a dedicated overlay audit artifact file
- save a dedicated matched-null summary

This keeps:

- ensemble research
- regime overlay research
- audit workflows

cleanly separated.

## 12. Stop / Go Gates

The overlay experiment should proceed to “interesting candidate” status only if:

1. it improves `information_ratio` versus the volatility-targeted baseline
2. it improves `active_calmar`
3. it reduces active drawdown
4. it passes:
   - `same_average_exposure_random`
   - `same_turnover_random`
   - `same_regime_exposure_random`
5. turnover remains operationally reasonable

If it fails these gates, the current regime layer should not be promoted further.

## 13. Implementation Risks To Address First

### Risk 1: HMM probability leakage

Most important unresolved issue.

Action before implementation:

- verify whether HMM `predict_proba(...)` is filtered or smoothed on validation/test slices

### Risk 2: Warmup handling for volatility targeting

Need explicit convention for:

- first rows without enough realized-vol history
- first row of each validation/test slice

Recommended default:

- stay flat until the lookback is available

### Risk 3: Baseline-relative evaluation confusion

Need to ensure:

- overlay-vs-market metrics
- overlay-vs-vol-target baseline metrics

are both computed and not conflated

### Risk 4: Artifact schema drift

Overlay outputs must use the standardized audit artifact path from the start.

## 14. Recommended Next Implementation Order

1. Verify and decide the probability-safety path:
   - GMM-only first, or filtered-HMM upgrade first
2. Create `evaluation/regime_overlay_experiment.py`
3. Add a dedicated overlay artifact schema extension
4. Add synthetic unit tests for split-local overlay parameter selection
5. Add a read-only smoke test using synthetic walk-forward slices
6. Only then run the first real overlay experiment

## 15. Final Design Position

This is the correct next research track after the failure of regime stacking.

Why:

- it reframes the problem from direction prediction to risk control
- it uses regime information as an overlay, not as a stacker feature
- it aligns with the reset-plan requirement to test whether regime logic adds value beyond matched random de-risking

What it still does **not** prove:

- that the existing regime detector is useful
- that HMM probabilities are live-safe
- that the overlay has alpha

Those claims require the actual experiment and matched-null gates to pass.  
This document only defines the safest path to implement that experiment next.
