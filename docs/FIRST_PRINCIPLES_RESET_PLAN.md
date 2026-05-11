# FIRST PRINCIPLES RESET PLAN

## 1. Executive Summary

The current `ml-trading-signal-exploration` framework has passed several important engineering hygiene checks, including leakage audits around feature engineering, train-only scaling, walk-forward splitting, HMM regime fitting, and stacking OOF partitioning.

However, the latest audit and comparative tests show that the original `RegimeStackingEnsemble` champion is not validated alpha.

The original high Sharpe was largely explained by structural long exposure to SPY during a strong equity market period. The ensemble was effectively near-always-long:

- `stacking_ensemble_baseline`:
  - net Sharpe around 1.308
  - average long exposure around 0.996
  - fraction positive predictions around 1.000
- `buy_and_hold_spy_baseline`:
  - net Sharpe around 1.296
  - similar exposure profile

The difference between the ensemble and buy-and-hold was economically and statistically weak. The strategy looked sophisticated but mostly behaved like passive long SPY exposure.

A bias-fix experiment reduced the regime stack’s long exposure from approximately 0.996 to approximately 0.862, but performance worsened:

- `regime_stacking_ensemble_regime` after bias fix:
  - net Sharpe around 1.169
  - information ratio around `-0.166`
  - excess net Sharpe around `-0.073`
  - average long exposure around 0.862

This means the long-bias penalty worked mechanically, but the model’s deviations from passive long exposure were not useful. The model did not know when to reduce exposure.

The current conclusion is:

The framework is valuable.  
The original champion is not validated alpha.  
The current daily SPY directional prediction target is probably the wrong research problem.  
The system should be reframed as a risk-management, regime-detection, active-allocation, or cross-asset decision engine.

## 2. Current Status Classification

Record the current state as follows:

```yaml
champion_v1.0:
  former_claim: RegimeStackingEnsemble alpha model
  revised_status: frozen engineering baseline only
  production_candidate: false
  reason:
    - near-always-long exposure
    - benchmark-relative skill not demonstrated
    - label-shuffle/null tests failed
    - bias-fix reduced long exposure but worsened active performance
    - ensemble appears to harvest beta rather than generate alpha

current_system_interpretation:
  not: validated daily SPY directional alpha model
  likely: volatility-responsive beta allocation / regime-risk research platform
```

Do not continue referring to the current regime stack as a production candidate.

## 3. Strategic Diagnosis

The current return profile should be decomposed as:

```text
strategy return =
    market beta exposure
  + timing skill
  + sizing skill
  + regime/risk avoidance skill
  - costs
  + noise
```

The audit suggests:

- market beta exposure: high
- timing skill: weak, unproven, or negative
- sizing skill: not yet proven
- regime/risk avoidance skill: possible but not yet validated
- cost impact: present but not the core issue
- noise: large enough to make raw Sharpe misleading

The original model mostly learned:

```text
SPY usually goes up, so stay long.
```

The next research question should not be:

```text
Can ML predict daily SPY direction?
```

The next research question should be:

```text
Can the system make better exposure, risk, or allocation decisions than simple baselines and matched random strategies?
```

## 4. What We Must Stop Doing

Stop doing the following until the evaluation layer is fixed:

1. Do not optimize raw Sharpe as the primary objective.
2. Do not rank models primarily by net Sharpe.
3. Do not tune more deep learning architectures.
4. Do not add dynamic position sizing yet.
5. Do not try to rescue the current ensemble with more meta-learner complexity.
6. Do not hardcode HMM regime IDs such as `regime_id == 1`.
7. Do not interpret high standalone Sharpe as alpha unless active-return metrics and matched nulls pass.
8. Do not treat the current `RegimeStackingEnsemble` as a validated production candidate.

## 5. Immediate Priority Order

The correct next sequence is:

1. Fix active-return metrics and reporting.
2. Fix exposure vs turnover reporting.
3. Add matched-exposure and matched-turnover null baselines.
4. Rerun the existing models against the corrected metrics and matched nulls.
5. Test HMM/regime logic as a fold-local risk overlay on a volatility-targeted SPY baseline.
6. Reset the prediction target.
7. Only then revisit model architectures.

## 6. Phase 1: Measurement Repair

### 6.1 Correct Active-Return Metrics

Do not rely on:

```text
excess_net_sharpe = strategy_net_sharpe - benchmark_net_sharpe
```

That value can remain as a rough diagnostic, but it is not the correct active performance metric.

The correct active-return series is:

```text
active_return_t = strategy_return_t - benchmark_return_t
```

Then compute:

```text
information_ratio =
    mean(active_return_t) / std(active_return_t) * sqrt(252)
```

Add the following metrics:

- annualized_active_return
- active_volatility
- tracking_error
- information_ratio
- active_max_drawdown
- active_calmar
- correlation_to_benchmark
- beta_to_benchmark
- alpha_after_beta
- active_return_p_value if null testing is available

The active-return series should become the core object of benchmark-relative evaluation.

### 6.2 Fix Exposure vs Turnover Reporting

The current `trade_frequency` column appears to be mislabeled. Buy-and-hold showing `trade_frequency` near 1.0 means this is likely measuring exposure or time in market, not actual trading frequency.

Split the reporting into separate metrics:

- average_long_exposure
- average_short_exposure
- average_abs_position
- fraction_in_market
- fraction_positive_predictions
- position_flip_count
- daily_turnover
- annualized_turnover
- average_holding_period_days
- round_trip_count
- cost_drag
- cost_per_unit_active_return

Turnover should be calculated from positions:

```text
daily_turnover_t = abs(position_t - position_{t-1})

annualized_turnover = mean(daily_turnover_t) * 252
```

Transaction costs should be charged on turnover:

```text
cost_t = transaction_cost_bps / 10000 * abs(position_t - position_{t-1})
```

This correctly handles:

- `0 -> +1` as 1 unit turnover
- `+1 -> 0` as 1 unit turnover
- `+1 -> -1` as 2 units turnover
- `-1 -> +1` as 2 units turnover

### 6.3 Add Synthetic Metric Tests

Before trusting the revised metrics, add small synthetic test cases.

Test cases should include:

```yaml
always_flat:
  exposure: 0
  turnover: 0
  strategy_return: 0

buy_and_hold:
  exposure: approximately 1
  turnover: approximately 0 after initial entry
  flip_count: 0

alternating_long_flat_daily:
  high_turnover: true
  many_position_changes: true

alternating_long_short_daily:
  very_high_turnover: true
  daily_turnover: approximately 2
```

These tests should catch most metric/reporting mistakes.

## 7. Phase 2: Matched Null Framework

The key validation question is no longer:

```text
Does the strategy have a high Sharpe?
```

The key validation question is:

```text
Does the strategy beat dumb random strategies with the same exposure and turnover profile?
```

Add these null baselines:

- same_average_exposure_random
- same_turnover_random
- same_exposure_and_turnover_random
- same_regime_exposure_random
- block_bootstrap_same_exposure_random

The most important null is:

```text
same_regime_exposure_random
```

Example logic:

If the model is long:

- 91% of the time in fold-local regime A
- 35% of the time in fold-local regime B
- 95% of the time in fold-local regime C

Then generate random strategies that match those rates inside each regime but randomly choose the specific long/flat days.

If the model cannot beat this null, the regime logic is not timing intelligently. It is merely reducing exposure.

For each canonical model or overlay, report:

- canonical_active_IR
- mean_null_active_IR
- 95th_percentile_null_active_IR
- null_p_value
- canonical_active_drawdown
- mean_null_active_drawdown
- canonical_turnover
- mean_null_turnover

A model or overlay should not be considered interesting unless it beats the matched nulls.

## 8. Phase 3: Clean Existing-Model Truth Test

After Phase 1 and Phase 2, rerun the current model set.

Do not rank by raw Sharpe.

Rank by:

- information_ratio
- active_calmar
- annualized_active_return
- active_max_drawdown
- matched-null p-value
- turnover-adjusted active return

The clean comparison report should include:

- cash
- buy_and_hold_spy
- vol_targeted_spy
- simple_momentum
- simple_moving_average_trend
- legacy_regime_stack
- bias_fixed_regime_stack
- plain_stacking
- itransformer_clean_non_interaction
- itransformer_interaction
- same_exposure_random
- same_turnover_random
- same_regime_exposure_random

Expected outcome:

Most or all current ML models will probably fail active-skill tests.

That is acceptable. This phase is designed to close the book cleanly on the original daily-direction formulation.

## 9. Phase 4: Regime-Risk Overlay Experiment

This is the first promising new experiment.

Do not use the HMM merely as another meta-feature in the stacker.

Instead, test the HMM as an independent risk overlay.

### 9.1 Base Strategy

Use a volatility-targeted SPY baseline, not raw buy-and-hold.

Reason:

If the HMM only beats raw buy-and-hold, it may just be rediscovering volatility targeting. If the HMM improves a volatility-targeted baseline, that is more meaningful.

Example base-position logic:

```text
realized_vol_t = rolling realized volatility using past data only

base_position_t = target_vol / realized_vol_t

base_position_t = clip(base_position_t, 0, 1)

base_return_t = base_position_{t-1} * spy_return_t - costs
```

### 9.2 Do Not Hardcode HMM Regime IDs

Never write a rule like:

```text
if regime_id == 1:
    reduce exposure
```

HMM state labels are arbitrary across folds. In one walk-forward fold, `regime_id == 1` may correspond to high-vol stress. In another fold, it may correspond to low-vol trend.

Instead, use fold-local regime characterization.

For each walk-forward fold:

1. Fit HMM on training data only.
2. Infer training-period regime probabilities/states.
3. Characterize each regime using training data only.
4. Rank regimes by risk properties.
5. Define the dangerous regime for that fold.
6. Apply that fold-local regime mapping to the test period.

Possible training-only regime risk statistics:

- average return
- realized volatility
- downside volatility
- max drawdown tendency
- VIX z-score
- trend strength
- negative skew
- percentage of large down days

A simple danger score can be:

```text
danger_score =
    rank(realized_volatility)
  + rank(downside_volatility)
  + rank(drawdown_tendency)
  - rank(average_return)
```

Then:

```text
dangerous_regime = regime with highest danger_score
```

### 9.3 Use Filtered Probabilities, Not Smoothed Test Paths

For live realism:

```text
features through close t
-> filtered regime probability at t
-> position for t+1
```

Avoid using a full test-window Viterbi path if it uses future observations inside the test period.

The system must not let future test-period observations influence today’s regime probability.

### 9.4 Overlay Rules

Start simple.

Hard veto:

```text
if P(dangerous_regime) > threshold:
    final_position = base_position * risk_multiplier
else:
    final_position = base_position
```

Initial grid:

```yaml
threshold:
  - 0.50
  - 0.60
  - 0.70

risk_multiplier:
  - 0.00
  - 0.25
  - 0.50
```

Threshold and multiplier must be selected inside training/validation folds only, not from final test results.

A later soft overlay can be:

```text
final_position =
    base_position * (1 - cut_strength * P(dangerous_regime))
```

Start with the hard version because it is easier to audit.

### 9.5 Regime Overlay Success Criteria

The overlay is useful only if it passes:

- positive active return versus volatility-targeted SPY
- positive information ratio
- improved max drawdown
- improved active Calmar
- reasonable turnover
- passes same-exposure random null
- passes same-turnover random null
- passes same-regime-exposure random null

The core hypothesis:

```yaml
null_hypothesis:
  The HMM regime overlay is no better than random de-risking with the same exposure and turnover.

alternative_hypothesis:
  The HMM identifies risk states where reducing exposure improves active performance versus a volatility-targeted baseline.
```

## 10. Phase 5: Target Reset

Only after the measurement/null framework is fixed should labels be changed.

The current target is likely too close to:

```text
next_day_return > 0
```

That target allows the model to collapse into:

```text
SPY usually goes up, so stay long.
```

Replace it with targets where always-long is not a cheat code.

Candidate targets:

### 10.1 Future Drawdown Risk

```text
target = 1 if max_drawdown_over_next_N_days < -threshold
```

Example settings:

```yaml
N:
  - 10 days
  - 20 days

threshold:
  - -2%
  - -3%
  - -5%
```

This aligns with what the regime system may already be good at.

### 10.2 Forward Risk-Adjusted Return

```text
target = next_5d_return / recent_realized_vol

target = next_20d_return / recent_realized_vol
```

This asks:

```text
Is the return worth the risk?
```

not:

```text
Will SPY be up?
```

### 10.3 Cross-Asset Relative Ranking

Universe:

- SPY
- QQQ
- IWM
- TLT
- GLD
- BTC-USD

Targets:

- which asset has the best forward risk-adjusted return?
- rank assets by next_5d return
- rank assets by next_20d return
- rank assets by next_5d or next_20d volatility-adjusted return

Cross-asset ranking may be more tractable than single-asset timing.

### 10.4 Future Volatility Forecasting

```text
target = future_realized_volatility
```

This is not direct alpha, but it can improve:

- position sizing
- risk management
- portfolio allocation
- drawdown control

## 11. Phase 6: Model Architecture Reset

Do not restart with the full model zoo.

Start with simple models:

- logistic regression
- regularized linear model
- HistGradientBoosting
- LightGBM
- XGBoost
- simple regime rules

Only reintroduce complex models after a simple model shows positive active skill.

Complex models include:

- LSTM
- PatchTST
- iTransformer
- Mamba
- Temporal Fusion Transformer

New rule:

No complex model until a simple model beats the matched nulls.

Otherwise architecture work becomes expensive noise mining.

## 12. Implementation Deliverables

The next implementation sprint should produce the following modules or equivalent functionality. Adapt names to the existing repository structure if needed.

### 12.1 `metrics_active.py`

Required functionality:

- `active_return_series(strategy_returns, benchmark_returns)`
- `annualized_active_return(active_returns)`
- `tracking_error(active_returns)`
- `information_ratio(active_returns)`
- `active_max_drawdown(active_returns)`
- `active_calmar(active_returns)`
- `benchmark_correlation(strategy_returns, benchmark_returns)`
- `beta_to_benchmark(strategy_returns, benchmark_returns)`
- `alpha_after_beta(strategy_returns, benchmark_returns)`

### 12.2 `metrics_turnover.py`

Required functionality:

- `daily_turnover(positions)`
- `annualized_turnover(positions)`
- `position_flip_count(positions)`
- `average_holding_period(positions)`
- `round_trip_count(positions)`
- `cost_drag_from_turnover(positions, transaction_cost_bps)`
- `cost_per_unit_active_return(cost_drag, active_return)`

### 12.3 `null_baselines.py`

Required functionality:

- `same_average_exposure_random(...)`
- `same_turnover_random(...)`
- `same_exposure_and_turnover_random(...)`
- `same_regime_exposure_random(...)`
- `block_bootstrap_same_exposure_random(...)`

Nulls should support Monte Carlo runs and return p-values against canonical active metrics.

### 12.4 `clean_comparison_report.py`

Required functionality:

- no stale interaction feature-importance blocks
- no mixed run modes
- clean separation of interaction and non-interaction runs
- active-metric leaderboard
- exposure and turnover diagnostics
- matched-null summary
- clear status labels:
  - `beta_harvester`
  - `failed_active_skill`
  - `passes_matched_null`
  - `candidate_for_further_research`

### 12.5 `regime_overlay_experiment.py`

Required functionality:

- volatility-targeted SPY baseline
- fold-local HMM regime characterization
- training-only dangerous-regime mapping
- filtered regime probabilities if available
- hard veto overlay
- optional soft overlay
- matched-null comparison
- active metric report

## 13. Stop/Go Gates

### Gate 1: Existing Models

Proceed only if at least one current model:

- beats volatility-targeted SPY on active IR
- beats same-exposure random null
- beats same-turnover random null
- does not rely on near-always-long exposure

Expected result:

Probably fail.

If all models fail, stop working on current daily-direction model variants.

### Gate 2: Regime Overlay

Proceed only if the HMM regime overlay:

- improves active IR versus volatility-targeted SPY
- reduces active drawdown
- improves active Calmar
- passes same-regime-exposure random null
- does not create excessive turnover

If it passes, the regime layer has value as a risk classifier.

If it fails, move on from the current HMM design.

### Gate 3: New Target

Proceed to complex models only if a simple model on the new target:

- beats matched nulls
- has stable active performance across folds
- does not rely on near-always-long exposure
- maintains reasonable turnover after costs

## 14. What To Inspect Before Implementing Patches

Before implementation changes, inspect the codebase for:

- where `information_ratio` is calculated
- where `excess_net_sharpe` is calculated
- where `trade_frequency` is calculated
- how positions are represented
- how costs are charged
- how benchmark returns are aligned
- how HMM probabilities are generated in test periods
- whether test-period regime probabilities are filtered or smoothed
- how regime IDs are mapped across folds
- how reports combine interaction and non-interaction runs
- how Optuna objectives are defined
- how walk-forward fold artifacts are saved

Do not assume current metric names are accurate. Verify from code.

## 15. Definition of Done for This Documentation Task

This task is complete when:

1. `docs/FIRST_PRINCIPLES_RESET_PLAN.md` exists.
2. The file includes all major sections above.
3. The document clearly states that the original ensemble champion is no longer considered validated alpha.
4. The document clearly states that no further model tuning should happen before measurement repair and matched nulls.
5. The document clearly states that HMM regime IDs must not be hardcoded across folds.
6. The document includes the required stop/go gates.
7. The document includes the implementation deliverables for the next sprint.
8. No model code is modified.
9. No training runs are started.
