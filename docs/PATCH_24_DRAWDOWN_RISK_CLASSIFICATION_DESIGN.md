# Future Drawdown-Risk Classification Design

## 1. Executive Summary

The next research track should test whether the framework can predict future drawdown-risk events directly, instead of trying to predict daily SPY direction or time SPY exposure with overlay rules.

The core question is:

```text
Can the framework predict elevated future drawdown risk better than simple baselines, and can that prediction improve exposure decisions versus a volatility-targeted baseline?
```

The proposed target is a binary future risk event such as:

```text
future max drawdown over the next 10 or 20 days is less than -3%
```

In plain terms:

- today’s features are used to estimate whether the next 10 to 20 trading days are likely to experience a meaningful drawdown event
- the model is evaluated first as a classifier
- only after classification quality is established should it be converted into an exposure rule
- any exposure rule must still beat matched random de-risking and a volatility-targeted baseline

This is a target reset, not another overlay rescue.

It should not reuse:

- the failed regime-stacking ensemble
- the failed HMM hard-veto overlay
- the failed simple volatility-quantile overlay
- raw daily-direction labels as the primary target

## 2. Motivation

The project has already falsified several prior formulations:

1. the original `RegimeStackingEnsemble` was not validated alpha
2. the HMM hard-veto overlay failed Gate 2
3. the simple volatility-quantile overlay failed Gate 2

Those failures point to a broader conclusion:

```text
The current framework is not demonstrating robust daily SPY timing skill in its present formulation.
```

That does not mean the framework is useless. It means the question was wrong.

Drawdown-risk classification is the right next step because:

- it aligns more naturally with risk-management decisions
- it avoids the “SPY usually goes up” cheat code built into daily direction targets
- it can still be judged with active-return and matched-null discipline
- it is a simpler and more direct test of whether the features contain useful risk information

If drawdown-risk prediction fails, that is strong evidence that this current single-asset formulation should be deprioritized further.

If it succeeds, it creates a cleaner bridge into exposure control without forcing the model to predict exact daily return sign.

## 3. Files and Functions Inspected

### `features/feature_engineering.py`

Relevant paths:

- `FeatureSet`
- `build_feature_set(...)`
- `_add_base_features(...)`
- `_add_advanced_features(...)`
- `build_vix_features(...)`

Current behavior:

- features are built from lagged or contemporaneous information only
- the current target path creates:
  - `forward_simple_return_1d`
  - `target_return_risk_adjusted`
  - `target_direction`
- the current module is target-coupled to next-period direction logic

Design implication:

- drawdown-risk should be added as a first-class target family, not as an ad hoc post-processing hack
- the feature blocks are still reusable

### `evaluation/walk_forward.py`

Relevant path:

- `generate_walk_forward_splits(...)`

Current behavior:

- strictly ordered train / validation / test slices
- no leakage from overlap within a split

Design implication:

- drawdown target generation must happen before splitting, but only with forward windows fully contained inside each row’s timestamp
- model selection must remain validation-only
- final evaluation must remain test-only

### `utils/experiment.py`

Relevant paths:

- `prepare_experiment(...)`
- `prepare_experiment_from_market_data(...)`
- `evaluate_model(...)`

Current behavior:

- `prepare_experiment_from_market_data(...)` calls `build_feature_set(...)`
- `evaluate_model(...)` fits using `target_column="target_direction"`
- reporting and artifacts assume existing return/benchmark columns are present in the feature frame

Design implication:

- drawdown-risk should be integrated as a new target column and eventually a new experiment branch
- do not overload `target_direction`
- keep audit-artifact behavior standardized

### `evaluation/metrics.py`

Relevant paths:

- active-return diagnostics
- turnover diagnostics
- benchmark-relative metrics

Design implication:

- classifier quality alone is not enough
- any downstream exposure rule must still be evaluated on:
  - `information_ratio`
  - `annualized_active_return`
  - `active_calmar`
  - turnover / cost diagnostics

### `evaluation/null_baselines.py`

Relevant paths:

- `run_matched_null_suite(...)`
- `same_average_exposure_random(...)`
- `same_turnover_random(...)`
- `same_exposure_and_turnover_random(...)`
- `same_regime_exposure_random(...)`
- `block_bootstrap_same_exposure_random(...)`

Design implication:

- the matched-null framework already exists and should be reused
- a future drawdown-based overlay should be judged against matched nulls, not just raw active return

### `evaluation/audit_artifacts.py`

Relevant path:

- `build_standard_audit_artifact_frame(...)`

Design implication:

- future drawdown experiments should save standardized artifacts from day one
- that means saving:
  - `target`
  - `prediction_probability`
  - `executed_position`
  - realized return streams

## 4. New Research Question

The old failed question was:

```text
Can the system predict daily SPY direction or time SPY exposure well enough to add alpha?
```

The new question should be:

```text
Can the system identify elevated future drawdown-risk events better than simple baselines, and can that information improve exposure decisions versus a volatility-targeted SPY baseline?
```

This is narrower, more falsifiable, and better aligned with the evidence from the failed overlay tracks.

## 5. Target Definition

### 5.1 Core Binary Event

Primary candidate target:

```text
target_drawdown_event = 1
if future_max_drawdown over the next N days < threshold
else 0
```

Example settings:

- horizon `N = 10`
- horizon `N = 20`
- threshold `-2%`
- threshold `-3%`
- threshold `-5%`

Recommended first-pass grid:

- `N in {10, 20}`
- `threshold in {-0.03, -0.05}`

### 5.2 Exact Label Construction

For each date `t`:

1. start from price at `t`
2. inspect the forward path from `t+1` through `t+N`
3. compute the worst cumulative drawdown from that forward path relative to the starting point
4. label `1` if that worst drawdown breaches the threshold

Equivalent formulation:

```text
future_path_return_k = price_{t+k} / price_t - 1
future_max_drawdown_t = min_k(future_path_return_k) for k in 1..N
target = 1 if future_max_drawdown_t <= threshold else 0
```

Design note:

- this target is explicitly forward-looking by construction
- the label is valid as long as features use information through `t` only

### 5.3 Why This Target Is Better

This target is better than daily direction because:

- always-long is not a cheap solution
- it aligns with drawdown control, which is closer to the observed value proposition of the framework
- it matches the failure evidence from the overlay work

## 6. Label Construction Requirements

### 6.1 Leakage Rules

The drawdown label must be created with future observations only for the label itself, never for the features.

Rules:

- features through close `t`
- drawdown label over `t+1 ... t+N`
- no future returns may enter the feature columns
- no row may remain in the dataset if its forward window is incomplete

### 6.2 Train / Validation / Test Handling

The cleanest approach is:

- construct the label once on the full chronologically ordered frame
- drop rows near the end that cannot support a full forward horizon
- then split with `generate_walk_forward_splits(...)`

That is acceptable because the label uses future realized prices only to define the supervised target, not to leak into training features for other rows.

### 6.3 Class Balance Check

Drawdown events may be relatively rare.

Before model work, every candidate target should report:

- event rate overall
- event rate by fold
- event rate by train / validation / test slice

Reject targets that are too sparse to support stable fold-level evaluation.

## 7. Feature Strategy

The initial drawdown-risk experiment should reuse the current stationary feature backbone rather than inventing a new feature universe immediately.

Reusable current features include:

- `return_1d`
- `return_5d`
- `return_20d`
- `vol_ratio`
- `momentum_norm`
- `volume_zscore`
- `range_norm`
- `sma_ratio`
- `realized_vol_20`
- `close_to_open_gap`
- `price_acceleration`
- `downside_vol_ratio`

Advanced features that may be especially relevant for drawdown risk:

- `relative_strength_vs_benchmark`
- `overnight_gap_zscore`
- `volume_trend_strength`
- `autocorrelation_zscore`
- `volatility_regime`
- `asset_return_vs_spy`
- `relative_vol_ratio`
- VIX-derived features when available

Additional drawdown-risk-specific features can be considered later, but should not be required to start.

## 8. Baselines

The new target should be judged against simple baselines before any complex model is introduced.

### 8.1 Classification Baselines

Required baselines:

- always-negative baseline
- historical event-rate baseline
- rolling event-rate baseline
- logistic regression
- regularized linear model
- HistGradientBoosting

Optional later:

- LightGBM
- XGBoost

Do not start with deep sequence models.

### 8.2 Trading / Exposure Baselines

If a classifier is converted into an exposure rule, compare against:

- cash
- buy-and-hold SPY
- volatility-targeted SPY
- simple deterministic cut rule if useful

The benchmark for active performance should be:

- volatility-targeted SPY, not raw buy-and-hold

## 9. Evaluation Layers

This research track needs two distinct evaluation layers.

### 9.1 Classification Layer

Measure whether the target is even predictable.

Recommended metrics:

- ROC AUC
- precision / recall
- balanced accuracy
- Brier score
- calibration curve summary
- event-rate stability by fold

Important:

- do not treat directional accuracy as the main metric here
- drawdown-risk is an event-detection problem, not a next-day sign problem

### 9.2 Exposure / Economic Layer

Only after a classifier shows fold-stable skill should it be converted into an exposure rule.

Example rule:

```text
if P(drawdown_event) > threshold:
    final_position = base_position * risk_multiplier
else:
    final_position = base_position
```

Then evaluate using:

- `information_ratio`
- `annualized_active_return`
- `active_calmar`
- `active_max_drawdown`
- `daily_turnover`
- `annualized_turnover`
- `cost_drag`

## 10. Matched Null Framework for Drawdown Risk

The matched-null framework must remain mandatory.

For any drawdown-based exposure rule, test:

- `same_average_exposure_random`
- `same_turnover_random`
- `same_exposure_and_turnover_random`
- `block_bootstrap_same_exposure_random`

Add a drawdown-state-specific null once the rule exists:

```text
same_predicted_risk_bucket_exposure_random
```

Meaning:

- preserve the model’s exposure rate within risk-score buckets or event-probability buckets
- randomize which days receive the cut inside those buckets

If the strategy cannot beat this null, it is not adding timing information beyond generic de-risking on flagged days.

## 11. Candidate Experiment Structure

### Phase A: Label Feasibility

Purpose:

- determine whether candidate drawdown labels are statistically workable

Tasks:

1. compute label prevalence
2. inspect fold stability
3. inspect event clustering
4. reject pathological label definitions

### Phase B: Pure Classifier Test

Purpose:

- test whether simple models can predict the label better than trivial baselines

Rules:

- no exposure conversion yet
- no overlay claims yet
- rank by classification and calibration quality first

### Phase C: Exposure Rule Test

Purpose:

- translate the best simple classifier into a risk-reduction rule against vol-targeted SPY

Rules:

- threshold selected on validation only
- test-only final scoring
- matched nulls required

## 12. Integration Points

### 12.1 `features/feature_engineering.py`

Recommended future change:

- add a target-family parameter, rather than always creating only `target_direction`

Suggested concept:

```text
target_type = direction | drawdown_risk | forward_risk_adjusted_return
```

For this track, add fields like:

- `target_drawdown_event_10d_3pct`
- `target_drawdown_event_20d_3pct`
- optional continuous helper:
  - `future_max_drawdown_10d`
  - `future_max_drawdown_20d`

### 12.2 `utils/experiment.py`

Recommended future change:

- let `evaluate_model(...)` receive `target_column` from configuration or explicit experiment setup

Current issue:

- it is still hardcoded to `target_direction`

That should be generalized before implementation.

### 12.3 `evaluation/audit_artifacts.py`

Recommended future use:

- save the chosen drawdown target in `target`
- save classifier probability in `prediction_probability`
- save any downstream executed exposure rule in `executed_position`

### 12.4 `evaluation/null_baselines.py`

Recommended future extension:

- add a risk-bucket-matched null once the drawdown classifier is converted to a policy

## 13. Stop / Go Gates

### Gate 1: Label Feasibility

Proceed only if:

- event rate is not too sparse
- event prevalence is reasonably stable across folds
- train / validation / test slices all contain enough events

If this fails, change the horizon or threshold before any model work.

### Gate 2: Pure Classification Skill

Proceed to exposure conversion only if a simple model:

- beats trivial classification baselines
- shows stable fold-level AUC or equivalent event-detection skill
- is reasonably calibrated

If this fails, stop. Do not jump to complex models.

### Gate 3: Economic / Exposure Skill

Proceed only if the classifier-derived policy:

- beats volatility-targeted SPY on active IR
- improves or preserves active Calmar
- passes matched nulls
- does not create excessive turnover

If this fails, the classifier may still be academically predictive, but it is not yet economically useful.

### Gate 4: Complexity Escalation

Only consider more complex models if a simple drawdown-risk model:

- passes classification gates
- passes exposure economics gates
- passes matched-null validation

Otherwise complexity is just noise mining again.

## 14. Recommended First Implementation Order

1. Add drawdown target construction to the feature pipeline.
2. Add label prevalence and fold-balance diagnostics.
3. Generalize experiment plumbing so `target_column` is not hardcoded to `target_direction`.
4. Run simple-model classification baselines only.
5. Choose the strongest simple classifier.
6. Convert it into a volatility-targeted overlay rule.
7. Apply matched-null validation.

## 15. What Not To Do

Do not:

- revive the failed regime-stacking ensemble
- re-run HMM overlay variants as the main path
- re-run volatility-quantile overlay rescue variants as the main path
- jump to deep sequence models before simple models work
- treat classifier accuracy alone as economic evidence

## 16. Final Recommendation

The next serious research track should be future drawdown-risk classification.

This is the cleanest target reset because it:

- responds directly to the failure evidence
- better matches the likely value proposition of the feature set
- avoids the structural long-bias trap of daily direction targets
- still allows strict active-return and matched-null validation

If this track also fails with simple models and matched-null discipline, the repository should strongly consider shifting away from single-asset SPY timing entirely and toward cross-asset or allocation formulations.
