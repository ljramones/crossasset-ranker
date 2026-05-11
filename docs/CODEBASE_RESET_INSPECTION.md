# CODEBASE RESET INSPECTION

Inspection date: 2026-05-08  
Repository: `ml-trading-signal-exploration`  
Source of truth: [FIRST_PRINCIPLES_RESET_PLAN.md](/Users/larrym/prediction/docs/FIRST_PRINCIPLES_RESET_PLAN.md)

This is a read-only inspection report. No model code, evaluation code, configs, result files, or scripts were modified during this task.

## 1. Repository Structure

### Relevant top-level directories

- [audit](/Users/larrym/prediction/audit)
  - Integrity audit logic and adversarial/null-testing paths.
- [champions](/Users/larrym/prediction/champions)
  - Frozen champion manifests:
    - [champion_v1.0_manifest.yaml](/Users/larrym/prediction/champions/champion_v1.0_manifest.yaml)
    - [current_champion_manifest.yaml](/Users/larrym/prediction/champions/current_champion_manifest.yaml)
- [config](/Users/larrym/prediction/config)
  - Runtime configuration:
    - [config.yaml](/Users/larrym/prediction/config/config.yaml)
- [data](/Users/larrym/prediction/data)
  - Market-data loading and local caches:
    - [market_data.py](/Users/larrym/prediction/data/market_data.py)
    - `multi_asset_cache/`
- [docs](/Users/larrym/prediction/docs)
  - Planning/reset documentation:
    - [FIRST_PRINCIPLES_RESET_PLAN.md](/Users/larrym/prediction/docs/FIRST_PRINCIPLES_RESET_PLAN.md)
- [evaluation](/Users/larrym/prediction/evaluation)
  - Core metrics and walk-forward splitting:
    - [metrics.py](/Users/larrym/prediction/evaluation/metrics.py)
    - [walk_forward.py](/Users/larrym/prediction/evaluation/walk_forward.py)
- [features](/Users/larrym/prediction/features)
  - Feature engineering and regime-derived features:
    - [feature_engineering.py](/Users/larrym/prediction/features/feature_engineering.py)
    - [regime_features.py](/Users/larrym/prediction/features/regime_features.py)
    - [engineering.py](/Users/larrym/prediction/features/engineering.py) wrapper
- [models](/Users/larrym/prediction/models)
  - Base models and ensembles:
    - [ensemble.py](/Users/larrym/prediction/models/ensemble.py)
    - sequence / tree model modules
- [optimization](/Users/larrym/prediction/optimization)
  - Optuna tuning logic:
    - [optuna_tuner.py](/Users/larrym/prediction/optimization/optuna_tuner.py)
- [portfolio](/Users/larrym/prediction/portfolio)
  - Selective portfolio utilities:
    - [asset_selector.py](/Users/larrym/prediction/portfolio/asset_selector.py)
    - [portfolio_builder.py](/Users/larrym/prediction/portfolio/portfolio_builder.py)
- [regime](/Users/larrym/prediction/regime)
  - HMM/GMM regime detection:
    - [regime_detection.py](/Users/larrym/prediction/regime/regime_detection.py)
- [results](/Users/larrym/prediction/results)
  - Saved comparisons, ensemble artifacts, audit reports, optimization outputs, multi-asset outputs.
- [tests](/Users/larrym/prediction/tests)
  - Metric, feature, walk-forward, and regime tests.
- [utils](/Users/larrym/prediction/utils)
  - Experiment assembly, reporting, config, seeding:
    - [experiment.py](/Users/larrym/prediction/utils/experiment.py)
    - [reporting.py](/Users/larrym/prediction/utils/reporting.py)

### Likely experiment / script entry points

- [main.py](/Users/larrym/prediction/main.py)
  - Single CLI entry point for baseline, regime, multi-asset, portfolio, tuning, and audit workflows.
- [notebooks](/Users/larrym/prediction/notebooks)
  - Exploratory notebooks only; not primary implementation paths.

### Where tests live

- [test_metrics.py](/Users/larrym/prediction/tests/test_metrics.py)
- [test_features.py](/Users/larrym/prediction/tests/test_features.py)
- [test_regime_features.py](/Users/larrym/prediction/tests/test_regime_features.py)
- [test_walk_forward.py](/Users/larrym/prediction/tests/test_walk_forward.py)

## 2. Metric Calculation Paths

### Core trading metrics

```text
sharpe:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 28-80, especially 56 and 62
  current_formula_or_behavior: gross Sharpe = annualized Sharpe of gross_strategy_returns
  notes: uses _annualized_sharpe(); this is pre-cost.

net_sharpe:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 28-80, especially 58 and 64
  current_formula_or_behavior: annualized Sharpe of net_strategy_returns
  notes: this is the main ranking metric in many report paths.

sortino:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 28-80, especially 57 and 63
  current_formula_or_behavior: annualized Sortino of gross_strategy_returns
  notes: pre-cost.

net_sortino:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 28-80, especially 59 and 65
  current_formula_or_behavior: annualized Sortino of net_strategy_returns
  notes: averaged into comparison rows.

calmar:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 70, helpers 157-170
  current_formula_or_behavior: annualized return from net returns divided by absolute max drawdown
  notes: uses net strategy returns, not active-return stream.

information_ratio:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 47, 71, helper 142-146
  current_formula_or_behavior: _annualized_sharpe(active_returns), where active_returns = net_strategy_returns - benchmark_returns
  notes: benchmark-relative metric exists already, but it is only one field among many raw-return metrics.

excess_net_sharpe:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 55-67
  current_formula_or_behavior: net_sharpe - benchmark_sharpe
  notes: this is explicitly the rough diagnostic called out in the reset plan as insufficient.

benchmark_sharpe:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 55-67
  current_formula_or_behavior: annualized Sharpe of benchmark_returns
  notes: benchmark_returns are passed in from feature-engineered frame column benchmark_return_1d.

max_drawdown:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics / _max_drawdown
  approximate_lines: 72, helper 173-176
  current_formula_or_behavior: min drawdown of cumulative net equity curve
  notes: raw strategy drawdown, not active drawdown.
```

### Classification metrics

```text
directional_accuracy:
  file: evaluation/metrics.py
  function/class: compute_classification_metrics
  approximate_lines: 10-25, especially 23
  current_formula_or_behavior: sklearn accuracy_score(y_true, y_pred)
  notes: binary direction classification metric.

auc_roc:
  file: evaluation/metrics.py
  function/class: compute_classification_metrics
  approximate_lines: 17-24
  current_formula_or_behavior: sklearn roc_auc_score(y_true, y_score), fallback 0.5 on ValueError
  notes: fallback can mask degenerate prediction streams.
```

### Exposure / activity metrics

```text
trade_frequency:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 50-51, 73
  current_formula_or_behavior: executed_position.ne(0.0).mean()
  notes: this is fraction of bars with nonzero executed position, not actual trade count. The reset-plan concern is valid.

average_long_exposure:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 50-52, 74
  current_formula_or_behavior: executed_position.mean()
  notes: with long/flat signals this is average long exposure. With shorting it would become signed average exposure.

average_position_size:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 50-53, 75
  current_formula_or_behavior: executed_position.abs().mean()
  notes: this is average absolute exposure magnitude.

fraction_positive_predictions:
  file: evaluation/metrics.py
  function/class: compute_trading_metrics
  approximate_lines: 54, 76
  current_formula_or_behavior: signal.mean()
  notes: based on unshifted prediction stream, not executed position.
```

### Portfolio-level metrics

```text
portfolio_metrics:
  file: evaluation/metrics.py
  function/class: compute_return_stream_metrics
  approximate_lines: 83-114
  current_formula_or_behavior: computes sharpe, net_sharpe, sortino, net_sortino, calmar, information_ratio, max_drawdown from precomputed return streams
  notes: this path does not currently compute benchmark_sharpe, excess_net_sharpe, or detailed exposure diagnostics.
```

## 3. Turnover, Cost, and Position Representation Paths

### Position and return mechanics

```text
positions_representation:
  file: evaluation/metrics.py
  function/class: compute_strategy_returns
  approximate_lines: 117-127
  current_formula_or_behavior: executed position = signal.shift(1).fillna(0.0)
  notes: positions are executed with a one-bar lag relative to predictions.

turnover_series:
  file: evaluation/metrics.py
  function/class: compute_signal_turnover
  approximate_lines: 130-133
  current_formula_or_behavior: position.diff().abs().fillna(position.abs())
  notes: this correctly treats 0->1 as 1, 1->0 as 1, 1->-1 as 2, -1->1 as 2 if short positions are present.

transaction_cost_application:
  file: evaluation/metrics.py
  function/class: compute_strategy_returns
  approximate_lines: 124-127
  current_formula_or_behavior: transaction_cost = turnover * (transaction_cost_bps / 10000.0)
  notes: charged directly against return stream on each turnover event.
```

### Current test coverage

```text
turnover_test:
  file: tests/test_metrics.py
  function/class: test_compute_signal_turnover_charges_only_on_position_flips
  approximate_lines: 45-50
  current_formula_or_behavior: checks [0,1,1,0,0,1] -> [0,1,0,1,0,1]
  notes: basic flip-only coverage exists, but there are no synthetic tests yet for alternating long/short, always-flat, or buy-and-hold reporting.
```

## 4. Benchmark Alignment Paths

```text
benchmark_loading_single_asset:
  file: data/market_data.py
  function/class: load_market_data
  approximate_lines: 35-52
  current_formula_or_behavior: joins asset OHLCV with benchmark Close renamed to BenchmarkClose, then computes benchmark_return_1d = BenchmarkClose.pct_change()
  notes: benchmark is aligned by inner join on date, then dataset.dropna().sort_index().

benchmark_loading_multi_asset:
  file: data/market_data.py
  function/class: load_market_data_bundle
  approximate_lines: 77-92
  current_formula_or_behavior: same benchmark join per asset, same benchmark_return_1d pct_change
  notes: VIX is left-joined then forward-filled.

feature_set_benchmark_usage:
  file: features/feature_engineering.py
  function/class: build_feature_set / _add_advanced_features
  approximate_lines: 40-42, 147-149, 167-168, 190-194
  current_formula_or_behavior: benchmark_close and benchmark_simple_return_1d feed relative-strength and relative-volatility features
  notes: benchmark series influences both features and evaluation metric paths.
```

## 5. Feature Engineering and Label Paths

```text
feature_entrypoint:
  file: features/feature_engineering.py
  function/class: build_feature_set
  approximate_lines: 20-86
  current_formula_or_behavior: builds base features, optional advanced features, optional VIX features, then creates target columns
  notes: main feature-set constructor used by experiment assembly.

target_definition:
  file: features/feature_engineering.py
  function/class: build_feature_set
  approximate_lines: 73-77
  current_formula_or_behavior: target_return_risk_adjusted = next_return / forward_vol; target_direction = target_return_risk_adjusted > 0
  notes: current target is not raw next-day return > 0; it is next return scaled by future vol and binarized.

rolling_zscore_path:
  file: features/feature_engineering.py
  function/class: _rolling_zscore
  approximate_lines: 285-290
  current_formula_or_behavior: (series - rolling_mean) / rolling_std over trailing window
  notes: static inspection suggests trailing-only, no explicit shift.

regime_feature_expansion:
  file: features/regime_features.py
  function/class: add_regime_features
  approximate_lines: 8-35
  current_formula_or_behavior: adds regime_id, regime_prob_0/1/2, plus selected interactions with momentum_norm and vol_ratio
  notes: this is the regime-feature path used in regime-aware experiments.
```

## 6. Walk-Forward, Experiment Assembly, and OOF Paths

```text
walk_forward_splitter:
  file: evaluation/walk_forward.py
  function/class: generate_walk_forward_splits
  approximate_lines: 20-52
  current_formula_or_behavior: strictly ordered train -> validation -> test windows using iloc slices
  notes: no overlap leakage across test windows by construction; step_size controls rolling advance.

experiment_assembly:
  file: utils/experiment.py
  function/class: prepare_experiment / prepare_experiment_from_market_data
  approximate_lines: 113-173
  current_formula_or_behavior: loads config, seeds runtime, loads market data, builds feature set, prints ADF summary, generates walk-forward splits
  notes: this is the main setup path for most CLI workflows.

single_model_oof_capture:
  file: utils/experiment.py
  function/class: evaluate_model
  approximate_lines: 202-327
  current_formula_or_behavior: per split, fit on train+val context, predict on test, evaluate on test, store OOF probability/prediction rows, then aggregate mean metrics
  notes: OOF frame includes target_direction, returns, benchmark_return_1d, and optional regime metadata columns.

ensemble_oof_frame_builder:
  file: main.py
  function/class: _build_ensemble_oof_frame
  approximate_lines: 1062-1118
  current_formula_or_behavior: reruns each base model across all splits, renames probability -> probability__{label}, then inner-merges OOF outputs by date/split/target/return/benchmark and regime metadata
  notes: this is the critical path for stacking/regime-weighted ensemble training inputs.
```

## 7. HMM / Regime Detection Paths

```text
regime_detector_config:
  file: regime/regime_detection.py
  function/class: RegimeDetectionConfig
  approximate_lines: 16-32
  current_formula_or_behavior: defaults to model_type=hmm, n_regimes=3, and a fixed inference feature set
  notes: inference feature list is hardcoded in config dataclass, not loaded from YAML directly.

train_only_regime_fit:
  file: regime/regime_detection.py
  function/class: MarketRegimeDetector.fit
  approximate_lines: 63-84
  current_formula_or_behavior: feature subset selection -> dropna -> StandardScaler.fit_transform(training_features) -> HMM/GMM fit on training-window data only
  notes: structural train-only behavior is clear here.

test_period_regime_probabilities:
  file: regime/regime_detection.py
  function/class: MarketRegimeDetector.predict / _predict_raw
  approximate_lines: 86-110 and 168-183
  current_formula_or_behavior: scaler.transform(valid rows) -> model.predict / model.predict_proba on the full frame slice
  notes: this produces probabilities on the entire slice at once. For GMM this is not a smoothing issue. For HMM, hmmlearn predict_proba on a full sequence likely uses sequence-level posterior probabilities, not explicitly one-step filtered probabilities. This is a key inspection target for the reset plan.

canonical_regime_mapping:
  file: regime/regime_detection.py
  function/class: _build_canonical_mapping
  approximate_lines: 185-208
  current_formula_or_behavior: maps raw labels to canonical ids using train-period mean return_1d and realized_vol_20
  notes: bull=0 via max return, bear=1 via min return, remaining label -> high_vol=2.

best_regime_selection:
  file: regime/regime_detection.py
  function/class: identify_best_regime
  approximate_lines: 112-137
  current_formula_or_behavior: computes regime-conditional net_sharpe for a long-only signal within each regime and picks argmax
  notes: this is exactly the hardcoded "best regime by raw net_sharpe" logic the reset plan wants to move away from.

regime_overlay_feature_insertion:
  file: main.py
  function/class: _build_regime_aware_experiment
  approximate_lines: 969-1059
  current_formula_or_behavior: per split, fit detector on train, predict train/val/test labels and probabilities, optionally identify best regime, add aggressive-trade-filter columns, then add regime features
  notes: fold-local train-only fit is correct; dangerous-regime characterization overlay does not exist yet.
```

## 8. Reporting and Ranking Paths

```text
summary_table_ordering:
  file: utils/reporting.py
  function/class: summarize_results
  approximate_lines: 8-38
  current_formula_or_behavior: orders columns and sorts descending by net_sharpe if present, else sharpe
  notes: this directly conflicts with the reset-plan requirement to stop ranking primarily by raw/net Sharpe.

cli_comparison_printing:
  file: main.py
  function/class: _print_comparison
  approximate_lines: 1529-1539
  current_formula_or_behavior: prints "Ranked walk-forward comparison by net_sharpe" when available
  notes: another raw-Sharpe-first path to patch later.

regime_summary_builder:
  file: main.py
  function/class: _build_regime_summary
  approximate_lines: 1656-1802
  current_formula_or_behavior: constructs mixed summary sections for baseline, regime weighted, regime stacking, interaction delta, feature importance, per-regime metrics
  notes:
    - currently mixes interaction and non-interaction artifact sources
    - feature importance path falls back between interaction and non-interaction CSVs
    - ranking deltas are all on net_sharpe/net_sortino/calmar

regime_summary_printer:
  file: main.py
  function/class: _print_regime_weighted_summary
  approximate_lines: 1578-1615
  current_formula_or_behavior: prints a compact regime summary block focused on stacking / regime-weighted / regime-stacking
  notes: still centered on net_sharpe/net_sortino/calmar, not active metrics.

ensemble_bias_fix_summary:
  file: main.py
  function/class: _save_ensemble_bias_fix_comparison
  approximate_lines: 297-329
  current_formula_or_behavior: saves focused CSV with model, net_sharpe, information_ratio, excess_net_sharpe, net_sortino, calmar, average_long_exposure, fraction_positive_predictions, trade_frequency
  notes: useful current comparison path for long-bias diagnosis, but still includes `trade_frequency` instead of explicit turnover/trade counts.
```

## 9. Ensemble and Meta-Learner Objective Paths

```text
plain_weighted_average_objective:
  file: models/ensemble.py
  function/class: WeightedAverageEnsemble._objective
  approximate_lines: 71-91
  current_formula_or_behavior: optimize weights to maximize net_sharpe
  notes: raw-Sharpe objective.

regime_weighted_objective:
  file: models/ensemble.py
  function/class: RegimeWeightedEnsemble._objective
  approximate_lines: 211-227
  current_formula_or_behavior: optimize regime-specific weights to maximize net_sharpe
  notes: also raw-Sharpe objective.

plain_stacking_meta_learner:
  file: models/ensemble.py
  function/class: StackingEnsemble
  approximate_lines: 230-266
  current_formula_or_behavior: LogisticRegression meta-learner over base-model probabilities only; fixed classification threshold
  notes: no active-return or exposure penalty logic.

regime_stacking_meta_learner:
  file: models/ensemble.py
  function/class: RegimeStackingEnsemble
  approximate_lines: 269-426
  current_formula_or_behavior:
    - LogisticRegression over base probabilities + regime probabilities
    - optional explicit interaction features
    - threshold selected on train OOF by either information_ratio or net_sharpe
    - long-bias penalty uses average_long_exposure and fraction_positive_predictions above max_long_exposure
  notes:
    - objective selection happens in _select_threshold(), not by changing the logistic loss itself
    - true meta-learner training still optimizes classification log-loss internally
    - this is the main path to revisit if ensemble objective is rebuilt around active skill
```

## 10. Optuna Tuning Paths

```text
optuna_workflow:
  file: optimization/optuna_tuner.py
  function/class: run_optuna_tuning
  approximate_lines: 13-56
  current_formula_or_behavior: creates Optuna study per model with direction="maximize" and study_name=f"{model_name}_sharpe"
  notes: naming and objective are still Sharpe-centric.

optuna_objective:
  file: optimization/optuna_tuner.py
  function/class: _objective
  approximate_lines: 87-108
  current_formula_or_behavior: evaluate_model(...) and return artifacts.comparison_row["net_sharpe"]
  notes: this is the exact tuning objective path to patch in a future implementation sprint.

search_spaces:
  file: optimization/optuna_tuner.py
  function/class: _suggest_params
  approximate_lines: 121-163
  current_formula_or_behavior: search spaces for lstm / itransformer / patchtst only
  notes: no simple-model active-skill-first gating exists.
```

## 11. Audit and Null-Test Paths

```text
comparative_null_test_metrics:
  file: audit/integrity_audit.py
  function/class: _evaluate_single_model_under_shuffle / _evaluate_regime_stacking_meta / _build_strategy_diagnostics
  approximate_lines: 700-859
  current_formula_or_behavior:
    - computes net_sharpe, net_sortino, calmar, information_ratio, excess_net_sharpe
    - computes exposure diagnostics
    - uses benchmark-relative decision metric = information_ratio
  notes:
    - _evaluate_single_model_under_shuffle currently derives average_long_exposure from artifacts.comparison_row["trade_frequency"] at line 722, which is semantically wrong given current metric naming
    - benchmark-relative logic exists, but it is layered on top of existing metric/reporting semantics
```

## 12. Tests and Current Gaps

### Existing useful tests

- [tests/test_metrics.py](/Users/larrym/prediction/tests/test_metrics.py)
  - classification keys
  - trading metric key presence
  - turnover charged only on flips
- [tests/test_walk_forward.py](/Users/larrym/prediction/tests/test_walk_forward.py)
  - walk-forward split generation
- [tests/test_features.py](/Users/larrym/prediction/tests/test_features.py)
  - feature engineering checks
- [tests/test_regime_features.py](/Users/larrym/prediction/tests/test_regime_features.py)
  - regime feature checks

### Missing tests relative to reset plan

- No synthetic active-metric sanity suite for:
  - always-flat
  - buy-and-hold
  - alternating long/flat
  - alternating long/short
- No dedicated tests for:
  - `information_ratio` vs explicit active-return series
  - turnover vs exposure decomposition
  - cost drag / cost per unit active return
  - filtered vs smoothed regime-probability semantics
  - clean report separation between interaction and non-interaction runs

## 13. Highest-Priority Patch Targets for the Next Sprint

These are the exact code paths most directly implicated by the reset plan.

### Measurement repair

1. [evaluation/metrics.py](/Users/larrym/prediction/evaluation/metrics.py)
   - `compute_trading_metrics()`
   - `compute_return_stream_metrics()`
   - `compute_strategy_returns()`
   - `compute_signal_turnover()`
   - likely destination for active-return and turnover decomposition repairs

2. [utils/reporting.py](/Users/larrym/prediction/utils/reporting.py)
   - `summarize_results()`
   - current `net_sharpe` sorting needs replacement or mode-aware ranking

3. [main.py](/Users/larrym/prediction/main.py)
   - `_print_comparison()`
   - `_build_regime_summary()`
   - `_print_regime_weighted_summary()`
   - `_save_ensemble_bias_fix_comparison()`
   - current report composition mixes modes and still emphasizes raw Sharpe

### Matched null framework

4. [audit/integrity_audit.py](/Users/larrym/prediction/audit/integrity_audit.py)
   - useful reference for existing Monte Carlo/null machinery
   - likely foundation for future `null_baselines.py`

### HMM overlay auditability

5. [regime/regime_detection.py](/Users/larrym/prediction/regime/regime_detection.py)
   - `predict()`
   - `identify_best_regime()`
   - `_build_canonical_mapping()`
   - needs explicit review for filtered-vs-smoothed probability semantics and fold-local danger scoring

6. [main.py](/Users/larrym/prediction/main.py)
   - `_build_regime_aware_experiment()`
   - this is where fold-local HMM overlay logic would likely be introduced first

### Ensemble / objective reset

7. [models/ensemble.py](/Users/larrym/prediction/models/ensemble.py)
   - `WeightedAverageEnsemble._objective()`
   - `RegimeWeightedEnsemble._objective()`
   - `RegimeStackingEnsemble._select_threshold()`
   - current ensemble objectives remain mostly raw-Sharpe-oriented or threshold-layer penalties on top of logistic classification

### Tuning objective reset

8. [optimization/optuna_tuner.py](/Users/larrym/prediction/optimization/optuna_tuner.py)
   - `_objective()`
   - `run_optuna_tuning()`
   - explicit raw `net_sharpe` maximize path

## 14. Key Findings Against the Reset Plan

1. The reset-plan concerns are supported by code.
   - `trade_frequency` is currently exposure-in-market, not trade count.
   - `excess_net_sharpe` is currently `net_sharpe - benchmark_sharpe`, not Sharpe of the active-return series.
   - ranking and many summaries still prioritize `net_sharpe`.

2. Benchmark-relative evaluation exists, but only partially.
   - `information_ratio` is already computed from `net_strategy_returns - benchmark_returns`.
   - broader active-return diagnostics from the reset plan are not implemented.

3. HMM test-probability semantics need explicit verification.
   - regime prediction is run on the full frame slice via `predict_proba`.
   - there is no explicit one-step filtered probability path in the current implementation.

4. The current HMM selection logic is still raw-Sharpe-centric.
   - `identify_best_regime()` picks the best regime by regime-conditional `net_sharpe`.

5. Reporting paths remain a major source of context drift.
   - `summarize_results()` sorts by `net_sharpe`.
   - CLI output labels comparisons by `net_sharpe`.
   - regime summary logic mixes interaction and non-interaction artifact sources.

## 15. Recommended Inspection-to-Implementation Order

Before patching, start in this order:

1. [evaluation/metrics.py](/Users/larrym/prediction/evaluation/metrics.py)
2. [utils/reporting.py](/Users/larrym/prediction/utils/reporting.py)
3. [main.py](/Users/larrym/prediction/main.py)
4. [audit/integrity_audit.py](/Users/larrym/prediction/audit/integrity_audit.py)
5. [regime/regime_detection.py](/Users/larrym/prediction/regime/regime_detection.py)
6. [optimization/optuna_tuner.py](/Users/larrym/prediction/optimization/optuna_tuner.py)
7. [models/ensemble.py](/Users/larrym/prediction/models/ensemble.py)

That sequence matches the reset-plan priority:

- repair measurement first
- repair reporting second
- add matched nulls third
- only then revisit regime overlays and model objectives
