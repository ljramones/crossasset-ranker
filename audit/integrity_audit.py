"""Backtest integrity audit for the current frozen champion."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.market_data import load_market_data
from evaluation.metrics import compute_strategy_returns, compute_trading_metrics
from evaluation.null_baselines import evaluate_position_strategy, run_matched_null_suite
from features.feature_engineering import build_feature_set
from features.regime_features import add_regime_features
from models.ensemble import RegimeStackingEnsemble, StackingEnsemble
from optimization.optuna_tuner import _normalize_tuned_params
from regime.regime_detection import (
    MarketRegimeDetector,
    RegimeDetectionConfig,
    add_aggressive_trade_filter_columns,
)
from utils.experiment import (
    PreparedExperiment,
    build_registry,
    configure_runtime_noise,
    evaluate_model,
    prepare_experiment,
)
from utils.reproducibility import seed_everything


RAW_PRICE_COLUMNS = {"Open", "High", "Low", "Close", "Adj Close", "BenchmarkClose", "VIXClose", "Volume"}


@dataclass(slots=True)
class AuditCheck:
    """One integrity audit outcome row."""

    name: str
    status: str
    detail: str
    evidence: dict[str, Any] | None = None


LABEL_SHUFFLE_MONTE_CARLO_RUNS = 50
MATCHED_NULL_MONTE_CARLO_RUNS = 25


def run_integrity_audit(
    *,
    config_path: Path,
    profile_override: str = "full",
    results_dir: Path | None = None,
    comparative_null_test: bool = False,
) -> list[AuditCheck]:
    """Run the integrity audit suite and write a markdown report."""

    experiment = prepare_experiment(config_path, profile_override=profile_override)
    split_budget = int(experiment.config.get("audit", {}).get("split_budget", 3))
    experiment = _limit_splits(experiment, max_splits=min(split_budget, len(experiment.splits)))
    configure_runtime_noise(experiment.config["project"].get("suppress_lightning_warnings", True))
    seed_everything(int(experiment.config["project"]["seed"]))

    report_dir = results_dir or experiment.results_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    checks: list[AuditCheck] = []
    checks.append(_check_no_raw_price_features(experiment))
    checks.append(_check_feature_future_invariance(experiment))
    checks.append(_check_scaler_train_only())
    checks.append(_check_regime_fit_train_only())
    checks.append(_check_regime_probability_future_invariance(experiment))
    checks.append(_check_stacking_oof_time_partition(experiment))
    checks.append(_check_optuna_time_ordering())
    checks.append(_check_transaction_cost_calculation())

    comparative_results: pd.DataFrame | None = None
    adversarial_checks = _run_adversarial_tests(experiment)
    if comparative_null_test:
        comparative_results = _run_comparative_null_test(experiment)
    checks.extend(adversarial_checks)

    _write_report(
        report_dir / "integrity_audit_report.md",
        checks=checks,
        experiment=experiment,
        comparative_results=comparative_results,
    )
    if comparative_results is not None:
        _write_comparative_null_report(report_dir / "comparative_null_test_report.md", comparative_results)
    return checks


def _limit_splits(experiment: PreparedExperiment, max_splits: int) -> PreparedExperiment:
    """Limit split count for an audit run without mutating caller state."""

    if len(experiment.splits) <= max_splits:
        return experiment
    return PreparedExperiment(
        config=experiment.config,
        results_dir=experiment.results_dir,
        feature_set=experiment.feature_set,
        splits=experiment.splits[:max_splits],
    )


def _check_no_raw_price_features(experiment: PreparedExperiment) -> AuditCheck:
    """Ensure the feature matrix excludes raw price levels."""

    leaked = sorted(set(experiment.feature_set.feature_columns).intersection(RAW_PRICE_COLUMNS))
    if leaked:
        return AuditCheck(
            name="raw_price_feature_exclusion",
            status="FAIL",
            detail="Raw price-like columns were found in the feature matrix.",
            evidence={"leaked_columns": leaked},
        )
    return AuditCheck(
        name="raw_price_feature_exclusion",
        status="PASS",
        detail="No raw OHLCV or benchmark level columns appear in the feature matrix.",
        evidence={"feature_count": len(experiment.feature_set.feature_columns)},
    )


def _check_feature_future_invariance(experiment: PreparedExperiment) -> AuditCheck:
    """Changing future raw data must not alter already-known engineered features."""

    market_data = _load_single_asset_market_data(experiment.config)
    original = build_feature_set(
        market_data=market_data,
        target_horizon=experiment.config["features"]["target_horizon"],
        short_window=experiment.config["features"]["rolling_windows"]["short"],
        medium_window=experiment.config["features"]["rolling_windows"]["medium"],
        adf_significance=experiment.config["features"]["adf_significance"],
        dropna=experiment.config["features"]["dropna"],
        advanced_features=experiment.config["features"].get("advanced_features", False),
        vix_features=experiment.config["features"].get("vix_features", False),
    )
    cutoff_position = max(int(len(market_data) * 0.75), 80)
    cutoff_index = market_data.index[min(cutoff_position, len(market_data) - 1)]
    perturbed_market = market_data.copy()
    future_mask = perturbed_market.index > cutoff_index
    perturbation_columns = ["Adj Close", "Close", "Open", "High", "Low", "BenchmarkClose", "VIXClose", "Volume"]
    for column in perturbation_columns:
        if column not in perturbed_market.columns:
            continue
        if column == "Volume":
            perturbed_market.loc[future_mask, column] = perturbed_market.loc[future_mask, column] * 1.75
        else:
            perturbed_market.loc[future_mask, column] = perturbed_market.loc[future_mask, column] * 1.10

    perturbed = build_feature_set(
        market_data=perturbed_market,
        target_horizon=experiment.config["features"]["target_horizon"],
        short_window=experiment.config["features"]["rolling_windows"]["short"],
        medium_window=experiment.config["features"]["rolling_windows"]["medium"],
        adf_significance=experiment.config["features"]["adf_significance"],
        dropna=experiment.config["features"]["dropna"],
        advanced_features=experiment.config["features"].get("advanced_features", False),
        vix_features=experiment.config["features"].get("vix_features", False),
    )

    common_index = original.frame.index.intersection(perturbed.frame.index)
    stable_index = common_index[common_index <= cutoff_index]
    original_slice = original.frame.loc[stable_index, original.feature_columns]
    perturbed_slice = perturbed.frame.loc[stable_index, original.feature_columns]
    max_abs_diff = float((original_slice - perturbed_slice).abs().to_numpy().max()) if not stable_index.empty else 0.0
    status = "PASS" if max_abs_diff < 1e-10 else "FAIL"
    detail = (
        "Future-data perturbation left all pre-cutoff engineered features unchanged."
        if status == "PASS"
        else "Future-data perturbation changed pre-cutoff engineered features."
    )
    return AuditCheck(
        name="feature_engineering_future_invariance",
        status=status,
        detail=detail,
        evidence={"cutoff_date": str(cutoff_index.date()), "max_abs_diff": max_abs_diff},
    )


def _check_scaler_train_only() -> AuditCheck:
    """Inspect champion base model wrappers for train-only scaler fitting."""

    import models.itransformer_model as itransformer_model
    import models.lstm_model as lstm_model
    import models.patchtst_model as patchtst_model
    import models.tft_model as tft_model

    sources = {
        "lstm": inspect.getsource(lstm_model.LSTMModel.fit),
        "itransformer": inspect.getsource(itransformer_model.ITransformerModel.fit),
        "patchtst": inspect.getsource(patchtst_model.PatchTSTModel.fit),
        "temporal_fusion_transformer": inspect.getsource(tft_model.TemporalFusionTransformerModel.fit),
    }
    matched = {
        name: "StandardScaler().fit(train_frame[feature_columns])" in source
        for name, source in sources.items()
    }
    if all(matched.values()):
        return AuditCheck(
            name="scaler_train_only_fit",
            status="PASS",
            detail="All champion base-model wrappers fit their scalers on train_frame only.",
            evidence=matched,
        )
    return AuditCheck(
        name="scaler_train_only_fit",
        status="FAIL",
        detail="One or more champion base-model wrappers did not show an explicit train_frame-only scaler fit.",
        evidence=matched,
    )


def _check_regime_fit_train_only() -> AuditCheck:
    """Inspect the regime detector for train-only fit behavior."""

    fit_source = inspect.getsource(MarketRegimeDetector.fit)
    uses_train_only = "train_frame[self.feature_columns]" in fit_source and "self.scaler.fit_transform(training_features)" in fit_source
    if uses_train_only:
        return AuditCheck(
            name="regime_detector_train_only_fit",
            status="PASS",
            detail="Regime detector fit path scales and fits on training-window features only.",
            evidence={"backend_preference": "hmm", "n_regimes": 3},
        )
    return AuditCheck(
        name="regime_detector_train_only_fit",
        status="FAIL",
        detail="Regime detector fit path did not clearly show train-only fitting.",
        evidence=None,
    )


def _check_regime_probability_future_invariance(experiment: PreparedExperiment) -> AuditCheck:
    """Changing future data must not change already-realized regime labels or probabilities."""

    market_data = _load_single_asset_market_data(experiment.config)
    cutoff_position = max(int(len(market_data) * 0.75), 80)
    cutoff_index = market_data.index[min(cutoff_position, len(market_data) - 1)]

    base_experiment = prepare_experiment_from_market_data_for_audit(experiment.config, market_data)
    base_experiment = _limit_splits(base_experiment, max_splits=min(3, len(base_experiment.splits)))
    original_augmented = _augment_regime_splits(base_experiment)

    perturbed_market = market_data.copy()
    future_mask = perturbed_market.index > cutoff_index
    for column in ["Adj Close", "Close", "Open", "High", "Low", "BenchmarkClose", "VIXClose"]:
        if column in perturbed_market.columns:
            perturbed_market.loc[future_mask, column] = perturbed_market.loc[future_mask, column] * 1.15
    if "Volume" in perturbed_market.columns:
        perturbed_market.loc[future_mask, "Volume"] = perturbed_market.loc[future_mask, "Volume"] * 2.0

    perturbed_experiment = prepare_experiment_from_market_data_for_audit(experiment.config, perturbed_market)
    perturbed_experiment = _limit_splits(perturbed_experiment, max_splits=min(3, len(perturbed_experiment.splits)))
    perturbed_augmented = _augment_regime_splits(perturbed_experiment)

    max_abs_diff = 0.0
    label_mismatch = 0
    for original_split, perturbed_split in zip(original_augmented.splits, perturbed_augmented.splits, strict=False):
        for frame_name in ["train", "validation", "test"]:
            original_frame = getattr(original_split, frame_name)
            perturbed_frame = getattr(perturbed_split, frame_name)
            common_index = original_frame.index.intersection(perturbed_frame.index)
            stable_index = common_index[common_index <= cutoff_index]
            if stable_index.empty:
                continue
            for column in ["regime_prob_0", "regime_prob_1", "regime_prob_2"]:
                if column in original_frame.columns and column in perturbed_frame.columns:
                    diff = (original_frame.loc[stable_index, column] - perturbed_frame.loc[stable_index, column]).abs().max()
                    max_abs_diff = max(max_abs_diff, float(diff))
            if "regime_id" in original_frame.columns and "regime_id" in perturbed_frame.columns:
                label_mismatch += int(
                    original_frame.loc[stable_index, "regime_id"].ne(perturbed_frame.loc[stable_index, "regime_id"]).sum()
                )

    status = "PASS" if max_abs_diff < 1e-10 and label_mismatch == 0 else "FAIL"
    detail = (
        "Future-data perturbation left pre-cutoff regime labels and probabilities unchanged."
        if status == "PASS"
        else "Future-data perturbation changed pre-cutoff regime outputs."
    )
    return AuditCheck(
        name="regime_probability_future_invariance",
        status=status,
        detail=detail,
        evidence={"cutoff_date": str(cutoff_index.date()), "max_abs_diff": max_abs_diff, "label_mismatch_count": label_mismatch},
    )


def _check_stacking_oof_time_partition(experiment: PreparedExperiment) -> AuditCheck:
    """Ensure the meta-learner only trains on prior OOF splits."""

    augmented = _augment_regime_splits(_limit_splits(experiment, max_splits=min(3, len(experiment.splits))))
    tuned_params = _load_best_params(augmented)
    oof_frame = _build_ensemble_oof_frame_local(augmented, tuned_params)
    unique_splits = sorted(oof_frame["split_id"].unique())
    violations: list[dict[str, Any]] = []
    for split_id in unique_splits[1:]:
        train_splits = sorted(oof_frame.loc[oof_frame["split_id"] < split_id, "split_id"].unique().tolist())
        test_splits = sorted(oof_frame.loc[oof_frame["split_id"] == split_id, "split_id"].unique().tolist())
        if any(train_split >= split_id for train_split in train_splits) or test_splits != [split_id]:
            violations.append({"test_split": int(split_id), "train_splits": train_splits, "test_splits": test_splits})
    if violations:
        return AuditCheck(
            name="stacking_oof_partition",
            status="FAIL",
            detail="OOF ensemble partitioning included current/future split data in meta-learner training.",
            evidence={"violations": violations},
        )
    return AuditCheck(
        name="stacking_oof_partition",
        status="PASS",
        detail="OOF ensemble partitioning uses only prior split ids for meta-learner training.",
        evidence={"evaluated_split_ids": [int(value) for value in unique_splits]},
    )


def _check_optuna_time_ordering() -> AuditCheck:
    """Inspect the tuner for walk-forward reuse instead of random CV leakage."""

    import optimization.optuna_tuner as optuna_tuner

    objective_source = inspect.getsource(optuna_tuner._objective)
    evaluate_source = inspect.getsource(optuna_tuner.evaluate_tuned_models)
    uses_walk_forward = "evaluate_model(" in objective_source and "splits=experiment.splits" in objective_source
    reevaluates_out_of_sample = "evaluate_model(" in evaluate_source and "splits=experiment.splits" in evaluate_source
    status = "PASS" if uses_walk_forward and reevaluates_out_of_sample else "FAIL"
    return AuditCheck(
        name="optuna_walk_forward_discipline",
        status=status,
        detail="Optuna objective and tuned re-evaluation use the same walk-forward split structure." if status == "PASS" else "Optuna source did not clearly show walk-forward-only evaluation.",
        evidence={
            "objective_uses_walk_forward": uses_walk_forward,
            "tuned_model_reevaluation_uses_walk_forward": reevaluates_out_of_sample,
        },
    )


def _check_transaction_cost_calculation() -> AuditCheck:
    """Verify that costs are charged on executed position flips only."""

    returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], name="returns")
    signal = pd.Series([0, 1, 1, 0, 1], name="signal")
    net_returns = compute_strategy_returns(returns=returns, signal=signal, transaction_cost_bps=2.0)
    executed_position = signal.shift(1).fillna(0.0)
    expected_turnover = executed_position.diff().abs().fillna(executed_position.abs())
    expected_costs = expected_turnover * (2.0 / 10_000.0)
    expected_returns = -expected_costs
    max_abs_diff = float((net_returns - expected_returns).abs().max())
    status = "PASS" if max_abs_diff < 1e-12 else "FAIL"
    return AuditCheck(
        name="transaction_cost_flip_only",
        status=status,
        detail="Transaction costs are applied on executed position flips only." if status == "PASS" else "Transaction-cost path diverged from flip-only expectation.",
        evidence={
            "executed_position": executed_position.tolist(),
            "expected_turnover": expected_turnover.tolist(),
            "max_abs_diff": max_abs_diff,
        },
    )


def _run_adversarial_tests(experiment: PreparedExperiment) -> list[AuditCheck]:
    """Run lightweight adversarial tests against the frozen regime stack."""

    audited = _augment_regime_splits(_limit_splits(experiment, max_splits=min(3, len(experiment.splits))))
    tuned_params = _load_best_params(audited)
    oof_frame = _build_ensemble_oof_frame_local(audited, tuned_params)
    canonical_metrics = _evaluate_regime_stacking_meta(oof_frame=oof_frame, include_randomized_regimes=False, shuffle_labels=False, extra_signal_lag=0)
    label_shuffle_check = _run_label_shuffle_monte_carlo(
        oof_frame,
        n_shuffles=int(experiment.config.get("audit", {}).get("comparative_null_test", {}).get("n_shuffles", LABEL_SHUFFLE_MONTE_CARLO_RUNS)),
        significance_threshold=float(experiment.config.get("audit", {}).get("comparative_null_test", {}).get("significance_threshold", 0.05)),
    )
    lagged_metrics = _evaluate_regime_stacking_meta(oof_frame=oof_frame, include_randomized_regimes=False, shuffle_labels=False, extra_signal_lag=1)
    randomized_regime_metrics = _evaluate_regime_stacking_meta(oof_frame=oof_frame, include_randomized_regimes=True, shuffle_labels=False, extra_signal_lag=0)

    checks = [
        _adversarial_check(
            name="adversarial_signal_lag_plus_one_day",
            canonical=canonical_metrics,
            adversarial=lagged_metrics,
            detail="Adding an extra one-day signal lag should not outperform the canonical regime stack.",
        ),
        _adversarial_check(
            name="adversarial_randomized_regime_probabilities",
            canonical=canonical_metrics,
            adversarial=randomized_regime_metrics,
            detail="Randomized regime probabilities should not outperform the canonical regime stack.",
        ),
    ]
    checks.insert(
        0,
        AuditCheck(
            name="adversarial_reference_canonical",
            status="INFO",
            detail="Reference audit-sample performance for the frozen regime stack.",
            evidence=canonical_metrics,
        ),
    )
    checks.insert(1, label_shuffle_check)
    return checks


def _adversarial_check(name: str, canonical: dict[str, float], adversarial: dict[str, float], detail: str) -> AuditCheck:
    """Compare canonical and adversarial performance and flag suspicious improvements."""

    suspicious = adversarial["net_sharpe"] > canonical["net_sharpe"] + 0.05
    return AuditCheck(
        name=name,
        status="FAIL" if suspicious else "PASS",
        detail=detail,
        evidence={
            "canonical_net_sharpe": canonical["net_sharpe"],
            "adversarial_net_sharpe": adversarial["net_sharpe"],
            "canonical_calmar": canonical["calmar"],
            "adversarial_calmar": adversarial["calmar"],
        },
    )


def _run_label_shuffle_monte_carlo(
    oof_frame: pd.DataFrame,
    *,
    n_shuffles: int,
    significance_threshold: float,
) -> AuditCheck:
    """Run a Monte Carlo label-shuffle null test against the canonical regime stack."""

    canonical = _evaluate_regime_stacking_meta(
        oof_frame=oof_frame,
        include_randomized_regimes=False,
        shuffle_labels=False,
        extra_signal_lag=0,
    )
    shuffled_sharpes: list[float] = []
    shuffled_information_ratios: list[float] = []
    shuffled_excess_sharpes: list[float] = []

    for seed in range(n_shuffles):
        shuffled = _evaluate_regime_stacking_meta(
            oof_frame=oof_frame,
            include_randomized_regimes=False,
            shuffle_labels=True,
            extra_signal_lag=0,
            shuffle_seed=10_000 + seed,
        )
        shuffled_sharpes.append(float(shuffled["net_sharpe"]))
        shuffled_information_ratios.append(float(shuffled["information_ratio"]))
        shuffled_excess_sharpes.append(float(shuffled["excess_net_sharpe"]))

    shuffled_array = np.asarray(shuffled_sharpes, dtype=float)
    shuffled_ir_array = np.asarray(shuffled_information_ratios, dtype=float)
    shuffled_excess_array = np.asarray(shuffled_excess_sharpes, dtype=float)
    mean_shuffled = float(np.mean(shuffled_array))
    percentile_95 = float(np.percentile(shuffled_array, 95))
    mean_shuffled_ir = float(np.mean(shuffled_ir_array))
    percentile_95_ir = float(np.percentile(shuffled_ir_array, 95))
    mean_shuffled_excess = float(np.mean(shuffled_excess_array))
    percentile_95_excess = float(np.percentile(shuffled_excess_array, 95))
    exceedances = int(np.sum(shuffled_ir_array >= canonical["information_ratio"]))
    p_value = float(exceedances / len(shuffled_ir_array))
    passes = canonical["information_ratio"] > percentile_95_ir and p_value < significance_threshold

    return AuditCheck(
        name="adversarial_label_shuffle",
        status="PASS" if passes else "FAIL",
        detail="Monte Carlo label-shuffle null test for the regime-stacking meta-learner using benchmark-relative skill as the primary decision metric.",
        evidence={
            "runs": n_shuffles,
            "canonical_net_sharpe": round(float(canonical["net_sharpe"]), 6),
            "canonical_information_ratio": round(float(canonical["information_ratio"]), 6),
            "canonical_excess_net_sharpe": round(float(canonical["excess_net_sharpe"]), 6),
            "mean_shuffled_net_sharpe": round(mean_shuffled, 6),
            "percentile_95_shuffled_net_sharpe": round(percentile_95, 6),
            "mean_shuffled_information_ratio": round(mean_shuffled_ir, 6),
            "percentile_95_shuffled_information_ratio": round(percentile_95_ir, 6),
            "mean_shuffled_excess_net_sharpe": round(mean_shuffled_excess, 6),
            "percentile_95_shuffled_excess_net_sharpe": round(percentile_95_excess, 6),
            "p_value": round(p_value, 6),
            "significance_threshold": significance_threshold,
            "exceedance_count": exceedances,
            "decision_metric": "information_ratio",
        },
    )


def _run_comparative_null_test(experiment: PreparedExperiment) -> pd.DataFrame:
    """Run a comparative null test across ensemble and strong single-model candidates."""

    comparative_config = experiment.config.get("audit", {}).get("comparative_null_test", {})
    n_shuffles = int(comparative_config.get("n_shuffles", LABEL_SHUFFLE_MONTE_CARLO_RUNS))
    significance_threshold = float(comparative_config.get("significance_threshold", 0.05))
    matched_null_config = experiment.config.get("audit", {}).get("matched_null_test", {})
    matched_null_runs = int(matched_null_config.get("n_runs", comparative_config.get("matched_null_runs", MATCHED_NULL_MONTE_CARLO_RUNS)))
    matched_null_seed = int(matched_null_config.get("seed", 42))
    matched_null_decision_metric = str(matched_null_config.get("decision_metric", "information_ratio"))
    configured_models = list(
        comparative_config.get(
            "models",
            [
                "regime_stacking_ensemble_regime",
                "stacking_ensemble_baseline",
                "itransformer_tuned_regime",
                "lstm_regime",
            ],
        )
    )

    audited = _augment_regime_splits(_limit_splits(experiment, max_splits=min(int(experiment.config.get("audit", {}).get("split_budget", 3)), len(experiment.splits))))
    tuned_params = _load_best_params(audited)
    oof_frame = _build_ensemble_oof_frame_local(audited, tuned_params)

    rows: list[dict[str, Any]] = []
    for model_label in configured_models:
        result = _run_one_model_comparative_null(
            experiment=audited,
            oof_frame=oof_frame,
            model_label=model_label,
            tuned_params=tuned_params,
            n_shuffles=n_shuffles,
            significance_threshold=significance_threshold,
            matched_null_runs=matched_null_runs,
            matched_null_seed=matched_null_seed,
            matched_null_decision_metric=matched_null_decision_metric,
        )
        rows.append(result)
    return pd.DataFrame(rows).sort_values("canonical_net_sharpe", ascending=False).reset_index(drop=True)


def _run_one_model_comparative_null(
    *,
    experiment: PreparedExperiment,
    oof_frame: pd.DataFrame,
    model_label: str,
    tuned_params: dict[str, dict[str, Any]],
    n_shuffles: int,
    significance_threshold: float,
    matched_null_runs: int,
    matched_null_seed: int,
    matched_null_decision_metric: str,
) -> dict[str, Any]:
    """Run the Monte Carlo null test for one model label."""

    if model_label == "regime_stacking_ensemble_regime":
        canonical_frame = _collect_regime_stacking_prediction_frame(
            oof_frame=oof_frame,
            include_randomized_regimes=False,
            shuffle_labels=False,
            extra_signal_lag=0,
        )
        canonical = _evaluate_regime_stacking_meta(
            oof_frame=oof_frame,
            include_randomized_regimes=False,
            shuffle_labels=False,
            extra_signal_lag=0,
        )
        shuffled = [
            _evaluate_regime_stacking_meta(
                oof_frame=oof_frame,
                include_randomized_regimes=False,
                shuffle_labels=True,
                extra_signal_lag=0,
                shuffle_seed=20_000 + seed,
            )
            for seed in range(n_shuffles)
        ]
        model_type = "ensemble_meta_learner"
    elif model_label == "stacking_ensemble_baseline":
        canonical_frame = _collect_stacking_prediction_frame(
            oof_frame=oof_frame,
            shuffle_labels=False,
            shuffle_seed=42,
        )
        canonical = _evaluate_stacking_meta(
            oof_frame=oof_frame,
            shuffle_labels=False,
            shuffle_seed=42,
        )
        shuffled = [
            _evaluate_stacking_meta(
                oof_frame=oof_frame,
                shuffle_labels=True,
                shuffle_seed=30_000 + seed,
            )
            for seed in range(n_shuffles)
        ]
        model_type = "ensemble_meta_learner"
    elif model_label == "itransformer_tuned_regime":
        single_model_artifacts = _evaluate_single_model_artifacts(
            experiment=experiment,
            model_name="itransformer",
            tuned_params=tuned_params.get("itransformer"),
            shuffle_labels=False,
            shuffle_seed=42,
        )
        canonical_frame = _collect_single_model_prediction_frame(single_model_artifacts.oof_predictions)
        canonical = _evaluate_single_model_null_target(
            experiment=experiment,
            model_name="itransformer",
            tuned_params=tuned_params.get("itransformer"),
            shuffle_labels=False,
            shuffle_seed=42,
        )
        shuffled = [
            _evaluate_single_model_null_target(
                experiment=experiment,
                model_name="itransformer",
                tuned_params=tuned_params.get("itransformer"),
                shuffle_labels=True,
                shuffle_seed=40_000 + seed,
            )
            for seed in range(n_shuffles)
        ]
        model_type = "single_model"
    elif model_label == "lstm_regime":
        single_model_artifacts = _evaluate_single_model_artifacts(
            experiment=experiment,
            model_name="lstm",
            tuned_params=None,
            shuffle_labels=False,
            shuffle_seed=42,
        )
        canonical_frame = _collect_single_model_prediction_frame(single_model_artifacts.oof_predictions)
        canonical = _evaluate_single_model_null_target(
            experiment=experiment,
            model_name="lstm",
            tuned_params=None,
            shuffle_labels=False,
            shuffle_seed=42,
        )
        shuffled = [
            _evaluate_single_model_null_target(
                experiment=experiment,
                model_name="lstm",
                tuned_params=None,
                shuffle_labels=True,
                shuffle_seed=50_000 + seed,
            )
            for seed in range(n_shuffles)
        ]
        model_type = "single_model"
    else:
        raise ValueError(f"Unsupported comparative null-test model label: {model_label}")

    shuffled_net_sharpe_array = np.asarray([float(item["net_sharpe"]) for item in shuffled], dtype=float)
    shuffled_ir_array = np.asarray([float(item["information_ratio"]) for item in shuffled], dtype=float)
    shuffled_excess_array = np.asarray([float(item["excess_net_sharpe"]) for item in shuffled], dtype=float)
    mean_shuffled = float(np.mean(shuffled_net_sharpe_array))
    percentile_95 = float(np.percentile(shuffled_net_sharpe_array, 95))
    mean_shuffled_ir = float(np.mean(shuffled_ir_array))
    percentile_95_ir = float(np.percentile(shuffled_ir_array, 95))
    mean_shuffled_excess = float(np.mean(shuffled_excess_array))
    percentile_95_excess = float(np.percentile(shuffled_excess_array, 95))
    exceedances = int(np.sum(shuffled_ir_array >= canonical["information_ratio"]))
    p_value = float(exceedances / len(shuffled_ir_array))
    passed = canonical["information_ratio"] > percentile_95_ir and p_value < significance_threshold
    matched_null_summary = _run_matched_null_audit(
        model_name=model_label,
        asset_returns=canonical_frame["forward_simple_return_1d"],
        canonical_positions=canonical_frame["executed_position"],
        benchmark_returns=canonical_frame["benchmark_return_1d"],
        regime_labels=canonical_frame.get("regime_id"),
        canonical_metrics=None,
        transaction_cost_bps=2.0,
        n_runs=matched_null_runs,
        random_state=matched_null_seed,
        decision_metric=matched_null_decision_metric,
        significance_threshold=significance_threshold,
    )
    result = {
        "model": model_label,
        "model_type": model_type,
        "runs": n_shuffles,
        "canonical_net_sharpe": float(canonical["net_sharpe"]),
        "canonical_net_sortino": float(canonical["net_sortino"]),
        "canonical_calmar": float(canonical["calmar"]),
        "canonical_information_ratio": float(canonical["information_ratio"]),
        "canonical_excess_net_sharpe": float(canonical["excess_net_sharpe"]),
        "canonical_average_long_exposure": float(canonical["average_long_exposure"]),
        "canonical_average_position_size": float(canonical["average_position_size"]),
        "canonical_fraction_positive_predictions": float(canonical["fraction_positive_predictions"]),
        "mean_shuffled_net_sharpe": mean_shuffled,
        "percentile_95_shuffled_net_sharpe": percentile_95,
        "mean_shuffled_information_ratio": mean_shuffled_ir,
        "percentile_95_shuffled_information_ratio": percentile_95_ir,
        "mean_shuffled_excess_net_sharpe": mean_shuffled_excess,
        "percentile_95_shuffled_excess_net_sharpe": percentile_95_excess,
        "p_value": p_value,
        "significance_threshold": significance_threshold,
        "pass": bool(passed),
        "decision": "PASS" if passed else "FAIL",
        "decision_metric": "information_ratio",
        "matched_null_runs": matched_null_runs,
    }
    result.update(matched_null_summary)
    return result


def _evaluate_stacking_meta(
    *,
    oof_frame: pd.DataFrame,
    shuffle_labels: bool,
    shuffle_seed: int,
) -> dict[str, float]:
    """Evaluate the baseline stacking meta-learner on the OOF frame."""

    rng = np.random.default_rng(shuffle_seed)
    base_columns = [column for column in oof_frame.columns if column.startswith("probability__")]
    unique_splits = sorted(oof_frame["split_id"].unique())
    rows: list[dict[str, float]] = []
    for split_id in unique_splits[1:]:
        train_frame = oof_frame.loc[oof_frame["split_id"] < split_id].copy()
        test_frame = oof_frame.loc[oof_frame["split_id"] == split_id].copy()
        if shuffle_labels:
            shuffled = train_frame["target_direction"].to_numpy(copy=True)
            rng.shuffle(shuffled)
            train_frame["target_direction"] = shuffled
        ensemble = StackingEnsemble(signal_threshold=0.5)
        ensemble.fit(train_frame=train_frame, base_columns=base_columns)
        prediction = ensemble.predict(test_frame)
        rows.append(
            _build_strategy_diagnostics(
                returns=test_frame["forward_simple_return_1d"],
                benchmark_returns=test_frame["benchmark_return_1d"],
                signal=prediction.predictions,
                prediction=prediction.predictions,
            )
        )
    frame = pd.DataFrame(rows)
    return {
        "net_sharpe": float(frame["net_sharpe"].mean()),
        "net_sortino": float(frame["net_sortino"].mean()),
        "calmar": float(frame["calmar"].mean()),
        "information_ratio": float(frame["information_ratio"].mean()),
        "excess_net_sharpe": float(frame["excess_net_sharpe"].mean()),
        "average_long_exposure": float(frame["average_long_exposure"].mean()),
        "average_position_size": float(frame["average_position_size"].mean()),
        "fraction_positive_predictions": float(frame["fraction_positive_predictions"].mean()),
    }


def _evaluate_single_model_null_target(
    *,
    experiment: PreparedExperiment,
    model_name: str,
    tuned_params: dict[str, Any] | None,
    shuffle_labels: bool,
    shuffle_seed: int,
) -> dict[str, float]:
    """Evaluate one regime-aware single model under a label-shuffled null."""

    artifacts = _evaluate_single_model_artifacts(
        experiment=experiment,
        model_name=model_name,
        tuned_params=tuned_params,
        shuffle_labels=shuffle_labels,
        shuffle_seed=shuffle_seed,
    )
    return {
        "net_sharpe": float(artifacts.comparison_row["net_sharpe"]),
        "net_sortino": float(artifacts.comparison_row["net_sortino"]),
        "calmar": float(artifacts.comparison_row["calmar"]),
        "information_ratio": float(artifacts.comparison_row["information_ratio"]),
        "excess_net_sharpe": float(artifacts.comparison_row["net_sharpe"] - _benchmark_net_sharpe_from_oof(artifacts.oof_predictions)),
        "average_long_exposure": float(artifacts.comparison_row.get("average_long_exposure", artifacts.comparison_row.get("fraction_in_market", artifacts.comparison_row["trade_frequency"]))),
        "average_position_size": float(artifacts.oof_predictions["prediction"].shift(1).fillna(0.0).abs().mean()),
        "fraction_positive_predictions": float(artifacts.oof_predictions["prediction"].mean()),
    }


def _evaluate_single_model_artifacts(
    *,
    experiment: PreparedExperiment,
    model_name: str,
    tuned_params: dict[str, Any] | None,
    shuffle_labels: bool,
    shuffle_seed: int,
):
    """Evaluate one regime-aware single model and return full artifacts."""

    model, label = _instantiate_model_local(experiment, model_name, tuned_params)
    shuffled_experiment = experiment
    if shuffle_labels:
        shuffled_experiment = _with_shuffled_train_validation_labels(experiment, seed=shuffle_seed)
    return evaluate_model(
        model=model,
        label=label,
        feature_columns=shuffled_experiment.feature_set.feature_columns,
        splits=shuffled_experiment.splits,
        portfolio_config=shuffled_experiment.config["portfolio"],
        log_progress=False,
    )


def _benchmark_net_sharpe_from_oof(oof_predictions: pd.DataFrame) -> float:
    """Compute benchmark Sharpe on the OOF sample."""

    return _benchmark_net_sharpe_from_returns(oof_predictions["benchmark_return_1d"].astype(float))


def _with_shuffled_train_validation_labels(experiment: PreparedExperiment, seed: int) -> PreparedExperiment:
    """Return a copy of the experiment with shuffled train/validation labels only."""

    rng = np.random.default_rng(seed)
    shuffled_splits = []
    for split in experiment.splits:
        train = split.train.copy()
        validation = split.validation.copy()
        train_labels = train["target_direction"].to_numpy(copy=True)
        validation_labels = validation["target_direction"].to_numpy(copy=True)
        rng.shuffle(train_labels)
        rng.shuffle(validation_labels)
        train["target_direction"] = train_labels
        validation["target_direction"] = validation_labels
        shuffled_splits.append(
            type(split)(
                train=train,
                validation=validation,
                test=split.test.copy(),
                split_id=split.split_id,
            )
        )
    return PreparedExperiment(
        config=experiment.config,
        results_dir=experiment.results_dir,
        feature_set=experiment.feature_set,
        splits=shuffled_splits,
    )


def _evaluate_regime_stacking_meta(
    *,
    oof_frame: pd.DataFrame,
    include_randomized_regimes: bool,
    shuffle_labels: bool,
    extra_signal_lag: int,
    shuffle_seed: int = 42,
) -> dict[str, float]:
    """Evaluate the regime-stacking meta learner on an OOF frame under perturbation."""

    rng = np.random.default_rng(shuffle_seed)
    base_columns = [column for column in oof_frame.columns if column.startswith("probability__")]
    unique_splits = sorted(oof_frame["split_id"].unique())
    rows: list[dict[str, float]] = []

    for split_id in unique_splits[1:]:
        train_frame = oof_frame.loc[oof_frame["split_id"] < split_id].copy()
        test_frame = oof_frame.loc[oof_frame["split_id"] == split_id].copy()

        if shuffle_labels:
            shuffled = train_frame["target_direction"].to_numpy(copy=True)
            rng.shuffle(shuffled)
            train_frame["target_direction"] = shuffled

        if include_randomized_regimes:
            regime_columns = [column for column in train_frame.columns if column.startswith("regime_prob_")]
            for column in regime_columns:
                train_frame[column] = rng.permutation(train_frame[column].to_numpy())
                test_frame[column] = rng.permutation(test_frame[column].to_numpy())

        ensemble = RegimeStackingEnsemble(signal_threshold=0.5, meta_learner="logistic", include_interactions=False)
        ensemble.fit(train_frame=train_frame, base_columns=base_columns, n_regimes=3)
        prediction = ensemble.predict(test_frame)
        signal = prediction.predictions.copy()
        if extra_signal_lag > 0:
            signal = signal.shift(extra_signal_lag).fillna(0).astype(int)
        rows.append(
            _build_strategy_diagnostics(
                returns=test_frame["forward_simple_return_1d"],
                benchmark_returns=test_frame["benchmark_return_1d"],
                signal=signal,
                prediction=prediction.predictions,
            )
        )

    frame = pd.DataFrame(rows)
    return {
        "net_sharpe": float(frame["net_sharpe"].mean()),
        "net_sortino": float(frame["net_sortino"].mean()),
        "calmar": float(frame["calmar"].mean()),
        "information_ratio": float(frame["information_ratio"].mean()),
        "excess_net_sharpe": float(frame["excess_net_sharpe"].mean()),
        "average_long_exposure": float(frame["average_long_exposure"].mean()),
        "average_position_size": float(frame["average_position_size"].mean()),
        "fraction_positive_predictions": float(frame["fraction_positive_predictions"].mean()),
    }


def _build_strategy_diagnostics(
    *,
    returns: pd.Series,
    benchmark_returns: pd.Series,
    signal: pd.Series,
    prediction: pd.Series,
) -> dict[str, float]:
    """Compute raw, benchmark-relative, and exposure diagnostics for one prediction stream."""

    metrics = compute_trading_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        signal=signal,
        annualization_factor=252,
        transaction_cost_bps=2.0,
    )
    benchmark_sharpe = _benchmark_net_sharpe_from_returns(benchmark_returns)
    executed_position = signal.shift(1).fillna(0.0)
    return {
        "net_sharpe": float(metrics["net_sharpe"]),
        "net_sortino": float(metrics["net_sortino"]),
        "calmar": float(metrics["calmar"]),
        "information_ratio": float(metrics["information_ratio"]),
        "excess_net_sharpe": float(metrics["net_sharpe"] - benchmark_sharpe),
        "average_long_exposure": float(executed_position.mean()),
        "average_position_size": float(executed_position.abs().mean()),
        "fraction_positive_predictions": float(prediction.mean()),
    }


def _collect_stacking_prediction_frame(
    *,
    oof_frame: pd.DataFrame,
    shuffle_labels: bool,
    shuffle_seed: int,
) -> pd.DataFrame:
    """Collect split-local stacking predictions and executed positions."""

    rng = np.random.default_rng(shuffle_seed)
    base_columns = [column for column in oof_frame.columns if column.startswith("probability__")]
    unique_splits = sorted(oof_frame["split_id"].unique())
    frames: list[pd.DataFrame] = []
    for split_id in unique_splits[1:]:
        train_frame = oof_frame.loc[oof_frame["split_id"] < split_id].copy()
        test_frame = oof_frame.loc[oof_frame["split_id"] == split_id].copy()
        if shuffle_labels:
            shuffled = train_frame["target_direction"].to_numpy(copy=True)
            rng.shuffle(shuffled)
            train_frame["target_direction"] = shuffled
        ensemble = StackingEnsemble(signal_threshold=0.5)
        ensemble.fit(train_frame=train_frame, base_columns=base_columns)
        prediction = ensemble.predict(test_frame)
        signal = prediction.predictions.astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "date": test_frame["date"].values,
                    "split_id": test_frame["split_id"].values,
                    "forward_simple_return_1d": test_frame["forward_simple_return_1d"].astype(float).values,
                    "benchmark_return_1d": test_frame["benchmark_return_1d"].astype(float).values,
                    "prediction": signal.values,
                    # Split-local execution: first bar of each test slice starts flat.
                    "executed_position": signal.shift(1).fillna(0.0).astype(float).values,
                    "regime_id": test_frame["regime_id"].values if "regime_id" in test_frame.columns else np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _collect_regime_stacking_prediction_frame(
    *,
    oof_frame: pd.DataFrame,
    include_randomized_regimes: bool,
    shuffle_labels: bool,
    extra_signal_lag: int,
    shuffle_seed: int = 42,
) -> pd.DataFrame:
    """Collect split-local regime-stacking predictions and executed positions."""

    rng = np.random.default_rng(shuffle_seed)
    base_columns = [column for column in oof_frame.columns if column.startswith("probability__")]
    unique_splits = sorted(oof_frame["split_id"].unique())
    frames: list[pd.DataFrame] = []
    for split_id in unique_splits[1:]:
        train_frame = oof_frame.loc[oof_frame["split_id"] < split_id].copy()
        test_frame = oof_frame.loc[oof_frame["split_id"] == split_id].copy()
        if shuffle_labels:
            shuffled = train_frame["target_direction"].to_numpy(copy=True)
            rng.shuffle(shuffled)
            train_frame["target_direction"] = shuffled
        if include_randomized_regimes:
            regime_columns = [column for column in train_frame.columns if column.startswith("regime_prob_")]
            for column in regime_columns:
                train_frame[column] = rng.permutation(train_frame[column].to_numpy())
                test_frame[column] = rng.permutation(test_frame[column].to_numpy())
        ensemble = RegimeStackingEnsemble(signal_threshold=0.5, meta_learner="logistic", include_interactions=False)
        ensemble.fit(train_frame=train_frame, base_columns=base_columns, n_regimes=3)
        prediction = ensemble.predict(test_frame)
        signal = prediction.predictions.copy().astype(float)
        if extra_signal_lag > 0:
            signal = signal.shift(extra_signal_lag).fillna(0.0).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "date": test_frame["date"].values,
                    "split_id": test_frame["split_id"].values,
                    "forward_simple_return_1d": test_frame["forward_simple_return_1d"].astype(float).values,
                    "benchmark_return_1d": test_frame["benchmark_return_1d"].astype(float).values,
                    "prediction": prediction.predictions.astype(float).values,
                    # Split-local execution: first bar of each test slice starts flat.
                    "executed_position": signal.shift(1).fillna(0.0).astype(float).values,
                    "regime_id": test_frame["regime_id"].values if "regime_id" in test_frame.columns else np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _collect_single_model_prediction_frame(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Collect split-local executed positions from a single-model OOF frame."""

    frame = oof_predictions.sort_values(["split_id", "date"]).copy()
    frame["prediction"] = frame["prediction"].astype(float)
    frame["executed_position"] = (
        frame.groupby("split_id", sort=False)["prediction"].shift(1).fillna(0.0).astype(float)
    )
    if "regime_id" not in frame.columns:
        frame["regime_id"] = np.nan
    keep_columns = [
        "date",
        "split_id",
        "forward_simple_return_1d",
        "benchmark_return_1d",
        "prediction",
        "executed_position",
        "regime_id",
    ]
    return frame[keep_columns].reset_index(drop=True)


def _run_matched_null_audit(
    model_name: str,
    asset_returns,
    canonical_positions,
    benchmark_returns=None,
    regime_labels=None,
    canonical_metrics=None,
    transaction_cost_bps: float = 0.0,
    n_runs: int = 100,
    random_state: int | None = None,
    metrics_to_summarize: list[str] | None = None,
    decision_metric: str = "information_ratio",
    significance_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate matched null baselines on executed positions using active metrics.

    The `canonical_positions` input must already be an executed position path.
    This helper intentionally does not shift again.
    """

    del metrics_to_summarize  # Reserved for future expansion once more active diagnostics are summarized.
    benchmark = benchmark_returns if benchmark_returns is not None else pd.Series(0.0, index=pd.Series(asset_returns).index)
    suite = run_matched_null_suite(
        positions=canonical_positions,
        returns=asset_returns,
        benchmark_returns=benchmark,
        regime_labels=regime_labels,
        n_runs=n_runs,
        seed=42 if random_state is None else int(random_state),
        annualization_factor=252,
        transaction_cost_bps=transaction_cost_bps,
        decision_metric=decision_metric,
        include_block_bootstrap=True,
    )
    canonical = dict(canonical_metrics or suite["canonical_metrics"])
    result: dict[str, Any] = {
        "matched_null_decision_metric": decision_metric,
        "matched_null_canonical_information_ratio": float(canonical.get("information_ratio", 0.0)),
        "matched_null_canonical_fraction_in_market": float(canonical.get("fraction_in_market", 0.0)),
        "matched_null_canonical_daily_turnover": float(canonical.get("daily_turnover", 0.0)),
        "matched_null_canonical_position_flip_count": float(canonical.get("position_flip_count", 0.0)),
    }
    for label, payload in suite["null_summaries"].items():
        summary = payload["summary"]
        passed = summary.canonical_value > summary.percentile_95_null_value and summary.p_value < significance_threshold
        prefix = f"matched_null__{label}"
        result[f"{prefix}__mean"] = float(summary.mean_null_value)
        result[f"{prefix}__p95"] = float(summary.percentile_95_null_value)
        result[f"{prefix}__p_value"] = float(summary.p_value)
        result[f"{prefix}__runs"] = int(summary.n_runs)
        result[f"{prefix}__decision"] = "PASS" if passed else "FAIL"
    return result


def _benchmark_net_sharpe_from_returns(benchmark_returns: pd.Series) -> float:
    """Compute Sharpe for a benchmark return stream."""

    std = float(benchmark_returns.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252.0) * benchmark_returns.mean() / std)


def _build_ensemble_oof_frame_local(
    experiment: PreparedExperiment,
    tuned_params: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Locally rebuild the champion OOF base-model frame for audit purposes."""

    merged_frame: pd.DataFrame | None = None
    base_model_names = list(experiment.config["regime"]["base_models"])
    for model_name in base_model_names:
        model, label = _instantiate_model_local(experiment, model_name, tuned_params.get(model_name))
        artifacts = evaluate_model(
            model=model,
            label=label,
            feature_columns=experiment.feature_set.feature_columns,
            splits=experiment.splits,
            portfolio_config=experiment.config["portfolio"],
            log_progress=False,
        )
        frame = artifacts.oof_predictions.rename(columns={"probability": f"probability__{label}"})
        metadata_columns = [
            column
            for column in frame.columns
            if column in {"regime_id", "best_regime_id", "best_regime_prob", "trade_allowed_aggressive"}
            or column.startswith("regime_prob_")
        ]
        keep_columns = [
            "date",
            "split_id",
            "target_direction",
            "forward_simple_return_1d",
            "benchmark_return_1d",
            f"probability__{label}",
            *metadata_columns,
        ]
        frame = frame[keep_columns]
        if merged_frame is None:
            merged_frame = frame
        else:
            join_columns = ["date", "split_id", "target_direction", "forward_simple_return_1d", "benchmark_return_1d"]
            for metadata_column in metadata_columns:
                if metadata_column in merged_frame.columns:
                    join_columns.append(metadata_column)
            merged_frame = merged_frame.merge(frame, on=join_columns, how="inner")

    if merged_frame is None:
        raise ValueError("Failed to build the audit OOF frame.")
    return merged_frame.sort_values(["split_id", "date"]).reset_index(drop=True)


def _instantiate_model_local(
    experiment: PreparedExperiment,
    model_name: str,
    tuned_params: dict[str, Any] | None,
):
    """Instantiate one model using the experiment config and optional tuned params."""

    config = dict(experiment.config)
    config["models"] = dict(experiment.config["models"])
    if tuned_params:
        config["models"][model_name] = {
            **config["models"][model_name],
            **_normalize_tuned_params(model_name=model_name, params=tuned_params),
        }
    registry = build_registry(config, experiment.results_dir)
    model = registry.get_model(model_name)
    return model, model.get_display_name()


def _load_best_params(experiment: PreparedExperiment) -> dict[str, dict[str, Any]]:
    """Load persisted tuned params for the champion base models."""

    best_params_dir = Path(experiment.config["tuning"]["best_params_dir"])
    params_by_model: dict[str, dict[str, Any]] = {}
    if not best_params_dir.exists():
        return params_by_model
    for path in best_params_dir.glob("*.json"):
        params_by_model[path.stem] = _normalize_tuned_params(
            model_name=path.stem,
            params=json.loads(path.read_text(encoding="utf-8")),
        )
    return params_by_model


def prepare_experiment_from_market_data_for_audit(config: dict[str, Any], market_data: pd.DataFrame) -> PreparedExperiment:
    """Prepare an experiment from a provided market-data frame without reloading from disk."""

    from utils.experiment import prepare_experiment_from_market_data

    return prepare_experiment_from_market_data(config=config, market_data=market_data)


def _augment_regime_splits(experiment: PreparedExperiment) -> PreparedExperiment:
    """Locally augment splits with regime features using train-only detector fits."""

    regime_config = RegimeDetectionConfig(
        model_type=str(experiment.config["regime"].get("model_type", "hmm")),
        n_regimes=int(experiment.config["regime"].get("n_regimes", 3)),
    )
    min_regime_prob = float(experiment.config["regime"].get("min_regime_prob", 0.70))
    augmented_splits = []
    regime_feature_columns: list[str] | None = None

    for split in experiment.splits:
        detector = MarketRegimeDetector(regime_config)
        detector.fit(split.train)
        train_augmented = split.train.copy()
        validation_augmented = split.validation.copy()
        test_augmented = split.test.copy()
        for frame in [train_augmented, validation_augmented, test_augmented]:
            prediction = detector.predict(frame)
            frame["regime_id"] = prediction.labels.values
            for column in prediction.probabilities.columns:
                frame[column] = prediction.probabilities[column].values
        decision = detector.identify_best_regime(
            train_frame=train_augmented,
            annualization_factor=int(experiment.config["portfolio"]["annualization_factor"]),
        )
        for frame in [train_augmented, validation_augmented, test_augmented]:
            add_aggressive_trade_filter_columns(
                frame,
                best_regime_id=decision.best_regime_id,
                min_regime_prob=min_regime_prob,
            )
            generated = add_regime_features(frame)
            if regime_feature_columns is None:
                regime_feature_columns = generated
        augmented_splits.append(
            type(split)(
                train=train_augmented,
                validation=validation_augmented,
                test=test_augmented,
                split_id=split.split_id,
            )
        )

    if regime_feature_columns is None:
        raise ValueError("Regime feature generation failed during audit augmentation.")

    feature_set = type(experiment.feature_set)(
        frame=experiment.feature_set.frame,
        feature_columns=experiment.feature_set.feature_columns + regime_feature_columns,
        stationarity_summary=experiment.feature_set.stationarity_summary,
    )
    config = dict(experiment.config)
    config["regime"] = dict(experiment.config["regime"])
    config["regime"]["enabled"] = True
    return PreparedExperiment(
        config=config,
        results_dir=experiment.results_dir,
        feature_set=feature_set,
        splits=augmented_splits,
    )


def _load_single_asset_market_data(config: dict[str, Any]) -> pd.DataFrame:
    """Reload the primary single-asset market-data frame."""

    data_config = config["data"]
    return load_market_data(
        ticker=data_config["ticker"],
        benchmark_ticker=data_config["benchmark_ticker"],
        start_date=data_config["start_date"],
        end_date=data_config.get("end_date"),
        source=data_config.get("source", "yfinance"),
        cache_path=data_config.get("cache_path"),
        vix_ticker=data_config.get("vix_ticker"),
    )


def _write_report(
    path: Path,
    *,
    checks: list[AuditCheck],
    experiment: PreparedExperiment,
    comparative_results: pd.DataFrame | None,
) -> None:
    """Write the integrity audit as a markdown report."""

    passed = sum(1 for check in checks if check.status == "PASS")
    failed = sum(1 for check in checks if check.status == "FAIL")
    info = sum(1 for check in checks if check.status == "INFO")
    lines = [
        "# Integrity Audit Report",
        "",
        f"- Generated on: 2026-05-08",
        f"- Profile: `{experiment.config['project'].get('run_profile', 'full')}`",
        f"- Audit split budget: `{len(experiment.splits)}` walk-forward splits",
        f"- Champion under audit: `regime_stacking_ensemble_regime`",
        "",
        "## Summary",
        "",
        f"- `PASS`: {passed}",
        f"- `FAIL`: {failed}",
        f"- `INFO`: {info}",
        "",
        "## Checks",
        "",
    ]
    for check in checks:
        lines.append(f"### {check.name}")
        lines.append("")
        lines.append(f"- Status: `{check.status}`")
        lines.append(f"- Detail: {check.detail}")
        if check.evidence:
            lines.append("- Evidence:")
            for key, value in check.evidence.items():
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")
    if comparative_results is not None and not comparative_results.empty:
        lines.extend(
            [
                "## Comparative Null Test",
                "",
                "### Benchmark-Relative Decision Table",
                "",
                "| Model | Type | Canonical information_ratio | Mean shuffled information_ratio | 95th percentile shuffled information_ratio | Canonical excess net_sharpe | Mean shuffled excess net_sharpe | 95th percentile shuffled excess net_sharpe | p-value | Decision |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in comparative_results.iterrows():
            lines.append(
                "| "
                f"{row['model']} | {row['model_type']} | {row['canonical_information_ratio']:.6f} | "
                f"{row['mean_shuffled_information_ratio']:.6f} | {row['percentile_95_shuffled_information_ratio']:.6f} | "
                f"{row['canonical_excess_net_sharpe']:.6f} | {row['mean_shuffled_excess_net_sharpe']:.6f} | "
                f"{row['percentile_95_shuffled_excess_net_sharpe']:.6f} | "
                f"{row['p_value']:.6f} | {row['decision']} |"
            )
        lines.extend(
            [
                "",
                "### Raw net_sharpe Reference Table",
                "",
                "| Model | Canonical net_sharpe | Mean shuffled net_sharpe | 95th percentile shuffled net_sharpe |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in comparative_results.iterrows():
            lines.append(
                "| "
                f"{row['model']} | {row['canonical_net_sharpe']:.6f} | {row['mean_shuffled_net_sharpe']:.6f} | "
                f"{row['percentile_95_shuffled_net_sharpe']:.6f} |"
            )
        lines.extend(
            [
                "",
                "### Exposure Diagnostics",
                "",
                "| Model | Average long exposure | Average position size | Fraction positive predictions |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in comparative_results.iterrows():
            lines.append(
                "| "
                f"{row['model']} | {row['canonical_average_long_exposure']:.6f} | "
                f"{row['canonical_average_position_size']:.6f} | {row['canonical_fraction_positive_predictions']:.6f} |"
            )
        lines.extend(
            [
                "",
                "### Matched Null Decision Table",
                "",
                "Benchmark-relative matched-null diagnostics use executed positions and active `information_ratio` as the primary decision metric.",
                "",
                "| Model | Canonical information_ratio | Fraction in market | Daily turnover | Same exposure p-value | Same turnover p-value | Same exposure+turnover p-value | Same regime exposure p-value | Block bootstrap exposure p-value |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in comparative_results.iterrows():
            lines.append(
                "| "
                f"{row['model']} | {row['matched_null_canonical_information_ratio']:.6f} | "
                f"{row['matched_null_canonical_fraction_in_market']:.6f} | {row['matched_null_canonical_daily_turnover']:.6f} | "
                f"{row.get('matched_null__same_average_exposure_random__p_value', np.nan):.6f} | "
                f"{row.get('matched_null__same_turnover_random__p_value', np.nan):.6f} | "
                f"{row.get('matched_null__same_exposure_and_turnover_random__p_value', np.nan):.6f} | "
                f"{row.get('matched_null__same_regime_exposure_random__p_value', np.nan):.6f} | "
                f"{row.get('matched_null__block_bootstrap_same_exposure_random__p_value', np.nan):.6f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_comparative_null_report(path: Path, comparative_results: pd.DataFrame) -> None:
    """Write a standalone comparative null-test markdown report."""

    lines = [
        "# Comparative Null Test Report",
        "",
        "- Generated on: 2026-05-08",
        "- Purpose: compare canonical audit-sample performance against Monte Carlo label-shuffle nulls using benchmark-relative skill metrics.",
        "",
        "## Benchmark-Relative Decision Table",
        "",
        "| Model | Type | Runs | Canonical information_ratio | Mean shuffled information_ratio | 95th percentile shuffled information_ratio | Canonical excess net_sharpe | Mean shuffled excess net_sharpe | 95th percentile shuffled excess net_sharpe | p-value | Threshold | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in comparative_results.iterrows():
        lines.append(
            "| "
            f"{row['model']} | {row['model_type']} | {int(row['runs'])} | "
            f"{row['canonical_information_ratio']:.6f} | {row['mean_shuffled_information_ratio']:.6f} | "
            f"{row['percentile_95_shuffled_information_ratio']:.6f} | {row['canonical_excess_net_sharpe']:.6f} | "
            f"{row['mean_shuffled_excess_net_sharpe']:.6f} | {row['percentile_95_shuffled_excess_net_sharpe']:.6f} | "
            f"{row['p_value']:.6f} | {row['significance_threshold']:.6f} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Raw net_sharpe Reference Table",
            "",
            "| Model | Canonical net_sharpe | Canonical net_sortino | Canonical calmar | Mean shuffled net_sharpe | 95th percentile shuffled net_sharpe |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparative_results.iterrows():
        lines.append(
            "| "
            f"{row['model']} | {row['canonical_net_sharpe']:.6f} | {row['canonical_net_sortino']:.6f} | {row['canonical_calmar']:.6f} | "
            f"{row['mean_shuffled_net_sharpe']:.6f} | {row['percentile_95_shuffled_net_sharpe']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Exposure Diagnostics",
            "",
            "| Model | Average long exposure | Average position size | Fraction positive predictions | Decision metric |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in comparative_results.iterrows():
        lines.append(
            "| "
            f"{row['model']} | {row['canonical_average_long_exposure']:.6f} | {row['canonical_average_position_size']:.6f} | "
            f"{row['canonical_fraction_positive_predictions']:.6f} | {row['decision_metric']} |"
        )
    lines.extend(
        [
            "",
            "## Matched Null Decision Table",
            "",
            "Matched-null diagnostics treat canonical paths as executed positions, so exposure and turnover are preserved directly rather than reconstructed from raw predictions.",
            "",
            "| Model | Runs | Canonical information_ratio | Fraction in market | Daily turnover | Position flips | Same exposure p-value | Same turnover p-value | Same exposure+turnover p-value | Same regime exposure p-value | Block bootstrap exposure p-value |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in comparative_results.iterrows():
        lines.append(
            "| "
            f"{row['model']} | {int(row['matched_null_runs'])} | {row['matched_null_canonical_information_ratio']:.6f} | "
            f"{row['matched_null_canonical_fraction_in_market']:.6f} | {row['matched_null_canonical_daily_turnover']:.6f} | "
            f"{int(row['matched_null_canonical_position_flip_count'])} | "
            f"{row.get('matched_null__same_average_exposure_random__p_value', np.nan):.6f} | "
            f"{row.get('matched_null__same_turnover_random__p_value', np.nan):.6f} | "
            f"{row.get('matched_null__same_exposure_and_turnover_random__p_value', np.nan):.6f} | "
            f"{row.get('matched_null__same_regime_exposure_random__p_value', np.nan):.6f} | "
            f"{row.get('matched_null__block_bootstrap_same_exposure_random__p_value', np.nan):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Matched Null Decisions",
            "",
            "| Model | Same exposure | Same turnover | Same exposure+turnover | Same regime exposure | Block bootstrap exposure |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in comparative_results.iterrows():
        lines.append(
            "| "
            f"{row['model']} | "
            f"{row.get('matched_null__same_average_exposure_random__decision', 'N/A')} | "
            f"{row.get('matched_null__same_turnover_random__decision', 'N/A')} | "
            f"{row.get('matched_null__same_exposure_and_turnover_random__decision', 'N/A')} | "
            f"{row.get('matched_null__same_regime_exposure_random__decision', 'N/A')} | "
            f"{row.get('matched_null__block_bootstrap_same_exposure_random__decision', 'N/A')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
