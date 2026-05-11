"""Shared experiment utilities for baseline, tuning, and ensemble workflows."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import logging
from pathlib import Path
import time
from typing import Any
import warnings

import pandas as pd
from matplotlib import pyplot as plt

from data.market_data import load_market_data, summarize_market_frame
from evaluation.audit_artifacts import build_standard_audit_artifact_frame
from evaluation.metrics import compute_equity_curve, compute_strategy_returns, compute_trading_metrics
from evaluation.walk_forward import WalkForwardSplit, generate_walk_forward_splits
from features.engineering import FeatureSet, build_feature_set
from models.base import BaseSignalModel
from models.registry import build_default_registry
from utils.config import load_config
from utils.reproducibility import seed_everything


@dataclass(slots=True)
class PreparedExperiment:
    """Prepared dataset, splits, and runtime configuration."""

    config: dict[str, Any]
    results_dir: Path
    feature_set: FeatureSet
    splits: list[WalkForwardSplit]


@dataclass(slots=True)
class ModelArtifacts:
    """Artifacts produced from one walk-forward model run."""

    comparison_row: dict[str, float | str]
    per_split_results: list[dict[str, float]]
    oof_predictions: pd.DataFrame
    aggregated_curve: pd.DataFrame


def apply_runtime_mode(config: dict[str, Any], profile_override: str | None = None) -> dict[str, Any]:
    """Apply profile-based runtime overrides."""

    config = deepcopy(config)
    if profile_override:
        config["project"]["run_profile"] = profile_override

    runtime = config.get("runtime", {})
    profile = str(config["project"].get("run_profile", "")).lower()
    if profile == "fast":
        selected_profile = runtime.get("fast_mode", {})
        config["project"]["fast_mode"] = True
    elif profile == "full":
        selected_profile = runtime.get("full_mode", {})
        config["project"]["fast_mode"] = False
    elif config["project"].get("fast_mode", False):
        selected_profile = runtime.get("fast_mode", {})
    else:
        return config

    validation_overrides = selected_profile.get("validation", {})
    feature_overrides = selected_profile.get("features", {})
    model_overrides = selected_profile.get("models", {})

    config["validation"] = {**config["validation"], **validation_overrides}
    config["features"] = {**config["features"], **feature_overrides}
    config["models"] = {**config["models"], **{key: value for key, value in model_overrides.items() if key != "overrides"}}
    for model_name, overrides in model_overrides.get("overrides", {}).items():
        if model_name in config["models"]:
            config["models"][model_name] = {**config["models"][model_name], **overrides}
    return config


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested dictionary without mutating the caller input."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def configure_runtime_noise(suppress_lightning_warnings: bool) -> None:
    """Reduce repetitive Lightning and PyTorch Forecasting warnings in CLI runs."""

    if not suppress_lightning_warnings:
        return

    warnings.filterwarnings("ignore", message=r".*Attribute 'loss' is an instance of `nn.Module`.*")
    warnings.filterwarnings("ignore", message=r".*Attribute 'logging_metrics' is an instance of `nn.Module`.*")
    warnings.filterwarnings("ignore", message=r".*The '.*_dataloader' does not have many workers.*")
    warnings.filterwarnings("ignore", message=r".*Checkpoint directory .* exists and is not empty.*")
    warnings.filterwarnings("ignore", message=r".*LeafSpec.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*enable_nested_tensor is True.*")

    for logger_name in [
        "lightning",
        "lightning.pytorch",
        "lightning.fabric",
        "pytorch_lightning",
        "pytorch_forecasting",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def prepare_experiment(
    config_path: str | Path,
    profile_override: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> PreparedExperiment:
    """Load config, data, feature set, and walk-forward splits."""

    config = load_config(config_path)
    config = apply_runtime_mode(config, profile_override=profile_override)
    if config_overrides:
        config = _deep_update(config, deepcopy(config_overrides))
    configure_runtime_noise(config["project"].get("suppress_lightning_warnings", True))
    seed_everything(config["project"]["seed"])

    results_dir = Path(config["project"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "equity_curves").mkdir(parents=True, exist_ok=True)

    single_asset_data_config = {
        key: value
        for key, value in config["data"].items()
        if key in {"ticker", "benchmark_ticker", "start_date", "end_date", "source", "cache_path", "vix_ticker"}
    }
    market_data = load_market_data(**single_asset_data_config)
    return prepare_experiment_from_market_data(config=config, market_data=market_data)


def prepare_experiment_from_market_data(config: dict[str, Any], market_data: pd.DataFrame) -> PreparedExperiment:
    """Prepare a walk-forward experiment from an already-loaded market-data frame."""

    results_dir = Path(config["project"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "equity_curves").mkdir(parents=True, exist_ok=True)

    print("Loaded market data:", summarize_market_frame(market_data))

    feature_set = build_feature_set(
        market_data=market_data,
        target_horizon=config["features"]["target_horizon"],
        short_window=config["features"]["rolling_windows"]["short"],
        medium_window=config["features"]["rolling_windows"]["medium"],
        adf_significance=config["features"]["adf_significance"],
        dropna=config["features"]["dropna"],
        advanced_features=config["features"].get("advanced_features", False),
        vix_features=config["features"].get("vix_features", False),
    )

    print("\nADF stationarity summary:")
    print(feature_set.stationarity_summary.to_string(index=False))

    splits = generate_walk_forward_splits(
        frame=feature_set.frame,
        train_size=config["validation"]["train_size"],
        val_size=config["validation"]["val_size"],
        test_size=config["validation"]["test_size"],
        step_size=config["validation"]["step_size"],
    )

    config["models"]["_device_preference"] = config["project"].get("torch_device", "auto")
    config["models"]["_suppress_lightning_output"] = config["project"].get("suppress_lightning_output", True)
    return PreparedExperiment(config=config, results_dir=results_dir, feature_set=feature_set, splits=splits)


def get_enabled_models(config: dict[str, Any]) -> list[str]:
    """Return the enabled model names after applying disabled filters."""

    disabled_models = set(config["models"].get("disabled", []))
    return [name for name in config["models"]["enabled"] if name not in disabled_models]


def build_registry(config: dict[str, Any], results_dir: Path):
    """Build the model registry from runtime configuration."""

    return build_default_registry(config["models"], results_dir=results_dir)


def build_trade_filter_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return normalized regime trade-filter settings when enabled."""

    regime_config = config.get("regime", {})
    if not regime_config.get("enabled", False) or not regime_config.get("trade_filter", False):
        return None
    return {
        "filter_mode": str(regime_config.get("filter_mode", "allowlist")).lower(),
        "min_regime_prob": float(regime_config.get("min_regime_prob", 0.0)),
        "favorable_regimes": [int(value) for value in regime_config.get("favorable_regimes", [])],
    }


def evaluate_model(
    model: BaseSignalModel,
    label: str,
    feature_columns: list[str],
    splits: list[WalkForwardSplit],
    portfolio_config: dict[str, Any],
    results_dir: Path | None = None,
    artifact_prefix: str | None = None,
    log_progress: bool = False,
    progress_position: tuple[int, int] | None = None,
    trade_filter: dict[str, Any] | None = None,
) -> ModelArtifacts:
    """Run a model across walk-forward splits and optionally persist artifacts."""

    pipeline_start_time = time.perf_counter()
    per_split_results: list[dict[str, float]] = []
    curve_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    total_splits = len(splits)

    for split_position, split in enumerate(splits, start=1):
        split_start_time = time.perf_counter()
        if log_progress:
            header = f"  Split {split_position}/{total_splits}"
            if progress_position is not None:
                header = f"  Split {split_position}/{total_splits}"
            print(f"{header} (train={len(split.train)}, val={len(split.validation)}, test={len(split.test)})")

        model.fit(
            train_frame=split.train,
            val_frame=split.validation,
            feature_columns=feature_columns,
            target_column="target_direction",
        )
        prediction = model.predict(frame=split.test, feature_columns=feature_columns)
        prediction = _apply_trade_filter_to_model_prediction(split.test, prediction, trade_filter)
        result = model.evaluate(
            frame=split.test,
            feature_columns=feature_columns,
            target_column="target_direction",
            return_column="forward_simple_return_1d",
            benchmark_column="benchmark_return_1d",
            annualization_factor=portfolio_config["annualization_factor"],
            transaction_cost_bps=portfolio_config["transaction_cost_bps"],
            prediction=prediction,
        )
        result["model"] = label
        per_split_results.append(result)

        strategy_returns = compute_strategy_returns(
            returns=split.test["forward_simple_return_1d"],
            signal=prediction.predictions,
            transaction_cost_bps=portfolio_config["transaction_cost_bps"],
        )
        curve_frames.append(
            pd.DataFrame(
                {
                    "date": split.test.index,
                    "split_id": split.split_id,
                    "strategy_return": strategy_returns.values,
                    "equity_curve": compute_equity_curve(strategy_returns).values,
                }
            )
        )
        prediction_frames.append(
            build_standard_audit_artifact_frame(
                frame=split.test,
                label=label,
                prediction=prediction.predictions,
                probability=prediction.probabilities,
                split_id=split.split_id,
                transaction_cost_bps=float(portfolio_config["transaction_cost_bps"]),
            )
        )

        if log_progress:
            split_elapsed = time.perf_counter() - split_start_time
            avg_split_time = (time.perf_counter() - pipeline_start_time) / split_position
            eta_seconds = max((total_splits - split_position) * avg_split_time, 0.0)
            print(f"    Completed in {split_elapsed:.1f}s | split ETA {eta_seconds / 60.0:.1f} min")

    comparison_frame = pd.DataFrame(per_split_results)
    comparison_row = comparison_frame.mean(numeric_only=True).to_dict()
    comparison_row["model"] = label

    combined_curves = pd.concat(curve_frames, ignore_index=True)
    aggregated_curve = combined_curves.groupby("date", as_index=False)["strategy_return"].mean().sort_values("date")
    aggregated_curve["equity_curve"] = compute_equity_curve(
        aggregated_curve["strategy_return"].rename("strategy_return")
    ).values
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)

    if results_dir is not None and artifact_prefix is not None:
        (results_dir / "equity_curves").mkdir(parents=True, exist_ok=True)
        combined_curves.to_csv(results_dir / "equity_curves" / f"{artifact_prefix}_per_split.csv", index=False)
        aggregated_curve.to_csv(results_dir / "equity_curves" / f"{artifact_prefix}_aggregated.csv", index=False)
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(aggregated_curve["date"], aggregated_curve["equity_curve"], linewidth=1.5)
        axis.set_title(f"{label} aggregated equity curve")
        axis.set_xlabel("Date")
        axis.set_ylabel("Equity")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(results_dir / "equity_curves" / f"{artifact_prefix}_aggregated.png", dpi=150)
        plt.close(figure)

    return ModelArtifacts(
        comparison_row=comparison_row,
        per_split_results=per_split_results,
        oof_predictions=oof_predictions,
        aggregated_curve=aggregated_curve,
    )


def _apply_trade_filter_to_model_prediction(
    frame: pd.DataFrame,
    prediction: Any,
    trade_filter: dict[str, Any] | None,
):
    """Apply regime-based trade filtering to a model prediction container."""

    if trade_filter is None or "regime_id" not in frame.columns:
        return prediction

    allowed_mask = _build_trade_allowed_mask(frame=frame, trade_filter=trade_filter)
    predictions = prediction.predictions.mask(~allowed_mask, other=0)
    return type(prediction)(
        probabilities=prediction.probabilities,
        predictions=predictions,
        name=getattr(prediction, "name", None),
    )


def _build_trade_allowed_mask(frame: pd.DataFrame, trade_filter: dict[str, Any]) -> pd.Series:
    """Build a boolean trade-allowed mask from configured regime filter settings."""

    mode = str(trade_filter.get("filter_mode", "allowlist")).lower()
    if mode == "aggressive" and "trade_allowed_aggressive" in frame.columns:
        return frame["trade_allowed_aggressive"].astype(bool)

    allowed_regimes = {int(value) for value in trade_filter.get("favorable_regimes", [])}
    if not allowed_regimes:
        return pd.Series(True, index=frame.index)
    return frame["regime_id"].astype(int).isin(allowed_regimes)


def build_benchmark_row(
    splits: list[WalkForwardSplit],
    portfolio_config: dict[str, Any],
    results_dir: Path | None = None,
    label: str = "buy_and_hold_spy",
) -> dict[str, float | str]:
    """Construct the buy-and-hold benchmark row and optional artifacts."""

    benchmark_results: list[dict[str, float | str]] = []
    benchmark_curves: list[pd.DataFrame] = []
    for split in splits:
        benchmark_signal = pd.Series(1, index=split.test.index, name="benchmark_signal")
        benchmark_returns = compute_strategy_returns(
            returns=split.test["forward_simple_return_1d"],
            signal=benchmark_signal,
            transaction_cost_bps=0.0,
        )
        benchmark_results.append(
            {
                "model": label,
                "directional_accuracy": float("nan"),
                "auc_roc": float("nan"),
                **compute_trading_metrics(
                    returns=split.test["forward_simple_return_1d"],
                    benchmark_returns=split.test["benchmark_return_1d"],
                    signal=benchmark_signal,
                    annualization_factor=portfolio_config["annualization_factor"],
                    transaction_cost_bps=0.0,
                ),
            }
        )
        benchmark_curves.append(
            pd.DataFrame(
                {
                    "date": split.test.index,
                    "split_id": split.split_id,
                    "strategy_return": benchmark_returns.values,
                    "equity_curve": compute_equity_curve(benchmark_returns).values,
                }
            )
        )

    benchmark_row = pd.DataFrame(benchmark_results).mean(numeric_only=True).to_dict()
    benchmark_row["model"] = label

    if results_dir is not None:
        (results_dir / "equity_curves").mkdir(parents=True, exist_ok=True)
        combined = pd.concat(benchmark_curves, ignore_index=True)
        combined.to_csv(results_dir / "equity_curves" / f"{label}_per_split.csv", index=False)
        aggregated = combined.groupby("date", as_index=False)["strategy_return"].mean().sort_values("date")
        aggregated["equity_curve"] = compute_equity_curve(aggregated["strategy_return"].rename("strategy_return")).values
        aggregated.to_csv(results_dir / "equity_curves" / f"{label}_aggregated.csv", index=False)
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(aggregated["date"], aggregated["equity_curve"], linewidth=1.5)
        axis.set_title(f"{label} aggregated equity curve")
        axis.set_xlabel("Date")
        axis.set_ylabel("Equity")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(results_dir / "equity_curves" / f"{label}_aggregated.png", dpi=150)
        plt.close(figure)

    return benchmark_row
