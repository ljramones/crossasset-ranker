"""Optuna-based hyperparameter tuning for selected sequence models."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from utils.experiment import PreparedExperiment, build_registry, build_trade_filter_config, evaluate_model


def run_optuna_tuning(experiment: PreparedExperiment) -> tuple[list[dict[str, float | str]], dict[str, dict[str, Any]]]:
    """Tune the configured models and return tuned comparison rows plus best params."""

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("optuna is required for `--tune`. Add it to the project dependencies.") from exc

    tuning_config = experiment.config["tuning"]
    results_dir = Path(tuning_config["results_dir"])
    best_params_dir = Path(tuning_config["best_params_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    best_params_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    best_params_by_model: dict[str, dict[str, Any]] = {}
    for model_name in tuning_config["enabled_models"]:
        study = optuna.create_study(direction="maximize", study_name=f"{model_name}_sharpe")
        study.optimize(
            lambda trial: _objective(trial, experiment=experiment, model_name=model_name),
            n_trials=tuning_config["n_trials"],
            show_progress_bar=False,
        )

        best_params = dict(study.best_trial.params)
        best_params_by_model[model_name] = best_params
        (best_params_dir / f"{model_name}.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "model": model_name,
                "best_value": float(study.best_value),
                "best_sortino": float(study.best_trial.user_attrs.get("sortino", 0.0)),
                "best_calmar": float(study.best_trial.user_attrs.get("calmar", 0.0)),
                "n_trials": int(len(study.trials)),
            }
        )

    summary_path = Path(tuning_config["study_summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    tuned_rows = evaluate_tuned_models(experiment=experiment, best_params_by_model=best_params_by_model)
    return tuned_rows, best_params_by_model


def evaluate_tuned_models(
    experiment: PreparedExperiment,
    best_params_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, float | str]]:
    """Evaluate tuned models on the full walk-forward comparison."""

    tuned_rows: list[dict[str, float | str]] = []
    optimization_dir = Path(experiment.config["tuning"]["results_dir"])
    for model_name, params in best_params_by_model.items():
        tuned_config = _build_tuned_model_config(experiment.config, model_name, params)
        registry = build_registry(tuned_config, experiment.results_dir)
        model = registry.get_model(model_name)
        label = f"{model.get_display_name()}_tuned"
        artifacts = evaluate_model(
            model=model,
            label=label,
            feature_columns=experiment.feature_set.feature_columns,
            splits=experiment.splits,
            portfolio_config=experiment.config["portfolio"],
            results_dir=optimization_dir,
            artifact_prefix=label,
            log_progress=False,
            trade_filter=build_trade_filter_config(experiment.config),
        )
        tuned_rows.append(artifacts.comparison_row)
    return tuned_rows


def _objective(trial, experiment: PreparedExperiment, model_name: str) -> float:
    """Run one tuning trial and return the Sharpe ratio objective."""

    params = _suggest_params(trial, model_name=model_name)
    tuned_config = _build_tuned_model_config(experiment.config, model_name, params)
    registry = build_registry(tuned_config, experiment.results_dir)
    model = registry.get_model(model_name)
    label = f"{model.get_display_name()}_trial"
    artifacts = evaluate_model(
        model=model,
        label=label,
        feature_columns=experiment.feature_set.feature_columns,
        splits=experiment.splits,
        portfolio_config=experiment.config["portfolio"],
        log_progress=False,
        trade_filter=build_trade_filter_config(experiment.config),
    )
    trial.set_user_attr("sortino", float(artifacts.comparison_row.get("sortino", 0.0)))
    trial.set_user_attr("net_sortino", float(artifacts.comparison_row.get("net_sortino", 0.0)))
    trial.set_user_attr("calmar", float(artifacts.comparison_row.get("calmar", 0.0)))
    return float(artifacts.comparison_row["net_sharpe"])


def _build_tuned_model_config(config: dict[str, Any], model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Merge tuned params into the runtime config."""

    tuned_config = deepcopy(config)
    tuned_config["models"][model_name] = {
        **tuned_config["models"][model_name],
        **_normalize_tuned_params(model_name=model_name, params=params),
    }
    return tuned_config


def _suggest_params(trial, model_name: str) -> dict[str, Any]:
    """Define search spaces for the tuned models."""

    if model_name == "lstm":
        return {
            "input_size": trial.suggest_categorical("input_size", [20, 30, 40]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 96]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "max_epochs": trial.suggest_int("max_epochs", 6, 16),
        }
    if model_name == "itransformer":
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 96])
        n_heads = hidden_size // 16
        return {
            "input_size": trial.suggest_categorical("input_size", [20, 30, 40]),
            "hidden_size": hidden_size,
            "n_heads": n_heads,
            "feedforward_size": trial.suggest_categorical("feedforward_size", [64, 128, 192]),
            "encoder_layers": trial.suggest_categorical("encoder_layers", [1, 2, 3]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "max_epochs": trial.suggest_int("max_epochs", 6, 12),
        }
    if model_name == "patchtst":
        patch_len = trial.suggest_categorical("patch_len", [5, 10, 15])
        return {
            "input_size": trial.suggest_categorical("input_size", [20, 30, 40]),
            "patch_len": patch_len,
            "stride": trial.suggest_categorical("stride", [5, 10]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 96]),
            "feedforward_size": trial.suggest_categorical("feedforward_size", [64, 128, 192]),
            "encoder_layers": trial.suggest_categorical("encoder_layers", [1, 2, 3]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "max_epochs": trial.suggest_int("max_epochs", 6, 12),
        }
    raise ValueError(f"Unsupported tuning model: {model_name}")


def _normalize_tuned_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Remove search-only fields and map them to concrete model config values."""

    normalized = dict(params)
    if model_name == "itransformer":
        head_config = normalized.pop("head_config", None)
        if head_config is not None:
            hidden_size, n_heads = head_config
            normalized.setdefault("hidden_size", hidden_size)
            normalized.setdefault("n_heads", n_heads)
        if "hidden_size" in normalized and "n_heads" not in normalized:
            normalized["n_heads"] = int(normalized["hidden_size"]) // 16
    return normalized
