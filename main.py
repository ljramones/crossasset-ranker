"""CLI entry point for baseline, tuning, and ensemble experiment workflows."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from audit.integrity_audit import run_integrity_audit
from data.market_data import load_market_data_bundle
from evaluation.audit_artifacts import build_standard_audit_artifact_frame
from evaluation.metrics import compute_equity_curve, compute_strategy_returns, compute_trading_metrics
from features.regime_features import add_regime_features
from models.ensemble import (
    EnsemblePrediction,
    RegimeWeightedEnsemble,
    RegimeStackingEnsemble,
    StackingEnsemble,
    WeightedAverageEnsemble,
    evaluate_ensemble_predictions,
)
from optimization.optuna_tuner import run_optuna_tuning
from portfolio.asset_selector import select_eligible_assets
from portfolio.portfolio_builder import build_selective_portfolios, load_asset_signal_panel
from regime.regime_detection import (
    MarketRegimeDetector,
    RegimeDetectionConfig,
    add_aggressive_trade_filter_columns,
)
from utils.experiment import (
    ModelArtifacts,
    PreparedExperiment,
    apply_runtime_mode,
    build_benchmark_row,
    build_registry,
    build_trade_filter_config,
    configure_runtime_noise,
    evaluate_model,
    get_enabled_models,
    prepare_experiment,
    prepare_experiment_from_market_data,
)
from utils.config import load_config
from utils.reproducibility import seed_everything
from utils.reporting import get_primary_ranking_label, summarize_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment control."""

    parser = argparse.ArgumentParser(description="Run trading signal model comparisons.")
    parser.add_argument("--profile", choices=["fast", "full"], default=None, help="Override run profile from config.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning workflow.")
    parser.add_argument("--ensemble", action="store_true", help="Run ensemble workflow.")
    parser.add_argument("--multi-asset", action="store_true", help="Run the full multi-asset robustness workflow.")
    parser.add_argument("--regime", action="store_true", help="Run baseline vs regime-aware comparison.")
    parser.add_argument("--selective-portfolio", action="store_true", help="Run selective portfolio construction on multi-asset outputs.")
    parser.add_argument("--audit", action="store_true", help="Run the champion integrity audit and write a markdown report.")
    parser.add_argument("--comparative-null-test", action="store_true", help="When used with --audit, run the comparative Monte Carlo null test report.")
    parser.add_argument("--tune-trials", type=int, default=None, help="Override Optuna trial count for this run.")
    parser.add_argument("--splits", type=int, default=None, help="Limit the number of walk-forward splits used.")
    return parser.parse_args()


def main() -> None:
    """Dispatch baseline, tuning, or ensemble workflows."""

    args = parse_args()
    if args.audit:
        _run_audit_workflow(
            config_path=Path("config/config.yaml"),
            profile_override=args.profile or "full",
            comparative_null_test=args.comparative_null_test,
        )
        return
    if args.multi_asset:
        _run_multi_asset_workflow(
            config_path=Path("config/config.yaml"),
            profile_override=args.profile or "full",
            tune_trials=args.tune_trials,
            max_splits=args.splits,
            regime_enabled=args.regime,
            selective_portfolio=args.selective_portfolio,
        )
        return
    if args.regime:
        _run_regime_workflow(
            config_path=Path("config/config.yaml"),
            profile_override=args.profile or "full",
            tune_trials=args.tune_trials,
            max_splits=args.splits,
        )
        return
    if args.profile == "full" and not args.tune and not args.ensemble:
        _run_vix_impact_workflow(
            config_path=Path("config/config.yaml"),
            tune_trials=args.tune_trials,
            max_splits=args.splits,
        )
        return

    experiment = prepare_experiment(
        Path("config/config.yaml"),
        profile_override=args.profile,
        config_overrides=_build_cli_overrides(tune_trials=args.tune_trials),
    )
    experiment = _limit_experiment_splits(experiment, max_splits=args.splits)
    _print_runtime_configuration(experiment)

    if args.tune:
        _run_tuning_workflow(experiment)
        return

    if args.ensemble:
        _run_ensemble_workflow(experiment)
        return

    comparison = _run_baseline_suite(
        experiment=experiment,
        output_path=experiment.results_dir / "model_comparison.csv",
        results_dir=experiment.results_dir,
    )
    _print_comparison(comparison)


def _run_audit_workflow(config_path: Path, profile_override: str, comparative_null_test: bool) -> None:
    """Run the frozen-champion integrity audit and print a compact result summary."""

    checks = run_integrity_audit(
        config_path=config_path,
        profile_override=profile_override,
        comparative_null_test=comparative_null_test,
    )
    report_path = Path("results/integrity_audit_report.md")
    manifest_path = Path("champions/champion_v1.0_manifest.yaml")
    comparative_path = Path("results/comparative_null_test_report.md")
    passed = sum(1 for check in checks if check.status == "PASS")
    failed = sum(1 for check in checks if check.status == "FAIL")
    info = sum(1 for check in checks if check.status == "INFO")
    print(f"\nSaved integrity audit report to: {report_path}")
    if comparative_null_test:
        print(f"Saved comparative null test report to: {comparative_path}")
    print("\nIntegrity Audit Summary")
    print({"pass": passed, "fail": failed, "info": info})
    for check in checks:
        print(f"- [{check.status}] {check.name}: {check.detail}")
    print(f"\nChampion Frozen: {manifest_path}")


def _run_vix_impact_workflow(
    config_path: Path,
    tune_trials: int | None = None,
    max_splits: int | None = None,
) -> None:
    """Run paired full-profile experiments with and without VIX features."""

    without_vix = prepare_experiment(
        config_path,
        profile_override="full",
        config_overrides=_merge_cli_overrides(
            {"features": {"vix_features": False}},
            _build_cli_overrides(tune_trials=tune_trials),
        ),
    )
    without_vix = _limit_experiment_splits(without_vix, max_splits=max_splits)
    _print_runtime_configuration(without_vix)
    without_vix_comparison = _run_full_cost_aware_suite(
        experiment=without_vix,
        final_output_path=without_vix.results_dir / "model_comparison_without_vix.csv",
        save_advanced_feature_comparison=False,
        save_cost_comparison=False,
    )

    with_vix = prepare_experiment(
        config_path,
        profile_override="full",
        config_overrides=_merge_cli_overrides(
            {"features": {"vix_features": True}},
            _build_cli_overrides(tune_trials=tune_trials),
        ),
    )
    with_vix = _limit_experiment_splits(with_vix, max_splits=max_splits)
    _print_runtime_configuration(with_vix)
    with_vix_comparison = _run_full_cost_aware_suite(
        experiment=with_vix,
        final_output_path=with_vix.results_dir / "model_comparison_with_vix.csv",
        save_advanced_feature_comparison=False,
        save_cost_comparison=True,
    )
    _print_vix_impact_summary(with_vix_comparison=with_vix_comparison, without_vix_comparison=without_vix_comparison)


def _run_regime_workflow(
    config_path: Path,
    profile_override: str,
    tune_trials: int | None = None,
    max_splits: int | None = None,
) -> None:
    """Compare baseline, regime-aware, and regime-weighted ensemble evaluation."""

    baseline_experiment = prepare_experiment(
        config_path,
        profile_override=profile_override,
        config_overrides=_build_cli_overrides(tune_trials=tune_trials),
    )
    baseline_experiment = _limit_experiment_splits(baseline_experiment, max_splits=max_splits)
    _print_runtime_configuration(baseline_experiment)
    baseline_comparison = _run_full_cost_aware_suite(
        experiment=baseline_experiment,
        final_output_path=baseline_experiment.results_dir / "regime_baseline_reference.csv",
        save_advanced_feature_comparison=False,
        save_cost_comparison=False,
    )
    baseline_labeled = baseline_comparison.copy()
    baseline_labeled["comparison_mode"] = "baseline"
    baseline_labeled["model"] = baseline_labeled["model"].astype(str) + "_baseline"

    regime_base_experiment, timeline, transition_matrix = _build_regime_aware_experiment(baseline_experiment)
    regime_base_experiment = _with_regime_filter_state(regime_base_experiment, enabled=False)

    regime_no_interactions = _with_regime_stacking_interactions(regime_base_experiment, enabled=False)
    _print_runtime_configuration(regime_no_interactions)
    regime_no_interactions_comparison = _run_full_cost_aware_suite(
        experiment=regime_no_interactions,
        final_output_path=regime_no_interactions.results_dir / "regime_mode_reference_no_interactions.csv",
        save_advanced_feature_comparison=False,
        save_cost_comparison=False,
    )
    regime_labeled = regime_no_interactions_comparison.copy()
    regime_labeled["comparison_mode"] = "regime_features"
    regime_labeled["model"] = regime_labeled["model"].astype(str) + "_regime"

    regime_interactions = _with_regime_stacking_interactions(regime_base_experiment, enabled=True)
    _print_runtime_configuration(regime_interactions)
    regime_interactions_comparison = _run_full_cost_aware_suite(
        experiment=regime_interactions,
        final_output_path=regime_interactions.results_dir / "regime_mode_reference_interactions.csv",
        save_advanced_feature_comparison=False,
        save_cost_comparison=False,
    )
    interaction_labeled = regime_interactions_comparison.copy()
    interaction_labeled["comparison_mode"] = "regime_features_interactions"
    interaction_labeled["model"] = interaction_labeled["model"].astype(str) + "_regime_interactions"

    combined = summarize_results(
        baseline_labeled.to_dict("records")
        + regime_labeled.to_dict("records")
        + interaction_labeled.to_dict("records")
    )
    comparison_path = regime_interactions.results_dir / "regime_stacking_comparison.csv"
    combined.to_csv(comparison_path, index=False)
    interaction_comparison_path = regime_interactions.results_dir / "regime_stacking_interactions_comparison.csv"
    _build_regime_interactions_comparison(
        baseline_comparison=baseline_comparison,
        no_interactions_comparison=regime_no_interactions_comparison,
        interactions_comparison=regime_interactions_comparison,
    ).to_csv(interaction_comparison_path, index=False)
    combined.to_csv(regime_interactions.results_dir / "regime_weighted_comparison.csv", index=False)
    combined.to_csv(regime_interactions.results_dir / "regime_comparison.csv", index=False)

    regime_summary = _build_regime_summary(
        baseline_comparison=baseline_comparison,
        regime_comparison=regime_no_interactions_comparison,
        interactions_comparison=regime_interactions_comparison,
        regime_experiment=regime_interactions,
    )
    summary_path = regime_interactions.results_dir / "regime_summary.csv"
    regime_summary.to_csv(summary_path, index=False)

    _save_regime_timeline_plot(timeline=timeline, output_path=regime_interactions.results_dir / "regime_timeline_spy.png")
    _save_heatmap(
        matrix=pd.DataFrame(
            transition_matrix,
            index=[f"regime_{index}" for index in range(transition_matrix.shape[0])],
            columns=[f"regime_{index}" for index in range(transition_matrix.shape[1])],
        ),
        output_path=regime_interactions.results_dir / "regime_transition_heatmap.png",
        title="Regime Transition Probability Matrix",
    )

    print(f"Saved regime comparison to: {comparison_path}")
    print(f"Saved interaction comparison to: {interaction_comparison_path}")
    print(f"Saved regime summary to: {summary_path}")
    _print_comparison(combined)
    _print_regime_weighted_summary(combined=combined, regime_summary=regime_summary)
    _print_regime_interactions_delta(regime_summary)
    _save_ensemble_bias_fix_comparison(combined=combined, results_dir=regime_interactions.results_dir)
    print("\nPer-regime summary:")
    print(regime_summary.to_string(index=False))


def _save_ensemble_bias_fix_comparison(combined: pd.DataFrame, results_dir: Path) -> None:
    """Save and print the focused ensemble bias-fix comparison."""

    focus_models = [
        "regime_stacking_ensemble_regime",
        "regime_stacking_ensemble_legacy_regime",
        "itransformer_tuned_regime",
        "stacking_ensemble_baseline",
    ]
    subset = combined.loc[combined["model"].isin(focus_models)].copy()
    if subset.empty:
        return
    columns = [
        "model",
        "information_ratio",
        "active_calmar",
        "annualized_active_return",
        "net_sharpe",
        "excess_net_sharpe",
        "net_sortino",
        "calmar",
        "fraction_in_market",
        "average_long_exposure",
        "fraction_positive_predictions",
        "daily_turnover",
        "annualized_turnover",
        "position_flip_count",
        "average_holding_period_days",
        "round_trip_count",
        "cost_drag",
        "cost_per_unit_active_return",
    ]
    ranking_columns = [
        column
        for column in ["information_ratio", "active_calmar", "annualized_active_return", "net_sharpe"]
        if column in subset.columns
    ]
    output = subset[[column for column in columns if column in subset.columns]].sort_values(
        ranking_columns,
        ascending=[False] * len(ranking_columns),
    ).reset_index(drop=True)
    output_path = results_dir / "ensemble_bias_fix_comparison.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved ensemble bias-fix comparison to: {output_path}")
    print("\nEnsemble Bias-Fix Summary")
    print(output.to_string(index=False))


def _run_multi_asset_workflow(
    config_path: Path,
    profile_override: str,
    tune_trials: int | None = None,
    max_splits: int | None = None,
    regime_enabled: bool = False,
    selective_portfolio: bool = False,
) -> None:
    """Run the multi-asset robustness workflow, optionally with regime-stacking stress tests."""

    config = load_config(config_path)
    config = apply_runtime_mode(config, profile_override=profile_override)
    cli_overrides = _build_cli_overrides(tune_trials=tune_trials)
    if cli_overrides:
        config = _merge_cli_overrides(config, cli_overrides)
    config["features"]["advanced_features"] = True
    config["features"]["vix_features"] = True
    if regime_enabled:
        config["regime"]["enabled"] = True
        config["regime"]["include_interactions"] = False
    configure_runtime_noise(config["project"].get("suppress_lightning_warnings", True))
    seed_everything(config["project"]["seed"])

    tickers = list(config["data"]["tickers"])
    cache_dir = "data/multi_asset_cache"
    bundle = load_market_data_bundle(
        tickers=tickers,
        benchmark_ticker=config["data"]["benchmark_ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        source=config["data"]["source"],
        cache_dir=cache_dir,
        vix_ticker=config["data"].get("vix_ticker"),
    )

    results_root = Path(config["project"]["results_dir"])
    results_root.mkdir(parents=True, exist_ok=True)

    multi_asset_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    transaction_cost_scenarios = list(config.get("stress_testing", {}).get("transaction_cost_scenarios", [config["portfolio"]["transaction_cost_bps"]]))
    returns_frame = pd.DataFrame(
        {ticker: frame["Adj Close"].pct_change().rename(ticker) for ticker, frame in bundle.items()}
    ).dropna(how="all")
    _save_heatmap(
        matrix=returns_frame.corr(),
        output_path=results_root / "multi_asset_correlation_heatmap.png",
        title="Asset Daily Returns Correlation",
    )

    for ticker in tickers:
        for cost_bps in transaction_cost_scenarios:
            asset_config = _build_multi_asset_config(
                config=config,
                ticker=ticker,
                results_root=results_root,
                cost_bps=cost_bps,
                regime_enabled=regime_enabled,
            )
            experiment = prepare_experiment_from_market_data(config=asset_config, market_data=bundle[ticker])
            experiment = _limit_experiment_splits(experiment, max_splits=max_splits)
            if regime_enabled:
                experiment, _, _ = _build_regime_aware_experiment(experiment)
                experiment = _with_regime_filter_state(experiment, enabled=False)
                experiment = _with_regime_stacking_interactions(experiment, enabled=False)
            _print_runtime_configuration(experiment)
            run_records.append(
                {
                    "ticker": ticker,
                    "cost_bps": float(cost_bps),
                    "results_dir": str(experiment.results_dir),
                }
            )
            benchmark_label = f"buy_and_hold_{ticker.lower().replace('-', '_')}"
            final_comparison = _run_full_cost_aware_suite(
                experiment=experiment,
                final_output_path=experiment.results_dir / "model_comparison.csv",
                save_advanced_feature_comparison=False,
                save_cost_comparison=False,
                benchmark_label=benchmark_label,
            )
            selected = _select_multi_asset_rows(
                final_comparison=final_comparison,
                ticker=ticker,
                benchmark_label=benchmark_label,
                cost_bps=cost_bps,
                regime_enabled=regime_enabled,
                champion_model=str(asset_config["regime"].get("default_model", "regime_stacking_ensemble")),
            )
            multi_asset_rows.extend(selected.to_dict("records"))

    comparison = pd.DataFrame(multi_asset_rows)
    if regime_enabled:
        comparison_path = results_root / "multi_asset_regime_stacking.csv"
        clean_comparison = comparison[
            ["ticker", "model", "cost_bps", "net_sharpe", "net_sortino", "calmar", "excess_vs_bh"]
        ].sort_values(["cost_bps", "ticker", "net_sharpe"], ascending=[True, True, False]).reset_index(drop=True)
        clean_comparison.to_csv(comparison_path, index=False)
        summary = _build_multi_asset_regime_stacking_summary(comparison)
        summary_path = results_root / "multi_asset_robustness_summary.csv"
        summary.to_csv(summary_path, index=False)
        _print_multi_asset_regime_stacking_report(comparison=clean_comparison, summary=summary)
        if selective_portfolio:
            _run_selective_portfolio_workflow(
                config=config,
                comparison=comparison,
                run_records=run_records,
                results_root=results_root,
            )
    else:
        comparison_path = results_root / "multi_asset_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        sharpe_matrix = comparison.pivot(index="ticker", columns="model", values="net_sharpe").sort_index()
        _save_heatmap(
            matrix=sharpe_matrix.corr(),
            output_path=results_root / "multi_asset_model_net_sharpe_correlation_heatmap.png",
            title="Model Net Sharpe Correlation Across Assets",
        )
        _print_multi_asset_robustness_summary(comparison)


def _run_baseline_suite(
    experiment: PreparedExperiment,
    output_path: Path,
    results_dir: Path,
    log_progress: bool = True,
    benchmark_label: str = "buy_and_hold_spy",
) -> pd.DataFrame:
    """Run the baseline model suite and save comparison artifacts."""

    registry = build_registry(experiment.config, experiment.results_dir)
    comparison_rows: list[dict[str, float | str]] = []
    enabled_models = get_enabled_models(experiment.config)
    trade_filter = _get_trade_filter_config(experiment.config)
    total_models = len(enabled_models)
    for model_index, model_name in enumerate(enabled_models, start=1):
        model = registry.get_model(model_name)
        label = model.get_display_name()
        if log_progress:
            print(f"\n[{model_index}/{total_models}] Starting model: {label}")
        artifacts = evaluate_model(
            model=model,
            label=label,
            feature_columns=experiment.feature_set.feature_columns,
            splits=experiment.splits,
            portfolio_config=experiment.config["portfolio"],
            results_dir=results_dir,
            artifact_prefix=label,
            log_progress=log_progress,
            trade_filter=trade_filter,
        )
        comparison_rows.append(artifacts.comparison_row)
        if log_progress:
            print(
                "  Completed model summary:",
                {
                    "model": label,
                    "sharpe": round(float(artifacts.comparison_row.get("sharpe", 0.0)), 4),
                    "auc_roc": round(float(artifacts.comparison_row.get("auc_roc", 0.0)), 4),
                    "directional_accuracy": round(
                        float(artifacts.comparison_row.get("directional_accuracy", 0.0)),
                        4,
                    ),
                },
            )

    comparison_rows.append(
        build_benchmark_row(
            experiment.splits,
            experiment.config["portfolio"],
            results_dir=results_dir,
            label=benchmark_label,
        )
    )
    comparison = summarize_results(comparison_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def _build_multi_asset_config(
    config: dict[str, Any],
    ticker: str,
    results_root: Path,
    cost_bps: float | None = None,
    regime_enabled: bool = False,
) -> dict[str, Any]:
    """Create an isolated config namespace for one asset in the multi-asset run."""

    asset_key = ticker.lower().replace("-", "_")
    cost_suffix = f"cost_{str(cost_bps).replace('.', '_')}bps" if cost_bps is not None else "default_cost"
    asset_results = results_root / "multi_asset" / asset_key / cost_suffix
    tuning_dir = asset_results / "optimization"
    ensemble_dir = asset_results / "ensembles"

    asset_config = deepcopy(config)
    asset_config["project"]["results_dir"] = str(asset_results)
    asset_config["data"]["ticker"] = ticker
    asset_config["data"]["cache_path"] = str(Path("data/multi_asset_cache") / f"{asset_key}_daily.csv")
    asset_config["tuning"]["results_dir"] = str(tuning_dir)
    asset_config["tuning"]["best_params_dir"] = str(tuning_dir / "best_params")
    asset_config["tuning"]["study_summary_path"] = str(tuning_dir / "study_summary.csv")
    asset_config["ensemble"]["results_dir"] = str(ensemble_dir)
    if cost_bps is not None:
        asset_config["portfolio"]["transaction_cost_bps"] = float(cost_bps)
    if regime_enabled:
        asset_config["regime"]["enabled"] = True
        asset_config["regime"]["include_interactions"] = False
    return asset_config


def _build_cli_overrides(tune_trials: int | None = None) -> dict[str, Any] | None:
    """Build config overrides from supported CLI flags."""

    overrides: dict[str, Any] = {}
    if tune_trials is not None:
        overrides["tuning"] = {"n_trials": int(tune_trials)}
    return overrides or None


def _merge_cli_overrides(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a nested override dictionary into an existing config mapping."""

    if not extra:
        return base
    merged = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _limit_experiment_splits(experiment: PreparedExperiment, max_splits: int | None) -> PreparedExperiment:
    """Limit the number of walk-forward splits for faster exploratory runs."""

    if max_splits is None or max_splits <= 0 or len(experiment.splits) <= max_splits:
        return experiment
    return PreparedExperiment(
        config=experiment.config,
        results_dir=experiment.results_dir,
        feature_set=experiment.feature_set,
        splits=experiment.splits[: max_splits],
    )


def _select_multi_asset_rows(
    final_comparison: pd.DataFrame,
    ticker: str,
    benchmark_label: str,
    cost_bps: float,
    regime_enabled: bool = False,
    champion_model: str = "regime_stacking_ensemble",
) -> pd.DataFrame:
    """Keep the core multi-asset comparison set and add buy-and-hold deltas."""

    if regime_enabled:
        selected_models = {benchmark_label, "stacking_ensemble", champion_model}
    else:
        selected_models = {
            benchmark_label,
            "stacking_ensemble",
            "patchtst_tuned",
            "itransformer_tuned",
            "lstm_tuned",
        }
    subset = final_comparison[final_comparison["model"].isin(selected_models)].copy()
    if subset.empty:
        return subset

    subset["ticker"] = ticker
    subset["cost_bps"] = float(cost_bps)
    buy_hold_net_sharpe = float(subset.loc[subset["model"] == benchmark_label, "net_sharpe"].iloc[0])
    subset["excess_net_sharpe_vs_buy_hold"] = subset["net_sharpe"] - buy_hold_net_sharpe
    subset["excess_vs_bh"] = subset["excess_net_sharpe_vs_buy_hold"]
    return subset.sort_values("net_sharpe", ascending=False).reset_index(drop=True)


def _save_heatmap(matrix: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a labeled heatmap using matplotlib only."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    heatmap = axis.imshow(matrix.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    axis.set_title(title)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _print_multi_asset_robustness_summary(comparison: pd.DataFrame) -> None:
    """Print a compact robustness table and summary paragraph."""

    benchmark_rows = comparison[comparison["model"].str.startswith("buy_and_hold_")][["ticker", "net_sharpe"]].rename(
        columns={"net_sharpe": "buy_hold_net_sharpe"}
    )
    learned_rows = comparison[~comparison["model"].str.startswith("buy_and_hold_")].copy()
    best_rows = learned_rows.sort_values(["ticker", "net_sharpe"], ascending=[True, False]).groupby("ticker", as_index=False).first()
    summary = best_rows.merge(benchmark_rows, on="ticker", how="left")
    summary["excess_vs_buy_hold"] = summary["net_sharpe"] - summary["buy_hold_net_sharpe"]
    printable = summary[["ticker", "model", "net_sharpe", "net_sortino", "calmar", "buy_hold_net_sharpe", "excess_vs_buy_hold"]]

    print("\nMulti-Asset Robustness Summary")
    print(printable.to_string(index=False))

    strongest = printable.sort_values("excess_vs_buy_hold", ascending=False).iloc[0]
    weakest = printable.sort_values("excess_vs_buy_hold", ascending=True).iloc[0]
    summary_text = (
        f"Strongest edge appears on {strongest['ticker']} via {strongest['model']} "
        f"(excess net Sharpe {strongest['excess_vs_buy_hold']:.3f} vs buy-and-hold). "
        f"Weakest relative result is on {weakest['ticker']} via {weakest['model']} "
        f"(excess net Sharpe {weakest['excess_vs_buy_hold']:.3f})."
    )
    print(summary_text)


def _build_multi_asset_regime_stacking_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Build the champion-vs-benchmark robustness table across assets and cost scenarios."""

    benchmark_rows = comparison[comparison["model"].str.startswith("buy_and_hold_")][
        ["ticker", "cost_bps", "net_sharpe", "net_sortino", "calmar"]
    ].rename(
        columns={
            "net_sharpe": "buy_hold_net_sharpe",
            "net_sortino": "buy_hold_net_sortino",
            "calmar": "buy_hold_calmar",
        }
    )
    stacking_rows = comparison[comparison["model"] == "stacking_ensemble"][
        ["ticker", "cost_bps", "net_sharpe", "net_sortino", "calmar"]
    ].rename(
        columns={
            "net_sharpe": "stacking_net_sharpe",
            "net_sortino": "stacking_net_sortino",
            "calmar": "stacking_calmar",
        }
    )
    champion_rows = comparison[comparison["model"] == "regime_stacking_ensemble"][
        ["ticker", "cost_bps", "net_sharpe", "net_sortino", "calmar"]
    ].rename(
        columns={
            "net_sharpe": "champion_net_sharpe",
            "net_sortino": "champion_net_sortino",
            "calmar": "champion_calmar",
        }
    )

    summary = champion_rows.merge(benchmark_rows, on=["ticker", "cost_bps"], how="left").merge(
        stacking_rows,
        on=["ticker", "cost_bps"],
        how="left",
    )
    summary["excess_vs_bh"] = summary["champion_net_sharpe"] - summary["buy_hold_net_sharpe"]
    summary["excess_vs_stacking"] = summary["champion_net_sharpe"] - summary["stacking_net_sharpe"]
    return summary.sort_values(["cost_bps", "ticker"]).reset_index(drop=True)


def _print_multi_asset_regime_stacking_report(comparison: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print the final multi-asset regime-stacking robustness report."""

    print("\nMulti-Asset Regime Stacking Robustness Report")
    printable = summary[
        [
            "ticker",
            "cost_bps",
            "champion_net_sharpe",
            "buy_hold_net_sharpe",
            "stacking_net_sharpe",
            "champion_net_sortino",
            "champion_calmar",
            "excess_vs_bh",
            "excess_vs_stacking",
        ]
    ]
    print(printable.to_string(index=False))

    average_excess = summary.groupby("cost_bps", as_index=False)["excess_vs_bh"].mean().rename(
        columns={"excess_vs_bh": "average_excess_vs_bh"}
    )
    print("\nAverage excess net Sharpe vs Buy & Hold by cost scenario:")
    print(average_excess.to_string(index=False))

    strongest = summary.sort_values("excess_vs_bh", ascending=False).iloc[0]
    weakest = summary.sort_values("excess_vs_bh", ascending=True).iloc[0]
    print(
        "\nRobustness summary:",
        (
            f"Strongest edge appears on {strongest['ticker']} at {strongest['cost_bps']:.1f} bps "
            f"(excess net Sharpe {strongest['excess_vs_bh']:.3f}). "
            f"Weakest edge appears on {weakest['ticker']} at {weakest['cost_bps']:.1f} bps "
            f"(excess net Sharpe {weakest['excess_vs_bh']:.3f})."
        ),
    )


def _run_selective_portfolio_workflow(
    *,
    config: dict[str, Any],
    comparison: pd.DataFrame,
    run_records: list[dict[str, Any]],
    results_root: Path,
) -> None:
    """Build and report selective portfolios from eligible regime-stacking assets."""

    champion_model = str(config["regime"].get("default_model", "regime_stacking_ensemble"))
    signal_panels = load_asset_signal_panel(run_records, champion_model=champion_model)
    portfolio_rows: list[dict[str, Any]] = []
    transaction_cost_scenarios = list(config.get("stress_testing", {}).get("transaction_cost_scenarios", []))
    min_excess_sharpe = float(config["portfolio"].get("min_excess_sharpe", 0.0))
    volatility_target = float(config["portfolio"].get("volatility_target", 0.10))
    annualization_factor = int(config["portfolio"]["annualization_factor"])

    for cost_bps in transaction_cost_scenarios:
        panel = signal_panels.get(float(cost_bps))
        if panel is None or panel.empty:
            continue
        eligibility = select_eligible_assets(
            comparison=comparison,
            cost_bps=float(cost_bps),
            champion_model=champion_model,
            min_excess_sharpe=min_excess_sharpe,
        )
        built = build_selective_portfolios(
            signal_panel=panel,
            eligibility=eligibility,
            annualization_factor=annualization_factor,
            transaction_cost_bps=float(cost_bps),
            volatility_target=volatility_target,
        )
        for row in built:
            row["cost_bps"] = float(cost_bps)
            row["selected_assets_threshold"] = min_excess_sharpe
            portfolio_rows.append(row)

        benchmark_rows = comparison.loc[
            comparison["cost_bps"].eq(float(cost_bps))
            & (
                comparison["model"].eq("buy_and_hold_spy")
                | (comparison["model"].eq(champion_model) & comparison["ticker"].eq("BTC-USD"))
            )
        ].copy()
        for _, row in benchmark_rows.iterrows():
            portfolio_rows.append(
                {
                    "model": "spy_buy_and_hold" if row["model"] == "buy_and_hold_spy" else "btc_champion",
                    "cost_bps": float(cost_bps),
                    "net_sharpe": float(row["net_sharpe"]),
                    "net_sortino": float(row["net_sortino"]),
                    "calmar": float(row["calmar"]),
                    "trade_frequency": float(row.get("trade_frequency", 0.0)),
                    "turnover": float(row.get("turnover", 0.0)),
                    "selected_assets": row.get("ticker", ""),
                    "selected_count": 1,
                }
            )
        for _, row in eligibility.iterrows():
            portfolio_rows.append(
                {
                    "model": "eligibility",
                    "cost_bps": float(cost_bps),
                    "selected_assets": str(row["ticker"]),
                    "selected_count": int(row["eligible"]),
                    "net_sharpe": float(row["champion_net_sharpe"]),
                    "net_sortino": float(row["champion_net_sortino"]),
                    "calmar": float(row["champion_calmar"]),
                    "excess_vs_bh": float(row["excess_vs_bh"]),
                    "eligible": bool(row["eligible"]),
                    "trade_frequency": 0.0,
                    "turnover": 0.0,
                }
            )

    portfolio_frame = pd.DataFrame(portfolio_rows)
    spy_rows = portfolio_frame.loc[portfolio_frame["model"] == "spy_buy_and_hold", ["cost_bps", "net_sharpe"]].rename(
        columns={"net_sharpe": "spy_buy_hold_net_sharpe"}
    )
    portfolio_frame = portfolio_frame.merge(spy_rows, on="cost_bps", how="left")
    portfolio_frame["excess_vs_spy_buy_hold"] = portfolio_frame["net_sharpe"] - portfolio_frame["spy_buy_hold_net_sharpe"]

    comparison_path = results_root / "portfolio_selective_comparison.csv"
    clean_columns = [
        "model",
        "cost_bps",
        "net_sharpe",
        "net_sortino",
        "calmar",
        "trade_frequency",
        "turnover",
        "selected_assets",
        "selected_count",
        "excess_vs_spy_buy_hold",
    ]
    portfolio_frame[clean_columns].to_csv(comparison_path, index=False)

    summary = portfolio_frame.loc[portfolio_frame["model"].isin(["selective_equal_weight", "selective_vol_target", "spy_buy_and_hold", "equal_weight_all_assets", "btc_champion"])].copy()
    summary_path = results_root / "portfolio_robustness_summary.csv"
    summary[clean_columns].to_csv(summary_path, index=False)
    _print_selective_portfolio_report(summary)


def _print_selective_portfolio_report(summary: pd.DataFrame) -> None:
    """Print the final selective portfolio robustness report."""

    print("\nSelective Portfolio Robustness Report")
    print(summary.to_string(index=False))

    portfolio_only = summary[summary["model"].isin(["selective_equal_weight", "selective_vol_target"])]
    if portfolio_only.empty:
        return
    strongest = portfolio_only.sort_values("excess_vs_spy_buy_hold", ascending=False).iloc[0]
    weakest = portfolio_only.sort_values("excess_vs_spy_buy_hold", ascending=True).iloc[0]
    average_excess = portfolio_only.groupby("cost_bps", as_index=False)["excess_vs_spy_buy_hold"].mean()
    print("\nAverage excess vs SPY Buy & Hold by cost scenario:")
    print(average_excess.to_string(index=False))
    print(
        "\nProduction candidate summary:",
        (
            f"Strongest selective portfolio is {strongest['model']} at {strongest['cost_bps']:.1f} bps "
            f"(excess Sharpe {strongest['excess_vs_spy_buy_hold']:.3f}, selected assets {strongest['selected_assets']}). "
            f"Weakest selective portfolio is {weakest['model']} at {weakest['cost_bps']:.1f} bps "
            f"(excess Sharpe {weakest['excess_vs_spy_buy_hold']:.3f})."
        ),
    )


def _run_tuning_workflow(experiment: PreparedExperiment) -> None:
    """Run the tuning workflow and save distinct optimization artifacts."""

    optimization_dir = Path(experiment.config["tuning"]["results_dir"])
    optimization_dir.mkdir(parents=True, exist_ok=True)
    baseline_comparison = _run_baseline_suite(
        experiment=experiment,
        output_path=experiment.results_dir / "model_comparison.csv",
        results_dir=experiment.results_dir,
        log_progress=False,
    )
    tuned_rows, _best_params = run_optuna_tuning(experiment)
    merged_rows = baseline_comparison.to_dict("records") + tuned_rows
    tuned_comparison = summarize_results(merged_rows)
    output_path = optimization_dir / "model_comparison_tuned.csv"
    tuned_comparison.to_csv(output_path, index=False)
    print(f"\nSaved tuning study summary to: {experiment.config['tuning']['study_summary_path']}")
    print(f"Saved tuned comparison to: {output_path}")
    _print_comparison(tuned_comparison)


def _run_ensemble_workflow(experiment: PreparedExperiment) -> None:
    """Run the ensemble workflow using tuned params when available."""

    ensembles_dir = Path(experiment.config["ensemble"]["results_dir"])
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    baseline_comparison = _run_baseline_suite(
        experiment=experiment,
        output_path=experiment.results_dir / "model_comparison.csv",
        results_dir=experiment.results_dir,
        log_progress=False,
    )

    tuned_params = _load_best_params(experiment)
    base_model_names = _get_ensemble_base_models(experiment.config)
    oof_frame = _build_ensemble_oof_frame(experiment, base_model_names, tuned_params, ensembles_dir)
    ensemble_rows = _evaluate_ensembles(experiment, oof_frame, ensembles_dir)

    merged_rows = baseline_comparison.to_dict("records") + ensemble_rows
    if experiment.config["ensemble"].get("use_tuned_params", True):
        tuned_comparison_path = Path(experiment.config["tuning"]["results_dir"]) / "model_comparison_tuned.csv"
        if tuned_comparison_path.exists():
            tuned_rows = pd.read_csv(tuned_comparison_path).to_dict("records")
            merged_rows = tuned_rows + ensemble_rows

    ensemble_comparison = summarize_results(merged_rows)
    output_path = ensembles_dir / "model_comparison_ensemble.csv"
    previous_best_sharpe = _load_previous_best_sharpe(
        experiment.results_dir / "model_comparison_with_advanced_features.csv"
    )
    if previous_best_sharpe is None:
        previous_best_sharpe = _load_previous_best_sharpe(output_path)
    ensemble_comparison.to_csv(output_path, index=False)
    if experiment.config["features"].get("advanced_features", False):
        advanced_output_path = experiment.results_dir / "model_comparison_with_advanced_features.csv"
        ensemble_comparison.to_csv(advanced_output_path, index=False)
        print(f"Saved advanced-feature comparison to: {advanced_output_path}")
        _print_feature_improvement_summary(ensemble_comparison, previous_best_sharpe)
    cost_output_path = experiment.results_dir / "model_comparison_with_costs.csv"
    ensemble_comparison.to_csv(cost_output_path, index=False)
    print(f"Saved cost-aware comparison to: {cost_output_path}")
    print(f"\nSaved ensemble comparison to: {output_path}")
    _print_comparison(ensemble_comparison)


def _run_full_cost_aware_suite(
    experiment: PreparedExperiment,
    final_output_path: Path,
    save_advanced_feature_comparison: bool,
    save_cost_comparison: bool,
    benchmark_label: str = "buy_and_hold_spy",
) -> pd.DataFrame:
    """Run baseline, tuning, and ensemble workflows and persist one final comparison."""

    baseline_comparison = _run_baseline_suite(
        experiment=experiment,
        output_path=experiment.results_dir / "model_comparison.csv",
        results_dir=experiment.results_dir,
        log_progress=False,
        benchmark_label=benchmark_label,
    )
    tuned_rows, _ = run_optuna_tuning(experiment)
    tuned_comparison = summarize_results(baseline_comparison.to_dict("records") + tuned_rows)
    tuned_output = Path(experiment.config["tuning"]["results_dir"]) / "model_comparison_tuned.csv"
    tuned_comparison.to_csv(tuned_output, index=False)

    tuned_params = _load_best_params(experiment)
    ensembles_dir = Path(experiment.config["ensemble"]["results_dir"])
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    oof_frame = _build_ensemble_oof_frame(experiment, _get_ensemble_base_models(experiment.config), tuned_params, ensembles_dir)
    ensemble_rows = _evaluate_ensembles(experiment, oof_frame, ensembles_dir)
    merged_rows = tuned_comparison.to_dict("records") + ensemble_rows
    final_comparison = summarize_results(merged_rows)

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_comparison.to_csv(final_output_path, index=False)
    if save_advanced_feature_comparison and experiment.config["features"].get("advanced_features", False):
        final_comparison.to_csv(experiment.results_dir / "model_comparison_with_advanced_features.csv", index=False)
    if save_cost_comparison:
        final_comparison.to_csv(experiment.results_dir / "model_comparison_with_costs.csv", index=False)
    return final_comparison


def _build_regime_aware_experiment(
    experiment: PreparedExperiment,
) -> tuple[PreparedExperiment, pd.DataFrame, np.ndarray]:
    """Augment each walk-forward split with train-only regime features."""

    regime_config = RegimeDetectionConfig(
        model_type=str(experiment.config["regime"].get("model_type", "hmm")),
        n_regimes=int(experiment.config["regime"].get("n_regimes", 3)),
    )
    augmented_splits = []
    timeline_rows: list[pd.DataFrame] = []
    transition_counts = np.zeros((regime_config.n_regimes, regime_config.n_regimes), dtype=float)
    regime_feature_columns: list[str] | None = None
    min_regime_prob = float(experiment.config["regime"].get("min_regime_prob", 0.70))

    for split in experiment.splits:
        detector = MarketRegimeDetector(regime_config)
        detector.fit(split.train)

        train_augmented = split.train.copy()
        val_augmented = split.validation.copy()
        test_augmented = split.test.copy()
        for frame in [train_augmented, val_augmented, test_augmented]:
            prediction = detector.predict(frame)
            frame["regime_id"] = prediction.labels.values
            for column in prediction.probabilities.columns:
                frame[column] = prediction.probabilities[column].values
        if experiment.config["regime"].get("auto_detect_best_regime", True):
            decision = detector.identify_best_regime(
                train_frame=train_augmented,
                annualization_factor=int(experiment.config["portfolio"]["annualization_factor"]),
            )
            best_regime_id = decision.best_regime_id
        else:
            configured = experiment.config["regime"].get("favorable_regimes", [0])
            best_regime_id = int(configured[0]) if configured else 0
        for frame in [train_augmented, val_augmented, test_augmented]:
            add_aggressive_trade_filter_columns(
                frame,
                best_regime_id=best_regime_id,
                min_regime_prob=min_regime_prob,
            )
            feature_columns = add_regime_features(frame)
            if regime_feature_columns is None:
                regime_feature_columns = feature_columns

        augmented_splits.append(
            type(split)(
                train=train_augmented,
                validation=val_augmented,
                test=test_augmented,
                split_id=split.split_id,
            )
        )
        timeline_rows.append(
            pd.DataFrame(
                {
                    "date": test_augmented.index,
                    "split_id": split.split_id,
                    "regime_id": test_augmented["regime_id"].values,
                    "best_regime_id": test_augmented["best_regime_id"].values,
                    "best_regime_prob": test_augmented["best_regime_prob"].values,
                    "trade_allowed_aggressive": test_augmented["trade_allowed_aggressive"].values,
                }
            )
        )
        if detector.transition_matrix_ is not None:
            transition_counts += detector.transition_matrix_

    if regime_feature_columns is None:
        raise ValueError("Failed to generate regime-aware feature columns.")

    augmented_feature_set = type(experiment.feature_set)(
        frame=experiment.feature_set.frame,
        feature_columns=experiment.feature_set.feature_columns + regime_feature_columns,
        stationarity_summary=experiment.feature_set.stationarity_summary,
    )
    augmented_config = deepcopy(experiment.config)
    augmented_config["regime"]["enabled"] = True
    timeline = pd.concat(timeline_rows, ignore_index=True).sort_values("date")
    normalized_transition = transition_counts / np.clip(transition_counts.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
    return (
        PreparedExperiment(
            config=augmented_config,
            results_dir=experiment.results_dir,
            feature_set=augmented_feature_set,
            splits=augmented_splits,
        ),
        timeline,
        normalized_transition,
    )


def _build_ensemble_oof_frame(
    experiment: PreparedExperiment,
    base_model_names: list[str],
    tuned_params: dict[str, dict[str, Any]],
    results_dir: Path,
) -> pd.DataFrame:
    """Build merged OOF predictions for the ensemble base models."""

    merged_frame: pd.DataFrame | None = None
    for model_name in base_model_names:
        params = tuned_params.get(model_name)
        model, label = _instantiate_model(experiment, model_name, tuned_params=params, tuned_suffix=False)
        artifacts = evaluate_model(
            model=model,
            label=label,
            feature_columns=experiment.feature_set.feature_columns,
            splits=experiment.splits,
            portfolio_config=experiment.config["portfolio"],
            results_dir=results_dir,
            artifact_prefix=label,
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
            join_columns = [
                "date",
                "split_id",
                "target_direction",
                "forward_simple_return_1d",
                "benchmark_return_1d",
            ]
            for metadata_column in metadata_columns:
                if metadata_column in merged_frame.columns:
                    join_columns.append(metadata_column)
            merged_frame = merged_frame.merge(frame, on=join_columns, how="inner")

    if merged_frame is None:
        raise ValueError("No ensemble base models were configured.")
    return merged_frame.sort_values(["split_id", "date"]).reset_index(drop=True)


def _evaluate_ensembles(
    experiment: PreparedExperiment,
    oof_frame: pd.DataFrame,
    results_dir: Path,
) -> list[dict[str, float | str]]:
    """Fit and evaluate the configured ensembles using time-ordered OOF predictions."""

    base_columns = [column for column in oof_frame.columns if column.startswith("probability__")]
    ensemble_rows: list[dict[str, float | str]] = []
    weighted_results: list[dict[str, float | str]] = []
    stacking_results: list[dict[str, float | str]] = []
    regime_weighted_results: list[dict[str, float | str]] = []
    regime_stacking_results: list[dict[str, float | str]] = []
    weighted_curves: list[pd.DataFrame] = []
    stacking_curves: list[pd.DataFrame] = []
    regime_weighted_curves: list[pd.DataFrame] = []
    regime_stacking_curves: list[pd.DataFrame] = []
    regime_weight_tables: list[pd.DataFrame] = []
    regime_stacking_importance_tables: list[pd.DataFrame] = []
    regime_stacking_prediction_frames: list[pd.DataFrame] = []
    trade_filter = _get_trade_filter_config(experiment.config)
    regime_enabled = bool(experiment.config.get("regime", {}).get("enabled", False))
    use_regime_weighted = bool(experiment.config.get("regime", {}).get("use_regime_weighted_ensemble", False))
    use_regime_stacking = bool(experiment.config.get("regime", {}).get("use_regime_stacking", False))
    include_interactions = bool(experiment.config.get("regime", {}).get("include_interactions", False))
    n_regimes = int(experiment.config.get("regime", {}).get("n_regimes", 3))
    regime_stacking_label = "regime_stacking_ensemble_interactions" if include_interactions else "regime_stacking_ensemble"

    unique_splits = sorted(oof_frame["split_id"].unique())
    for split_id in unique_splits[1:]:
        train_frame = oof_frame[oof_frame["split_id"] < split_id].copy()
        test_frame = oof_frame[oof_frame["split_id"] == split_id].copy()

        weighted = WeightedAverageEnsemble(signal_threshold=experiment.config["portfolio"]["signal_threshold"])
        weighted.fit(
            train_frame=train_frame,
            base_columns=base_columns,
            annualization_factor=experiment.config["portfolio"]["annualization_factor"],
            transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
        )
        weighted_prediction = weighted.predict(test_frame)
        weighted_prediction = _apply_trade_filter_to_prediction(test_frame, weighted_prediction, trade_filter)
        weighted_results.append(
            evaluate_ensemble_predictions(
                label="weighted_average_ensemble",
                frame=test_frame,
                prediction=weighted_prediction,
                annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
            )
        )
        weighted_returns = compute_strategy_returns(
            returns=test_frame["forward_simple_return_1d"],
            signal=weighted_prediction.predictions,
            transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
        )
        weighted_curves.append(
            pd.DataFrame(
                {
                    "date": test_frame["date"].values,
                    "split_id": split_id,
                    "strategy_return": weighted_returns.values,
                    "equity_curve": compute_equity_curve(weighted_returns).values,
                }
            )
        )

        stacking = StackingEnsemble(signal_threshold=experiment.config["portfolio"]["signal_threshold"])
        stacking.fit(train_frame=train_frame, base_columns=base_columns)
        stacking_prediction = stacking.predict(test_frame)
        stacking_prediction = _apply_trade_filter_to_prediction(test_frame, stacking_prediction, trade_filter)
        stacking_results.append(
            evaluate_ensemble_predictions(
                label="stacking_ensemble",
                frame=test_frame,
                prediction=stacking_prediction,
                annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
            )
        )
        stacking_returns = compute_strategy_returns(
            returns=test_frame["forward_simple_return_1d"],
            signal=stacking_prediction.predictions,
            transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
        )
        stacking_curves.append(
            pd.DataFrame(
                {
                    "date": test_frame["date"].values,
                    "split_id": split_id,
                    "strategy_return": stacking_returns.values,
                    "equity_curve": compute_equity_curve(stacking_returns).values,
                }
            )
        )

        if use_regime_weighted and regime_enabled and any(column.startswith("regime_prob_") for column in test_frame.columns):
            regime_weighted = RegimeWeightedEnsemble(
                signal_threshold=experiment.config["portfolio"]["signal_threshold"],
                mode="soft",
            )
            regime_weighted.fit(
                train_frame=train_frame,
                base_columns=base_columns,
                annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
                n_regimes=n_regimes,
            )
            regime_prediction = regime_weighted.predict(test_frame)
            regime_prediction = _apply_trade_filter_to_prediction(test_frame, regime_prediction, trade_filter)
            regime_weighted_results.append(
                evaluate_ensemble_predictions(
                    label="regime_weighted_ensemble",
                    frame=test_frame,
                    prediction=regime_prediction,
                    annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                    transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
                )
            )
            regime_returns = compute_strategy_returns(
                returns=test_frame["forward_simple_return_1d"],
                signal=regime_prediction.predictions,
                transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
            )
            regime_weighted_curves.append(
                pd.DataFrame(
                    {
                        "date": test_frame["date"].values,
                        "split_id": split_id,
                        "strategy_return": regime_returns.values,
                        "equity_curve": compute_equity_curve(regime_returns).values,
                    }
                )
            )
            weight_table = regime_weighted.regime_weight_table()
            if not weight_table.empty:
                weight_table.insert(0, "split_id", split_id)
                regime_weight_tables.append(weight_table)

        if use_regime_stacking and regime_enabled and any(column.startswith("regime_prob_") for column in test_frame.columns):
            legacy_regime_stacking = RegimeStackingEnsemble(
                signal_threshold=experiment.config["portfolio"]["signal_threshold"],
                meta_learner=str(experiment.config["regime"].get("meta_learner", "logistic")),
                include_interactions=bool(experiment.config["regime"].get("include_interactions", False)),
                objective="net_sharpe",
                long_bias_penalty=0.0,
                max_long_exposure=1.0,
            )
            legacy_regime_stacking.fit(
                train_frame=train_frame,
                base_columns=base_columns,
                n_regimes=n_regimes,
                annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
            )
            legacy_prediction = legacy_regime_stacking.predict(test_frame)
            legacy_prediction = _apply_trade_filter_to_prediction(test_frame, legacy_prediction, trade_filter)
            regime_stacking_results.append(
                evaluate_ensemble_predictions(
                    label="regime_stacking_ensemble_legacy",
                    frame=test_frame,
                    prediction=legacy_prediction,
                    annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                    transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
                )
            )

            regime_stacking = RegimeStackingEnsemble(
                signal_threshold=experiment.config["portfolio"]["signal_threshold"],
                meta_learner=str(experiment.config["regime"].get("meta_learner", "logistic")),
                include_interactions=bool(experiment.config["regime"].get("include_interactions", False)),
                objective=str(experiment.config.get("ensemble", {}).get("objective", "information_ratio")),
                long_bias_penalty=float(experiment.config.get("ensemble", {}).get("long_bias_penalty", 0.5)),
                max_long_exposure=float(experiment.config.get("ensemble", {}).get("max_long_exposure", 0.85)),
            )
            regime_stacking.fit(
                train_frame=train_frame,
                base_columns=base_columns,
                n_regimes=n_regimes,
                annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
            )
            regime_stacking_prediction = regime_stacking.predict(test_frame)
            regime_stacking_prediction = _apply_trade_filter_to_prediction(test_frame, regime_stacking_prediction, trade_filter)
            regime_stacking_results.append(
                evaluate_ensemble_predictions(
                    label=regime_stacking_label,
                    frame=test_frame,
                    prediction=regime_stacking_prediction,
                    annualization_factor=experiment.config["portfolio"]["annualization_factor"],
                    transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
                )
            )
            regime_stacking_returns = compute_strategy_returns(
                returns=test_frame["forward_simple_return_1d"],
                signal=regime_stacking_prediction.predictions,
                transaction_cost_bps=experiment.config["portfolio"]["transaction_cost_bps"],
            )
            regime_stacking_curves.append(
                pd.DataFrame(
                    {
                        "date": test_frame["date"].values,
                        "split_id": split_id,
                        "strategy_return": regime_stacking_returns.values,
                        "equity_curve": compute_equity_curve(regime_stacking_returns).values,
                    }
                )
            )
            importance_table = regime_stacking.feature_importance_table()
            if not importance_table.empty:
                importance_table.insert(0, "split_id", split_id)
                regime_stacking_importance_tables.append(importance_table)
            regime_stacking_prediction_frames.append(
                build_standard_audit_artifact_frame(
                    frame=test_frame.set_index("date", drop=False) if "date" in test_frame.columns else test_frame,
                    label=regime_stacking_label,
                    prediction=regime_stacking_prediction.predictions,
                    probability=regime_stacking_prediction.probabilities,
                    split_id=int(split_id),
                    transaction_cost_bps=float(experiment.config["portfolio"]["transaction_cost_bps"]),
                    asset_name=str(experiment.config.get("data", {}).get("ticker", "")) or None,
                )
            )

    ensemble_rows.append(_save_ensemble_artifacts("weighted_average_ensemble", weighted_results, weighted_curves, results_dir))
    ensemble_rows.append(_save_ensemble_artifacts("stacking_ensemble", stacking_results, stacking_curves, results_dir))
    if regime_weighted_results:
        ensemble_rows.append(
            _save_ensemble_artifacts(
                "regime_weighted_ensemble",
                regime_weighted_results,
                regime_weighted_curves,
                results_dir,
            )
        )
    if regime_weight_tables:
        pd.concat(regime_weight_tables, ignore_index=True).to_csv(
            results_dir / "regime_weighted_ensemble_weights.csv",
            index=False,
        )
    if regime_stacking_results:
        ensemble_rows.append(
            _save_ensemble_artifacts(
                regime_stacking_label,
                regime_stacking_results,
                regime_stacking_curves,
                results_dir,
            )
        )
    if regime_stacking_importance_tables:
        pd.concat(regime_stacking_importance_tables, ignore_index=True).to_csv(
            results_dir
            / (
                "regime_stacking_feature_importance_interactions.csv"
                if include_interactions
                else "regime_stacking_feature_importance.csv"
            ),
            index=False,
        )
    if regime_stacking_prediction_frames:
        pd.concat(regime_stacking_prediction_frames, ignore_index=True).to_csv(
            results_dir / ("regime_stacking_oof_interactions.csv" if include_interactions else "regime_stacking_oof.csv"),
            index=False,
        )
    return ensemble_rows


def _save_ensemble_artifacts(
    label: str,
    per_split_results: list[dict[str, float | str]],
    curve_frames: list[pd.DataFrame],
    results_dir: Path,
) -> dict[str, float | str]:
    """Save ensemble curves and return the aggregate comparison row."""

    comparison_row = pd.DataFrame(per_split_results).mean(numeric_only=True).to_dict()
    comparison_row["model"] = label
    combined_curves = pd.concat(curve_frames, ignore_index=True)
    (results_dir / "equity_curves").mkdir(parents=True, exist_ok=True)
    combined_curves.to_csv(results_dir / "equity_curves" / f"{label}_per_split.csv", index=False)
    aggregated_curve = combined_curves.groupby("date", as_index=False)["strategy_return"].mean().sort_values("date")
    aggregated_curve["equity_curve"] = compute_equity_curve(
        aggregated_curve["strategy_return"].rename("strategy_return")
    ).values
    aggregated_curve.to_csv(results_dir / "equity_curves" / f"{label}_aggregated.csv", index=False)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(aggregated_curve["date"], aggregated_curve["equity_curve"], linewidth=1.5)
    axis.set_title(f"{label} aggregated equity curve")
    axis.set_xlabel("Date")
    axis.set_ylabel("Equity")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(results_dir / "equity_curves" / f"{label}_aggregated.png", dpi=150)
    plt.close(figure)
    return comparison_row


def _instantiate_model(
    experiment: PreparedExperiment,
    model_name: str,
    tuned_params: dict[str, Any] | None,
    tuned_suffix: bool = True,
):
    """Instantiate a model, optionally with tuned parameter overrides."""

    config = deepcopy(experiment.config)
    if tuned_params:
        config["models"][model_name] = {
            **config["models"][model_name],
            **_normalize_loaded_params(model_name=model_name, params=tuned_params),
        }
    registry = build_registry(config, experiment.results_dir)
    model = registry.get_model(model_name)
    label = model.get_display_name()
    if tuned_params and tuned_suffix:
        label = f"{label}_tuned"
    return model, label


def _load_best_params(experiment: PreparedExperiment) -> dict[str, dict[str, Any]]:
    """Load tuned params from disk when available."""

    best_params_dir = Path(experiment.config["tuning"]["best_params_dir"])
    params_by_model: dict[str, dict[str, Any]] = {}
    if not best_params_dir.exists():
        return params_by_model

    for path in best_params_dir.glob("*.json"):
        params_by_model[path.stem] = _normalize_loaded_params(
            model_name=path.stem,
            params=json.loads(path.read_text(encoding="utf-8")),
        )
    return params_by_model


def _normalize_loaded_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Normalize persisted tuning params into valid model constructor fields."""

    normalized = dict(params)
    if model_name == "itransformer":
        head_config = normalized.pop("head_config", None)
        if head_config is not None:
            hidden_size, n_heads = head_config
            normalized.setdefault("hidden_size", hidden_size)
            normalized.setdefault("n_heads", n_heads)
    return normalized


def _get_ensemble_base_models(config: dict[str, Any]) -> list[str]:
    """Resolve the base-model list for ensemble workflows."""

    regime_config = config.get("regime", {})
    if regime_config.get("enabled", False) and regime_config.get("use_regime_weighted_ensemble", False):
        base_models = regime_config.get("base_models")
        if base_models:
            return list(base_models)
    return list(config["ensemble"]["base_models"])


def _print_runtime_configuration(experiment: PreparedExperiment) -> None:
    """Print runtime configuration summary."""

    print(
        "\nRuntime configuration:",
        {
            "run_profile": experiment.config["project"].get("run_profile", "default"),
            "fast_mode": experiment.config["project"].get("fast_mode", False),
            "advanced_features": experiment.config["features"].get("advanced_features", False),
            "vix_features": experiment.config["features"].get("vix_features", False),
            "enabled_models": get_enabled_models(experiment.config),
            "num_splits": len(experiment.splits),
            "validation": experiment.config["validation"],
        },
    )


def _load_previous_best_sharpe(path: Path) -> float | None:
    """Load the previous best Sharpe from an existing comparison file if present."""

    if not path.exists():
        return None

    comparison = pd.read_csv(path)
    if "sharpe" not in comparison.columns or comparison.empty:
        return None
    return float(comparison["sharpe"].max())


def _print_feature_improvement_summary(comparison: pd.DataFrame, previous_best_sharpe: float | None) -> None:
    """Print before/after Sharpe comparison for the advanced-feature run."""

    best_row = comparison.sort_values("sharpe", ascending=False).iloc[0]
    current_best_sharpe = float(best_row["sharpe"])
    delta = None if previous_best_sharpe is None else current_best_sharpe - previous_best_sharpe
    print(
        "\nFeature Improvement Summary:",
        {
            "best_model": str(best_row["model"]),
            "current_best_sharpe": round(current_best_sharpe, 4),
            "previous_best_sharpe": None if previous_best_sharpe is None else round(previous_best_sharpe, 4),
            "sharpe_delta": None if delta is None else round(delta, 4),
        },
    )


def _print_comparison(comparison: pd.DataFrame) -> None:
    """Print comparison table and standard footer."""

    ranking_label = get_primary_ranking_label(comparison)
    print(f"\nRanked walk-forward comparison by {ranking_label}:")
    print(comparison.to_string(index=False))
    if "trade_frequency" in comparison.columns:
        print(
            "\nReporting note: `trade_frequency` is a legacy alias for `fraction_in_market` "
            "and does not represent literal trade count."
        )
    print(
        "\nFinancial ML hygiene checks: no raw prices in the feature set, "
        "walk-forward windows are strictly time ordered, and evaluation uses only held-out test slices."
    )
    print(f"\nSaved comparison artifacts under: results")


def _print_vix_impact_summary(with_vix_comparison: pd.DataFrame, without_vix_comparison: pd.DataFrame) -> None:
    """Print paired-summary diagnostics for VIX feature experiments."""

    def _best_metrics(frame: pd.DataFrame) -> dict[str, float]:
        best = frame.sort_values("net_sharpe", ascending=False).iloc[0]
        return {
            "best_net_sharpe": float(best["net_sharpe"]),
            "best_net_sortino": float(best["net_sortino"]),
            "best_calmar": float(best["calmar"]),
        }

    with_metrics = _best_metrics(with_vix_comparison)
    without_metrics = _best_metrics(without_vix_comparison)
    common_models = sorted(set(with_vix_comparison["model"]).intersection(set(without_vix_comparison["model"])))
    deltas: list[tuple[str, float]] = []
    for model_name in common_models:
        with_value = float(with_vix_comparison.loc[with_vix_comparison["model"] == model_name, "net_sharpe"].iloc[0])
        without_value = float(without_vix_comparison.loc[without_vix_comparison["model"] == model_name, "net_sharpe"].iloc[0])
        deltas.append((model_name, with_value - without_value))
    deltas.sort(key=lambda item: item[1], reverse=True)

    print("\n=== VIX Features Impact Summary ===")
    print(
        {
            "without_vix": {key: round(value, 4) for key, value in without_metrics.items()},
            "with_vix": {key: round(value, 4) for key, value in with_metrics.items()},
            "delta": {
                "net_sharpe": round(with_metrics["best_net_sharpe"] - without_metrics["best_net_sharpe"], 4),
                "net_sortino": round(with_metrics["best_net_sortino"] - without_metrics["best_net_sortino"], 4),
                "calmar": round(with_metrics["best_calmar"] - without_metrics["best_calmar"], 4),
            },
            "top_model_benefits": [{name: round(delta, 4)} for name, delta in deltas[:5]],
        }
    )


def _print_regime_weighted_summary(combined: pd.DataFrame, regime_summary: pd.DataFrame) -> None:
    """Print the core regime ensemble comparison and interpretability tables."""

    focus_models = [
        "stacking_ensemble_baseline",
        "regime_weighted_ensemble_regime",
        "regime_stacking_ensemble_regime",
        "regime_stacking_ensemble_interactions_regime",
    ]
    focus_columns = [
        "model",
        "information_ratio",
        "active_calmar",
        "annualized_active_return",
        "net_sharpe",
        "net_sortino",
        "calmar",
        "fraction_in_market",
        "average_long_exposure",
        "daily_turnover",
    ]
    focus = combined.loc[
        combined["model"].isin(focus_models),
        [column for column in focus_columns if column in combined.columns],
    ].copy()
    print("\nRegime Stacking Summary")
    print(focus.to_string(index=False))
    print(
        "\nDiagnostic note: feature-importance and per-regime sections may still mix "
        "interaction and non-interaction artifacts when both files exist."
    )

    if not regime_summary.empty:
        print("\nRegime ensemble improvement summary:")
        improvement = regime_summary.loc[regime_summary["section"] == "three_way"]
        if not improvement.empty:
            print(improvement.to_string(index=False))

        interaction_delta = regime_summary.loc[regime_summary["section"] == "interaction_delta"]
        if not interaction_delta.empty:
            print("\nInteraction delta:")
            print(interaction_delta.to_string(index=False))

        feature_importance = regime_summary.loc[regime_summary["section"] == "feature_importance"]
        if not feature_importance.empty:
            print("\nRegime stacking meta-feature importance:")
            print(feature_importance.to_string(index=False))

        weights = regime_summary.loc[regime_summary["section"] == "regime_weights"]
        if not weights.empty:
            print("\nPer-regime model weights:")
            print(weights.to_string(index=False))

        per_regime = regime_summary.loc[regime_summary["section"] == "per_regime"]
        if not per_regime.empty:
            print("\nRegime stacking per-regime performance:")
            print(per_regime.to_string(index=False))


def _get_trade_filter_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the configured regime filter settings when trade filtering is enabled."""

    regime_config = config.get("regime", {})
    if not regime_config.get("enabled", False) or not regime_config.get("trade_filter", False):
        return None
    return build_trade_filter_config(config)


def _apply_trade_filter_to_prediction(
    frame: pd.DataFrame,
    prediction: EnsemblePrediction,
    trade_filter: dict[str, Any] | None,
) -> EnsemblePrediction:
    """Zero out trades in unfavorable regimes while leaving probabilities unchanged."""

    if trade_filter is None or "regime_id" not in frame.columns:
        return prediction
    allowed_mask = _build_trade_allowed_mask(frame=frame, trade_filter=trade_filter)
    return EnsemblePrediction(
        probabilities=prediction.probabilities,
        predictions=prediction.predictions.mask(~allowed_mask, other=0),
    )


def _build_trade_allowed_mask(frame: pd.DataFrame, trade_filter: dict[str, Any]) -> pd.Series:
    """Build a boolean trade mask from regime-filter metadata."""

    mode = str(trade_filter.get("filter_mode", "allowlist")).lower()
    if mode == "aggressive" and "trade_allowed_aggressive" in frame.columns:
        return frame["trade_allowed_aggressive"].astype(bool)

    allowed_regimes = {int(value) for value in trade_filter.get("favorable_regimes", [])}
    if not allowed_regimes:
        return pd.Series(True, index=frame.index)
    return frame["regime_id"].astype(int).isin(allowed_regimes)


def _build_regime_summary(
    baseline_comparison: pd.DataFrame,
    regime_comparison: pd.DataFrame,
    interactions_comparison: pd.DataFrame,
    regime_experiment: PreparedExperiment,
) -> pd.DataFrame:
    """Build a compact regime summary for baseline and both regime-stacking variants."""

    comparison_rows: list[dict[str, Any]] = []
    baseline_stacking = baseline_comparison.loc[baseline_comparison["model"] == "stacking_ensemble"]
    regime_weighted = regime_comparison.loc[regime_comparison["model"] == "regime_weighted_ensemble"]
    regime_stacking = regime_comparison.loc[regime_comparison["model"] == "regime_stacking_ensemble"]
    regime_stacking_interactions = interactions_comparison.loc[
        interactions_comparison["model"] == "regime_stacking_ensemble_interactions"
    ]

    if not baseline_stacking.empty:
        baseline_row = baseline_stacking.iloc[0]
        comparison_rows.append(
                {
                    "section": "three_way",
                    "model": "stacking_ensemble_baseline",
                    "information_ratio": float(baseline_row.get("information_ratio", np.nan)),
                    "active_calmar": float(baseline_row.get("active_calmar", np.nan)),
                    "annualized_active_return": float(baseline_row.get("annualized_active_return", np.nan)),
                    "net_sharpe": float(baseline_row["net_sharpe"]),
                    "net_sortino": float(baseline_row["net_sortino"]),
                    "calmar": float(baseline_row["calmar"]),
                    "fraction_in_market": float(baseline_row.get("fraction_in_market", baseline_row.get("trade_frequency", np.nan))),
                    "daily_turnover": float(baseline_row.get("daily_turnover", np.nan)),
                    "net_sharpe_delta": 0.0,
                    "net_sortino_delta": 0.0,
                    "calmar_delta": 0.0,
                }
            )
        if not regime_weighted.empty:
            weighted_row = regime_weighted.iloc[0]
            comparison_rows.append(
                {
                    "section": "three_way",
                    "model": "regime_weighted_ensemble_regime",
                    "information_ratio": float(weighted_row.get("information_ratio", np.nan)),
                    "active_calmar": float(weighted_row.get("active_calmar", np.nan)),
                    "annualized_active_return": float(weighted_row.get("annualized_active_return", np.nan)),
                    "net_sharpe": float(weighted_row["net_sharpe"]),
                    "net_sortino": float(weighted_row["net_sortino"]),
                    "calmar": float(weighted_row["calmar"]),
                    "fraction_in_market": float(weighted_row.get("fraction_in_market", weighted_row.get("trade_frequency", np.nan))),
                    "daily_turnover": float(weighted_row.get("daily_turnover", np.nan)),
                    "net_sharpe_delta": float(weighted_row["net_sharpe"] - baseline_row["net_sharpe"]),
                    "net_sortino_delta": float(weighted_row["net_sortino"] - baseline_row["net_sortino"]),
                    "calmar_delta": float(weighted_row["calmar"] - baseline_row["calmar"]),
                }
            )
        if not regime_stacking.empty:
            regime_row = regime_stacking.iloc[0]
            comparison_rows.append(
                {
                    "section": "three_way",
                    "model": "regime_stacking_ensemble_regime",
                    "information_ratio": float(regime_row.get("information_ratio", np.nan)),
                    "active_calmar": float(regime_row.get("active_calmar", np.nan)),
                    "annualized_active_return": float(regime_row.get("annualized_active_return", np.nan)),
                    "net_sharpe": float(regime_row["net_sharpe"]),
                    "net_sortino": float(regime_row["net_sortino"]),
                    "calmar": float(regime_row["calmar"]),
                    "fraction_in_market": float(regime_row.get("fraction_in_market", regime_row.get("trade_frequency", np.nan))),
                    "daily_turnover": float(regime_row.get("daily_turnover", np.nan)),
                    "net_sharpe_delta": float(regime_row["net_sharpe"] - baseline_row["net_sharpe"]),
                    "net_sortino_delta": float(regime_row["net_sortino"] - baseline_row["net_sortino"]),
                    "calmar_delta": float(regime_row["calmar"] - baseline_row["calmar"]),
                }
            )
        if not regime_stacking_interactions.empty:
            interaction_row = regime_stacking_interactions.iloc[0]
            comparison_rows.append(
                {
                    "section": "three_way",
                    "model": "regime_stacking_ensemble_interactions_regime",
                    "information_ratio": float(interaction_row.get("information_ratio", np.nan)),
                    "active_calmar": float(interaction_row.get("active_calmar", np.nan)),
                    "annualized_active_return": float(interaction_row.get("annualized_active_return", np.nan)),
                    "net_sharpe": float(interaction_row["net_sharpe"]),
                    "net_sortino": float(interaction_row["net_sortino"]),
                    "calmar": float(interaction_row["calmar"]),
                    "fraction_in_market": float(interaction_row.get("fraction_in_market", interaction_row.get("trade_frequency", np.nan))),
                    "daily_turnover": float(interaction_row.get("daily_turnover", np.nan)),
                    "net_sharpe_delta": float(interaction_row["net_sharpe"] - baseline_row["net_sharpe"]),
                    "net_sortino_delta": float(interaction_row["net_sortino"] - baseline_row["net_sortino"]),
                    "calmar_delta": float(interaction_row["calmar"] - baseline_row["calmar"]),
                }
            )

        if not regime_stacking.empty and not regime_stacking_interactions.empty:
            no_interaction_row = regime_stacking.iloc[0]
            interaction_row = regime_stacking_interactions.iloc[0]
            comparison_rows.append(
                {
                    "section": "interaction_delta",
                    "model": "regime_stacking_interactions_delta",
                    "information_ratio": float(interaction_row.get("information_ratio", np.nan)),
                    "active_calmar": float(interaction_row.get("active_calmar", np.nan)),
                    "annualized_active_return": float(interaction_row.get("annualized_active_return", np.nan)),
                    "net_sharpe": float(interaction_row["net_sharpe"]),
                    "net_sortino": float(interaction_row["net_sortino"]),
                    "calmar": float(interaction_row["calmar"]),
                    "fraction_in_market": float(interaction_row.get("fraction_in_market", interaction_row.get("trade_frequency", np.nan))),
                    "daily_turnover": float(interaction_row.get("daily_turnover", np.nan)),
                    "net_sharpe_delta": float(interaction_row["net_sharpe"] - no_interaction_row["net_sharpe"]),
                    "net_sortino_delta": float(interaction_row["net_sortino"] - no_interaction_row["net_sortino"]),
                    "calmar_delta": float(interaction_row["calmar"] - no_interaction_row["calmar"]),
                }
            )

    weights_path = Path(regime_experiment.config["ensemble"]["results_dir"]) / "regime_weighted_ensemble_weights.csv"
    if weights_path.exists():
        weights = pd.read_csv(weights_path)
        probability_columns = [column for column in weights.columns if column.startswith("probability__")]
        grouped = weights.groupby("regime_id", as_index=False)[probability_columns].mean()
        for _, row in grouped.iterrows():
            weight_row: dict[str, Any] = {
                "section": "regime_weights",
                "model": "regime_weighted_ensemble",
                "regime_id": int(row["regime_id"]),
            }
            for column in probability_columns:
                weight_row[column] = float(row[column])
            comparison_rows.append(weight_row)

    importance_path = Path(regime_experiment.config["ensemble"]["results_dir"]) / "regime_stacking_feature_importance_interactions.csv"
    if not importance_path.exists():
        importance_path = Path(regime_experiment.config["ensemble"]["results_dir"]) / "regime_stacking_feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        grouped = importance.groupby("feature", as_index=False)[["coefficient", "abs_coefficient"]].mean()
        for _, row in grouped.sort_values("abs_coefficient", ascending=False).head(12).iterrows():
            comparison_rows.append(
                {
                    "section": "feature_importance",
                    "model": "regime_stacking_ensemble",
                    "feature": str(row["feature"]),
                    "coefficient": float(row["coefficient"]),
                    "abs_coefficient": float(row["abs_coefficient"]),
                }
            )

    oof_path = Path(regime_experiment.config["ensemble"]["results_dir"]) / "regime_stacking_oof_interactions.csv"
    if not oof_path.exists():
        oof_path = Path(regime_experiment.config["ensemble"]["results_dir"]) / "regime_stacking_oof.csv"
    if oof_path.exists():
        oof_frame = pd.read_csv(oof_path)
        for regime_id, frame in oof_frame.groupby("regime_id"):
            trading = compute_trading_metrics(
                returns=frame["forward_simple_return_1d"],
                benchmark_returns=frame["benchmark_return_1d"],
                signal=frame["prediction"],
                annualization_factor=regime_experiment.config["portfolio"]["annualization_factor"],
                transaction_cost_bps=regime_experiment.config["portfolio"]["transaction_cost_bps"],
            )
            comparison_rows.append(
                {
                    "section": "per_regime",
                    "model": "regime_stacking_ensemble",
                    "regime_id": int(regime_id),
                    "fraction_in_market": float(trading.get("fraction_in_market", trading.get("trade_frequency", np.nan))),
                    "daily_turnover": float(trading.get("daily_turnover", np.nan)),
                    "net_sharpe": float(trading["net_sharpe"]),
                    "net_sortino": float(trading["net_sortino"]),
                    "calmar": float(trading["calmar"]),
                    "information_ratio": float(trading.get("information_ratio", np.nan)),
                    "active_calmar": float(trading.get("active_calmar", np.nan)),
                    "annualized_active_return": float(trading.get("annualized_active_return", np.nan)),
                }
            )

    return pd.DataFrame(comparison_rows)


def _evaluate_model_by_regime(regime_experiment: PreparedExperiment, model_name: str) -> pd.DataFrame:
    """Evaluate one regime-aware model and report trading metrics by regime."""

    base_model_name = _base_model_name(str(model_name))
    model, label = _instantiate_model(
        regime_experiment,
        base_model_name,
        tuned_params=_load_best_params(regime_experiment).get(base_model_name),
    )
    artifacts = evaluate_model(
        model=model,
        label=label,
        feature_columns=regime_experiment.feature_set.feature_columns,
        splits=regime_experiment.splits,
        portfolio_config=regime_experiment.config["portfolio"],
        log_progress=False,
        trade_filter=_get_trade_filter_config(regime_experiment.config),
    )
    rows: list[dict[str, Any]] = []
    for regime_id, frame in artifacts.oof_predictions.groupby("regime_id"):
        trading = compute_trading_metrics(
            returns=frame["forward_simple_return_1d"],
            benchmark_returns=frame["benchmark_return_1d"],
            signal=frame["prediction"],
            annualization_factor=regime_experiment.config["portfolio"]["annualization_factor"],
            transaction_cost_bps=regime_experiment.config["portfolio"]["transaction_cost_bps"],
        )
        rows.append(
            {
                "section": "per_regime",
                "model": label,
                "regime_id": int(regime_id),
                "best_regime_id": float(frame["best_regime_id"].mode().iloc[0]) if "best_regime_id" in frame.columns else np.nan,
                "trade_frequency": float(trading.get("trade_frequency", np.nan)),
                "net_sharpe": float(trading["net_sharpe"]),
                "net_sortino": float(trading["net_sortino"]),
                "calmar": float(trading["calmar"]),
                "net_sharpe_delta": np.nan,
                "calmar_delta": np.nan,
                "baseline_delta": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _with_regime_filter_state(experiment: PreparedExperiment, enabled: bool) -> PreparedExperiment:
    """Return a copy of the regime experiment with trade filtering toggled."""

    config = deepcopy(experiment.config)
    config["regime"]["trade_filter"] = bool(enabled)
    return PreparedExperiment(
        config=config,
        results_dir=experiment.results_dir,
        feature_set=experiment.feature_set,
        splits=experiment.splits,
    )


def _with_regime_stacking_interactions(experiment: PreparedExperiment, enabled: bool) -> PreparedExperiment:
    """Return a copy of the regime experiment with regime-stacking interactions toggled."""

    config = deepcopy(experiment.config)
    config["regime"]["include_interactions"] = bool(enabled)
    return PreparedExperiment(
        config=config,
        results_dir=experiment.results_dir,
        feature_set=experiment.feature_set,
        splits=experiment.splits,
    )


def _build_regime_interactions_comparison(
    baseline_comparison: pd.DataFrame,
    no_interactions_comparison: pd.DataFrame,
    interactions_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build a focused before/after interactions comparison table."""

    rows: list[dict[str, Any]] = []
    baseline_row = baseline_comparison.loc[baseline_comparison["model"] == "stacking_ensemble"].iloc[0]
    no_interaction_row = no_interactions_comparison.loc[
        no_interactions_comparison["model"] == "regime_stacking_ensemble"
    ].iloc[0]
    interaction_row = interactions_comparison.loc[
        interactions_comparison["model"] == "regime_stacking_ensemble_interactions"
    ].iloc[0]

    rows.append(
        {
            "variant": "baseline_stacking",
            "net_sharpe": float(baseline_row["net_sharpe"]),
            "net_sortino": float(baseline_row["net_sortino"]),
            "calmar": float(baseline_row["calmar"]),
            "net_sharpe_delta_vs_no_interactions": float(baseline_row["net_sharpe"] - no_interaction_row["net_sharpe"]),
            "net_sortino_delta_vs_no_interactions": float(baseline_row["net_sortino"] - no_interaction_row["net_sortino"]),
            "calmar_delta_vs_no_interactions": float(baseline_row["calmar"] - no_interaction_row["calmar"]),
        }
    )
    rows.append(
        {
            "variant": "regime_stacking_no_interactions",
            "net_sharpe": float(no_interaction_row["net_sharpe"]),
            "net_sortino": float(no_interaction_row["net_sortino"]),
            "calmar": float(no_interaction_row["calmar"]),
            "net_sharpe_delta_vs_no_interactions": 0.0,
            "net_sortino_delta_vs_no_interactions": 0.0,
            "calmar_delta_vs_no_interactions": 0.0,
        }
    )
    rows.append(
        {
            "variant": "regime_stacking_interactions",
            "net_sharpe": float(interaction_row["net_sharpe"]),
            "net_sortino": float(interaction_row["net_sortino"]),
            "calmar": float(interaction_row["calmar"]),
            "net_sharpe_delta_vs_no_interactions": float(interaction_row["net_sharpe"] - no_interaction_row["net_sharpe"]),
            "net_sortino_delta_vs_no_interactions": float(interaction_row["net_sortino"] - no_interaction_row["net_sortino"]),
            "calmar_delta_vs_no_interactions": float(interaction_row["calmar"] - no_interaction_row["calmar"]),
        }
    )
    return pd.DataFrame(rows)


def _print_regime_interactions_delta(regime_summary: pd.DataFrame) -> None:
    """Print the interaction-only delta block."""

    interaction_delta = regime_summary.loc[regime_summary["section"] == "interaction_delta"]
    if interaction_delta.empty:
        return
    print("\nRegime Stacking Interaction Delta")
    print(interaction_delta.to_string(index=False))


def _base_model_name(model_name: str) -> str:
    """Strip report suffixes back to the registry-backed base model name."""

    if model_name.endswith("_tuned"):
        return model_name[: -len("_tuned")]
    return model_name


def _save_regime_timeline_plot(timeline: pd.DataFrame, output_path: Path) -> None:
    """Save a simple regime timeline plot for the evaluated SPY test periods."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.step(timeline["date"], timeline["regime_id"], where="post")
    axis.set_title("SPY Regime Timeline")
    axis.set_xlabel("Date")
    axis.set_ylabel("Regime")
    axis.set_yticks([0, 1, 2])
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
