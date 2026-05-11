"""Cross-asset ranking feasibility experiment.

Pure logic only. No file IO, no argparse. The CLI wrapper lives in
``scripts.run_cross_asset_ranking_experiment``.

This is a *feasibility prototype*. It deliberately uses simple models, daily
reallocation, and a single small universe. It must never be cited as evidence
of production readiness.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.cross_asset_ranking import (
    _CROSS_SECTIONAL_FEATURE_COLUMNS,
    _REGIME_INTERACTION_FEATURE_COLUMNS,
    add_cross_sectional_features,
    add_cross_sectional_ranks,
    add_regime_interaction_features,
    add_vix_zscore_to_panel,
    apply_rebalance_schedule,
    asset_selection_counts,
    assets_selected_per_date,
    build_cross_asset_panel,
    build_equal_weight_allocations,
    build_lambdarank_groups,
    build_top_k_allocations,
    compute_allocation_returns,
    empirical_pvalue_one_sided_ge,
    make_lambdarank_relevance_labels,
    normalize_features_per_asset_train_only,
    random_top_k_allocations,
    select_cross_asset_feature_columns,
)


FEATURE_NORMALIZATION_CHOICES: tuple[str, ...] = ("none", "per_asset_train_zscore")
KNOWN_MODELS: tuple[str, ...] = (
    "momentum_baseline",
    "linear_regression",
    "hist_gradient_boosting",
    "lambdarank",
)
from evaluation.metrics import compute_return_stream_metrics
from evaluation.walk_forward import generate_walk_forward_splits


DEFAULT_TARGET_COLUMN = "forward_20d_risk_adjusted_return"
DEFAULT_RETURN_COLUMN = "return_1d"
DEFAULT_DATE_COLUMN = "date"
DEFAULT_ASSET_COLUMN = "asset"


def target_column_for_horizon(forward_horizon: int) -> str:
    """Return the target column name for a given forward horizon.

    Mirrors :func:`evaluation.cross_asset_ranking.build_cross_asset_panel`'s
    column-naming convention so the experiment runner and the panel builder
    stay in lockstep.
    """

    return f"forward_{int(forward_horizon)}d_risk_adjusted_return"


@dataclass
class CrossAssetRankingConfig:
    """Run configuration for the cross-asset ranking feasibility prototype."""

    assets: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD")
    forward_horizon: int = 20
    vol_window: int = 20
    train_size: int = 756
    val_size: int = 252
    test_size: int = 252
    step_size: int = 252
    transaction_cost_bps: float = 2.0
    top_k_values: tuple[int, ...] = (1, 2)
    model_names: tuple[str, ...] = (
        "momentum_baseline",
        "linear_regression",
        "hist_gradient_boosting",
    )
    random_null_runs: int = 100
    random_state: int = 42
    run_purpose: str = "plumbing"
    decision_grade: bool = False
    annualization_factor: int = 252
    rebalance_every: int = 1
    feature_normalization: str = "none"
    include_cross_sectional_features: bool = False
    include_regime_interactions: bool = False


def load_prepared_asset_frames(
    *,
    assets: tuple[str, ...],
    cache_dir: Path,
    benchmark_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    prepare_missing: bool = False,
    horizons: tuple[int, ...] = (10, 20),
    thresholds: tuple[float, ...] = (-0.02, -0.03, -0.05),
) -> dict[str, pd.DataFrame]:
    """Build a per-asset prepared feature frame, refusing to fetch by default.

    The experiment treats this layer as data-only. Models, splits, and metrics
    are computed downstream from the returned mapping.
    """

    from data.market_cache import (
        MarketCacheConfig,
        normalize_ticker_for_filename,
    )
    from scripts.prepare_feature_frame import prepare_single_asset_feature_frame

    cache_dir = Path(cache_dir)

    def _raw_cache_present(ticker: str) -> bool:
        return (cache_dir / f"{normalize_ticker_for_filename(ticker)}_daily.csv").exists()

    required_raw = {*assets, benchmark_ticker, vix_ticker}
    missing = sorted(t for t in required_raw if not _raw_cache_present(t))
    if missing and not prepare_missing:
        raise FileNotFoundError(
            "Missing raw OHLCV cache for: "
            + ", ".join(missing)
            + ". Run scripts/prepare_feature_frame.py with --execute (or pass "
            "prepare_missing=True / --prepare-missing) to fetch."
        )

    config = MarketCacheConfig(
        cache_dir=cache_dir,
        start_date=start_date,
        benchmark_ticker=benchmark_ticker,
        vix_ticker=vix_ticker,
    )

    frames: dict[str, pd.DataFrame] = {}
    for asset in assets:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            frame = prepare_single_asset_feature_frame(
                ticker=asset,
                benchmark_ticker=benchmark_ticker,
                vix_ticker=vix_ticker,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                include_drawdown_labels=False,
                horizons=horizons,
                thresholds=thresholds,
                output_csv=None,
                force_refresh=False,
            )
        # Use the asset's own ticker — overrides MarketCacheConfig defaults.
        _ = config  # quiet linters; config kept for future use
        frames[asset] = frame
    return frames


def _build_panel(
    frames: dict[str, pd.DataFrame],
    *,
    config: CrossAssetRankingConfig,
) -> pd.DataFrame:
    panel = build_cross_asset_panel(
        frames,
        date_col=DEFAULT_DATE_COLUMN,
        return_col=DEFAULT_RETURN_COLUMN,
        forward_horizon=config.forward_horizon,
        vol_window=config.vol_window,
        annualization=config.annualization_factor,
    )
    panel = add_cross_sectional_ranks(
        panel,
        date_col=DEFAULT_DATE_COLUMN,
        target_col=target_column_for_horizon(config.forward_horizon),
    )
    if config.include_cross_sectional_features:
        panel = add_cross_sectional_features(
            panel,
            date_col=DEFAULT_DATE_COLUMN,
            asset_col=DEFAULT_ASSET_COLUMN,
            return_col=DEFAULT_RETURN_COLUMN,
        )
    if config.include_regime_interactions:
        if not config.include_cross_sectional_features:
            raise ValueError(
                "include_regime_interactions requires include_cross_sectional_features=True "
                "(interactions are products of the cross-sectional rank features)."
            )
        panel = add_vix_zscore_to_panel(panel, date_col=DEFAULT_DATE_COLUMN, window=252)
        panel = add_regime_interaction_features(
            panel,
            base_features=_CROSS_SECTIONAL_FEATURE_COLUMNS,
            regime_col="vix_zscore_252d",
        )
    return panel


def _walk_forward_date_splits(panel: pd.DataFrame, *, config: CrossAssetRankingConfig) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, int]]:
    unique_dates = pd.DatetimeIndex(sorted(panel[DEFAULT_DATE_COLUMN].dropna().unique()))
    date_frame = pd.DataFrame({DEFAULT_DATE_COLUMN: unique_dates})
    splits = generate_walk_forward_splits(
        date_frame,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        step_size=config.step_size,
    )
    out: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, int]] = []
    for split in splits:
        out.append(
            (
                pd.DatetimeIndex(split.train[DEFAULT_DATE_COLUMN]),
                pd.DatetimeIndex(split.validation[DEFAULT_DATE_COLUMN]),
                pd.DatetimeIndex(split.test[DEFAULT_DATE_COLUMN]),
                split.split_id,
            )
        )
    return out


def _build_model(name: str) -> Pipeline | None:
    """Return a fitted-style sklearn pipeline, or ``None`` for non-fit models."""

    if name == "momentum_baseline":
        return None
    if name == "linear_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                # HGB handles NaNs natively, but we impute upstream features
                # (e.g. constant columns) for safety.
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=200,
                        max_depth=4,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model name: {name!r}")


def _score_test_panel(
    *,
    model_name: str,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, list[dict]]:
    """Return (scores, feature_importance_rows) aligned to ``test_panel`` rows.

    For model families that expose per-feature importance (currently just
    ``lambdarank``), the second element is a list of one dict per feature with
    keys ``feature``, ``gain``, ``split_count``. For other models the list is
    empty — the caller can still concatenate it harmlessly.
    """

    if model_name == "momentum_baseline":
        if "return_20d" in test_panel.columns:
            return test_panel["return_20d"].astype(float), []
        if "momentum_norm" in test_panel.columns:
            return test_panel["momentum_norm"].astype(float), []
        raise KeyError("momentum_baseline requires 'return_20d' or 'momentum_norm' in panel.")

    if model_name == "lambdarank":
        return _score_with_lambdarank(
            train_panel=train_panel,
            test_panel=test_panel,
            feature_columns=feature_columns,
            target_column=target_column,
        )

    pipeline = _build_model(model_name)
    if pipeline is None:
        raise RuntimeError(f"Pipeline missing for model {model_name!r}.")

    train = train_panel.dropna(subset=[target_column]).copy()
    if train.empty:
        return pd.Series(np.nan, index=test_panel.index, name="score"), []
    pipeline.fit(train[feature_columns], train[target_column])
    scores = pd.Series(pipeline.predict(test_panel[feature_columns]), index=test_panel.index, name="score")
    return scores, []


def _score_with_lambdarank(
    *,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    date_col: str = DEFAULT_DATE_COLUMN,
    asset_col: str = DEFAULT_ASSET_COLUMN,
) -> pd.Series:
    """Fit LightGBM LambdaRank with per-date groups and score the test panel.

    Training rows: only those with a non-NaN ``target_column``. Dates with
    fewer than 2 valid rows are dropped (no rankable pairs). Rows are sorted by
    ``date`` then ``asset`` so each date's group is contiguous, then converted
    into integer per-date relevance labels via
    :func:`make_lambdarank_relevance_labels`.

    At score time every row of ``test_panel`` is scored individually; LightGBM's
    predict() does not require group information.
    """

    from lightgbm import LGBMRanker

    train = train_panel.dropna(subset=[target_column]).copy()
    if train.empty:
        return pd.Series(np.nan, index=test_panel.index, name="score")

    train = train.sort_values([date_col, asset_col]).reset_index(drop=True)
    group_sizes = train.groupby(date_col, sort=False).size()
    keep_dates = group_sizes[group_sizes >= 2].index
    train = train[train[date_col].isin(keep_dates)].reset_index(drop=True)
    if train.empty:
        return pd.Series(np.nan, index=test_panel.index, name="score")

    relevance = make_lambdarank_relevance_labels(
        train,
        date_col=date_col,
        target_col=target_column,
    )
    train = train.assign(_relevance=relevance.astype("Int64"))
    train = train.dropna(subset=["_relevance"]).reset_index(drop=True)

    groups = build_lambdarank_groups(train, date_col=date_col)
    if not groups:
        return pd.Series(np.nan, index=test_panel.index, name="score")

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbose=-1,
    )
    X_train = train[feature_columns].astype(float)
    y_train = train["_relevance"].astype(int).to_numpy()
    model.fit(X_train, y_train, group=groups)

    X_test = test_panel[feature_columns].astype(float)
    scores = pd.Series(model.predict(X_test), index=test_panel.index, name="score")

    gain = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    split_count = np.asarray(model.booster_.feature_importance(importance_type="split"), dtype=int)
    importance_rows: list[dict] = [
        {
            "feature": str(feature),
            "gain": float(gain[i]),
            "split_count": int(split_count[i]),
        }
        for i, feature in enumerate(feature_columns)
    ]
    return scores, importance_rows


def _build_metrics_row(
    *,
    asset_name: str,
    model_name: str,
    k: int,
    split_id: int,
    test_dates: pd.DatetimeIndex,
    net_returns: pd.Series,
    gross_returns: pd.Series,
    benchmark_net_returns: pd.Series,
    turnover_series: pd.Series,
    annualization_factor: int,
    transaction_cost_bps: float,
    allocations: pd.DataFrame,
) -> dict[str, float | str]:
    aligned = pd.concat(
        [
            net_returns.rename("net"),
            gross_returns.rename("gross"),
            benchmark_net_returns.rename("benchmark"),
            turnover_series.rename("turnover"),
        ],
        axis=1,
    ).reindex(test_dates).dropna(subset=["net", "benchmark"])
    if aligned.empty:
        return {
            "split_id": int(split_id),
            "model": model_name,
            "top_k": int(k),
            "test_dates": int(len(test_dates)),
            "net_sharpe": float("nan"),
            "information_ratio": float("nan"),
            "annualized_active_return": float("nan"),
            "active_calmar": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float("nan"),
            "cost_drag": float("nan"),
            "mean_assets_selected": float("nan"),
        }

    daily_turnover = float(aligned["turnover"].mean())
    metrics = compute_return_stream_metrics(
        net_returns=aligned["net"],
        benchmark_returns=aligned["benchmark"],
        annualization_factor=annualization_factor,
        gross_returns=aligned["gross"],
        turnover=daily_turnover,
    )
    selected = assets_selected_per_date(allocations).reindex(test_dates).dropna()
    selection_counts = asset_selection_counts(allocations).to_dict()
    return {
        "split_id": int(split_id),
        "model": model_name,
        "top_k": int(k),
        "test_dates": int(len(aligned)),
        "net_sharpe": float(metrics["net_sharpe"]),
        "information_ratio": float(metrics["information_ratio"]),
        "annualized_active_return": float(metrics["annualized_active_return"]),
        "active_calmar": float(metrics["active_calmar"]),
        "active_max_drawdown": float(metrics["active_max_drawdown"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "turnover": float(metrics["turnover"]),
        "annualized_turnover": float(metrics["annualized_turnover"]),
        "cost_drag": float(metrics["cost_drag"]),
        "mean_assets_selected": float(selected.mean()) if not selected.empty else float("nan"),
        "asset_selection_counts": selection_counts,
        "transaction_cost_bps": float(transaction_cost_bps),
        "asset_universe": asset_name,
    }


def _run_random_nulls_for_split(
    *,
    panel_test: pd.DataFrame,
    benchmark_net_returns: pd.Series,
    test_dates: pd.DatetimeIndex,
    k: int,
    n_runs: int,
    random_state: int,
    transaction_cost_bps: float,
    annualization_factor: int,
    split_id: int,
    rebalance_every: int = 1,
) -> list[dict[str, float | str]]:
    runs = random_top_k_allocations(
        panel_test,
        k=k,
        n_runs=n_runs,
        random_state=random_state,
        date_col=DEFAULT_DATE_COLUMN,
        asset_col=DEFAULT_ASSET_COLUMN,
    )
    rows: list[dict[str, float | str]] = []
    for run_label, alloc in runs.items():
        if rebalance_every > 1:
            alloc = apply_rebalance_schedule(
                alloc,
                rebalance_every=rebalance_every,
                date_col=DEFAULT_DATE_COLUMN,
                asset_col=DEFAULT_ASSET_COLUMN,
            )
        gross, net, turnover = compute_allocation_returns(
            alloc,
            panel_test,
            date_col=DEFAULT_DATE_COLUMN,
            asset_col=DEFAULT_ASSET_COLUMN,
            return_col=DEFAULT_RETURN_COLUMN,
            transaction_cost_bps=transaction_cost_bps,
        )
        aligned = pd.concat(
            [net.rename("net"), benchmark_net_returns.rename("benchmark"), turnover.rename("turnover")],
            axis=1,
        ).reindex(test_dates).dropna(subset=["net", "benchmark"])
        if aligned.empty:
            continue
        metrics = compute_return_stream_metrics(
            net_returns=aligned["net"],
            benchmark_returns=aligned["benchmark"],
            annualization_factor=annualization_factor,
            gross_returns=aligned["net"],
            turnover=float(aligned["turnover"].mean()),
        )
        rows.append(
            {
                "split_id": int(split_id),
                "top_k": int(k),
                "null_run": run_label,
                "net_sharpe": float(metrics["net_sharpe"]),
                "information_ratio": float(metrics["information_ratio"]),
                "annualized_active_return": float(metrics["annualized_active_return"]),
                "turnover": float(metrics["turnover"]),
            }
        )
    return rows


def run_cross_asset_ranking_experiment(
    *,
    asset_frames: dict[str, pd.DataFrame],
    config: CrossAssetRankingConfig,
) -> dict[str, pd.DataFrame | dict]:
    """Run the prototype end-to-end on already-prepared per-asset frames."""

    target_column = target_column_for_horizon(config.forward_horizon)
    panel = _build_panel(asset_frames, config=config)
    feature_columns = select_cross_asset_feature_columns(panel, target_col=target_column)
    if not feature_columns:
        raise RuntimeError("No usable feature columns were selected from the panel.")

    splits = _walk_forward_date_splits(panel, config=config)
    if not splits:
        raise RuntimeError("Walk-forward produced no splits; reduce window sizes.")

    fold_rows: list[dict] = []
    scored_rows: list[pd.DataFrame] = []
    allocation_rows: list[pd.DataFrame] = []
    feature_importance_rows: list[dict] = []
    portfolio_rows: list[pd.DataFrame] = []
    null_rows: list[dict] = []
    null_pvalue_rows: list[dict] = []

    for train_dates, _val_dates, test_dates, split_id in splits:
        # Raw panels: used for allocation-return computation (return_1d must stay raw).
        train_panel_raw = panel[panel[DEFAULT_DATE_COLUMN].isin(train_dates)].copy()
        test_panel_raw = panel[panel[DEFAULT_DATE_COLUMN].isin(test_dates)].copy()
        if train_panel_raw.empty or test_panel_raw.empty:
            continue

        # Feature panels: what the model fits/scores on. Possibly normalized.
        if config.feature_normalization == "per_asset_train_zscore":
            feature_panel = normalize_features_per_asset_train_only(
                panel,
                train_dates=train_dates,
                feature_columns=feature_columns,
                date_col=DEFAULT_DATE_COLUMN,
                asset_col=DEFAULT_ASSET_COLUMN,
            )
        elif config.feature_normalization in {"none", ""}:
            feature_panel = panel
        else:
            raise ValueError(
                f"Unknown feature_normalization {config.feature_normalization!r}; "
                f"expected one of {FEATURE_NORMALIZATION_CHOICES}"
            )
        train_features = feature_panel[feature_panel[DEFAULT_DATE_COLUMN].isin(train_dates)].copy()
        test_features = feature_panel[feature_panel[DEFAULT_DATE_COLUMN].isin(test_dates)].copy()
        # Keep `train_panel` / `test_panel` aliases pointing at the *raw* panel
        # so anything reading return_1d (compute_allocation_returns, random
        # nulls) sees the unnormalized return series.
        train_panel = train_panel_raw
        test_panel = test_panel_raw

        equal_alloc = build_equal_weight_allocations(test_panel, date_col=DEFAULT_DATE_COLUMN, asset_col=DEFAULT_ASSET_COLUMN)
        equal_gross, equal_net, equal_turnover = compute_allocation_returns(
            equal_alloc,
            test_panel,
            date_col=DEFAULT_DATE_COLUMN,
            asset_col=DEFAULT_ASSET_COLUMN,
            return_col=DEFAULT_RETURN_COLUMN,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        benchmark_net_returns = equal_net.copy()
        equal_alloc_out = equal_alloc.assign(model="equal_weight", split_id=split_id, top_k=len(test_panel[DEFAULT_ASSET_COLUMN].unique()))
        allocation_rows.append(equal_alloc_out)
        equal_metrics = _build_metrics_row(
            asset_name="universe",
            model_name="equal_weight",
            k=int(test_panel[DEFAULT_ASSET_COLUMN].nunique()),
            split_id=split_id,
            test_dates=pd.DatetimeIndex(sorted(test_panel[DEFAULT_DATE_COLUMN].unique())),
            net_returns=equal_net,
            gross_returns=equal_gross,
            benchmark_net_returns=benchmark_net_returns,
            turnover_series=equal_turnover,
            annualization_factor=config.annualization_factor,
            transaction_cost_bps=config.transaction_cost_bps,
            allocations=equal_alloc,
        )
        fold_rows.append(equal_metrics)

        for model_name in config.model_names:
            scores, importance_for_split = _score_test_panel(
                model_name=model_name,
                train_panel=train_features,
                test_panel=test_features,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            for row in importance_for_split:
                feature_importance_rows.append({
                    "split_id": int(split_id),
                    "model": model_name,
                    **row,
                })
            scored = test_features[[DEFAULT_DATE_COLUMN, DEFAULT_ASSET_COLUMN]].copy()
            scored["score"] = scores.values
            scored["model"] = model_name
            scored["split_id"] = split_id
            scored_rows.append(scored)

            for k in config.top_k_values:
                model_alloc = build_top_k_allocations(
                    scored.rename(columns={"score": "model_score"}),
                    score_col="model_score",
                    k=k,
                    date_col=DEFAULT_DATE_COLUMN,
                    asset_col=DEFAULT_ASSET_COLUMN,
                )
                if config.rebalance_every > 1:
                    model_alloc = apply_rebalance_schedule(
                        model_alloc,
                        rebalance_every=config.rebalance_every,
                        date_col=DEFAULT_DATE_COLUMN,
                        asset_col=DEFAULT_ASSET_COLUMN,
                    )
                model_alloc_out = model_alloc.assign(model=model_name, split_id=split_id, top_k=k)
                allocation_rows.append(model_alloc_out)

                gross, net, turnover_series = compute_allocation_returns(
                    model_alloc,
                    test_panel,
                    date_col=DEFAULT_DATE_COLUMN,
                    asset_col=DEFAULT_ASSET_COLUMN,
                    return_col=DEFAULT_RETURN_COLUMN,
                    transaction_cost_bps=config.transaction_cost_bps,
                )
                row = _build_metrics_row(
                    asset_name="universe",
                    model_name=model_name,
                    k=k,
                    split_id=split_id,
                    test_dates=pd.DatetimeIndex(sorted(test_panel[DEFAULT_DATE_COLUMN].unique())),
                    net_returns=net,
                    gross_returns=gross,
                    benchmark_net_returns=benchmark_net_returns,
                    turnover_series=turnover_series,
                    annualization_factor=config.annualization_factor,
                    transaction_cost_bps=config.transaction_cost_bps,
                    allocations=model_alloc,
                )
                fold_rows.append(row)
                portfolio_rows.append(
                    pd.DataFrame(
                        {
                            DEFAULT_DATE_COLUMN: net.index,
                            "model": model_name,
                            "top_k": k,
                            "split_id": split_id,
                            "gross_return": gross.values,
                            "net_return": net.values,
                            "turnover": turnover_series.values,
                            "benchmark_net_return": benchmark_net_returns.reindex(net.index).values,
                        }
                    )
                )

        for k in config.top_k_values:
            null_for_k = _run_random_nulls_for_split(
                panel_test=test_panel,
                benchmark_net_returns=benchmark_net_returns,
                test_dates=pd.DatetimeIndex(sorted(test_panel[DEFAULT_DATE_COLUMN].unique())),
                k=k,
                n_runs=config.random_null_runs,
                random_state=config.random_state + split_id * 1000 + k,
                transaction_cost_bps=config.transaction_cost_bps,
                annualization_factor=config.annualization_factor,
                split_id=split_id,
                rebalance_every=config.rebalance_every,
            )
            null_rows.extend(null_for_k)
            null_ir_distribution = [r["information_ratio"] for r in null_for_k]
            for model_name in config.model_names:
                model_metric = next(
                    (
                        r["information_ratio"]
                        for r in fold_rows
                        if r.get("split_id") == split_id and r.get("model") == model_name and r.get("top_k") == k
                    ),
                    float("nan"),
                )
                pvalue = empirical_pvalue_one_sided_ge(model_metric, null_ir_distribution)
                null_pvalue_rows.append(
                    {
                        "split_id": int(split_id),
                        "model": model_name,
                        "top_k": int(k),
                        "model_information_ratio": float(model_metric) if pd.notna(model_metric) else float("nan"),
                        "null_runs": len(null_ir_distribution),
                        "pvalue_ir_ge_random_top_k": pvalue,
                    }
                )

    fold_details = pd.DataFrame(fold_rows)
    fold_details_export = fold_details.drop(columns=["asset_selection_counts"], errors="ignore")
    summary = (
        fold_details_export.groupby(["model", "top_k"], dropna=False, sort=True)
        [["net_sharpe", "information_ratio", "annualized_active_return", "active_calmar", "max_drawdown", "turnover", "cost_drag", "mean_assets_selected"]]
        .mean()
        .reset_index()
    )
    null_pvalues = pd.DataFrame(null_pvalue_rows)
    nulls_frame = pd.DataFrame(null_rows)
    if null_pvalues.empty:
        random_nulls = nulls_frame
    else:
        random_nulls = nulls_frame.merge(null_pvalues, on=["split_id", "top_k"], how="left", suffixes=("", "_pvalue_join"))

    metadata = {
        "main_py_used": False,
        "prepare_experiment_used": False,
        "old_model_zoo_used": False,
        "optuna_used": False,
        "deep_models_used": False,
        "stacking_used": False,
        "decision_grade": bool(config.decision_grade),
        "run_purpose": str(config.run_purpose),
        "assets": list(config.assets),
        "forward_horizon": int(config.forward_horizon),
        "vol_window": int(config.vol_window),
        "train_size": int(config.train_size),
        "val_size": int(config.val_size),
        "test_size": int(config.test_size),
        "step_size": int(config.step_size),
        "transaction_cost_bps": float(config.transaction_cost_bps),
        "top_k_values": list(config.top_k_values),
        "model_names": list(config.model_names),
        "random_null_runs": int(config.random_null_runs),
        "random_state": int(config.random_state),
        "annualization_factor": int(config.annualization_factor),
        "panel_row_count": int(len(panel)),
        "panel_date_start": str(pd.to_datetime(panel[DEFAULT_DATE_COLUMN].iloc[0]).date()) if not panel.empty else None,
        "panel_date_end": str(pd.to_datetime(panel[DEFAULT_DATE_COLUMN].iloc[-1]).date()) if not panel.empty else None,
        "feature_count": int(len(feature_columns)),
        "split_count": int(len(splits)),
        "lag_convention": "weights.shift(1) * returns — weights at close of t apply to returns of t+1",
        "calendar_convention": "strict inner-join across asset date sets",
        "policy": (
            f"rebalance every {int(config.rebalance_every)} bar(s); "
            "equal weight among top-k by score; non-rebalance days hold prior weights"
        ),
        "rebalance_every": int(config.rebalance_every),
        "feature_normalization": str(config.feature_normalization),
        "include_cross_sectional_features": bool(config.include_cross_sectional_features),
        "include_regime_interactions": bool(config.include_regime_interactions),
    }

    feature_importance_frame = (
        pd.DataFrame(feature_importance_rows)
        if feature_importance_rows
        else pd.DataFrame(columns=["split_id", "model", "feature", "gain", "split_count"])
    )

    # ICIR + per-fold Spearman. Computed once per (model, split) using realized
    # target values from the panel and the model's predicted scores from the
    # per-split scored panel. Spearman is the same regardless of top_k (since
    # top_k only changes the allocation policy, not the underlying scores), so
    # results are broadcast across the top_k rows of the summary.
    spearman_records, icir_records = _compute_spearman_and_icir(
        scored_rows=scored_rows,
        panel=panel,
        target_column=target_column,
        model_names=config.model_names,
        date_col=DEFAULT_DATE_COLUMN,
        asset_col=DEFAULT_ASSET_COLUMN,
    )
    if not spearman_records.empty:
        fold_details_export = fold_details_export.merge(
            spearman_records.rename(columns={"mean_spearman": "per_fold_mean_spearman"}),
            on=["split_id", "model"],
            how="left",
        )
    if not icir_records.empty:
        summary = summary.merge(icir_records, on="model", how="left")

    return {
        "summary": summary,
        "fold_details": fold_details_export,
        "scored_panel": pd.concat(scored_rows, axis=0, ignore_index=True) if scored_rows else pd.DataFrame(),
        "allocations": pd.concat(allocation_rows, axis=0, ignore_index=True) if allocation_rows else pd.DataFrame(),
        "portfolio_returns": pd.concat(portfolio_rows, axis=0, ignore_index=True) if portfolio_rows else pd.DataFrame(),
        "random_nulls": random_nulls,
        "null_pvalues": null_pvalues,
        "feature_importance": feature_importance_frame,
        "ranking_diagnostics": icir_records,
        "metadata": metadata,
    }


def _compute_spearman_and_icir(
    *,
    scored_rows: list[pd.DataFrame],
    panel: pd.DataFrame,
    target_column: str,
    model_names: tuple[str, ...],
    date_col: str,
    asset_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(model, split) mean Spearman of scores vs realized target, plus per-model ICIR.

    Returns (spearman_per_fold, icir_per_model) where
    spearman_per_fold has columns [split_id, model, mean_spearman, n_dates] and
    icir_per_model has columns [model, overall_mean_spearman, spearman_std_across_folds, icir, n_folds].
    """

    if not scored_rows or target_column not in panel.columns:
        return pd.DataFrame(columns=["split_id", "model", "mean_spearman", "n_dates"]), pd.DataFrame(
            columns=["model", "overall_mean_spearman", "spearman_std_across_folds", "icir", "n_folds"]
        )

    scored_all = pd.concat(scored_rows, axis=0, ignore_index=True)
    panel_target = panel.set_index([date_col, asset_col])[target_column]

    fold_rows: list[dict] = []
    icir_rows: list[dict] = []
    for model_name in model_names:
        model_scored = scored_all[scored_all["model"] == model_name]
        if model_scored.empty:
            continue
        per_fold_means: list[float] = []
        for split_id, split_group in model_scored.groupby("split_id"):
            rhos: list[float] = []
            for date, date_group in split_group.groupby(date_col):
                keys = [(date, a) for a in date_group[asset_col]]
                try:
                    targets = panel_target.reindex(keys)
                except KeyError:
                    continue
                merged = pd.DataFrame(
                    {
                        "score": date_group["score"].to_numpy(),
                        "target": targets.to_numpy(),
                    }
                ).dropna()
                if len(merged) < 3:
                    continue
                rho = merged["score"].rank().corr(merged["target"].rank(), method="pearson")
                if pd.notna(rho):
                    rhos.append(float(rho))
            if rhos:
                mean_rho = float(np.mean(rhos))
                fold_rows.append(
                    {
                        "split_id": int(split_id),
                        "model": model_name,
                        "mean_spearman": mean_rho,
                        "n_dates": len(rhos),
                    }
                )
                per_fold_means.append(mean_rho)
        if per_fold_means:
            overall_mean = float(np.mean(per_fold_means))
            if len(per_fold_means) > 1:
                fold_std = float(np.std(per_fold_means, ddof=1))
            else:
                fold_std = float("nan")
            icir = (
                overall_mean / fold_std
                if fold_std and not np.isnan(fold_std) and fold_std > 0.0
                else float("nan")
            )
            icir_rows.append(
                {
                    "model": model_name,
                    "overall_mean_spearman": overall_mean,
                    "spearman_std_across_folds": fold_std,
                    "icir": icir,
                    "n_folds": len(per_fold_means),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(icir_rows)
