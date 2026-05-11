"""Cross-asset ranking helpers for the feasibility prototype.

Pure functions only. No file IO, no argparse, no model fitting. Models live in
``experiments/cross_asset_ranking_experiment.py``; CLI lives in
``scripts/run_cross_asset_ranking_experiment.py``.

Lag convention used throughout this module:

* ``compute_allocation_returns`` shifts allocation weights by **+1 bar** before
  multiplying with returns. Weights derived from features available at the
  close of day ``t`` therefore apply to the realized return earned on day
  ``t+1``. Same convention as ``evaluation.metrics.compute_strategy_returns``.

Calendar convention:

* ``build_cross_asset_panel`` performs a strict inner-join across all asset
  date sets. Any date where one or more assets is missing data is dropped.
  This makes the cross-sectional rank well-defined at every panel date and
  forces the panel start to ``max(asset_first_dates)`` — which is BTC-USD's
  inception (~2014-09-17) for the default universe.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_RAW_PRICE_LEVEL_COLUMNS: frozenset[str] = frozenset(
    {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "BenchmarkClose",
        "VIXClose",
    }
)

_NEVER_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "asset",
        "split_id",
    }
)

_FORBIDDEN_FEATURE_PREFIXES: tuple[str, ...] = (
    "future_",
    "forward_",
    "target_",
    "label_",
    "is_top_",
    "cross_sectional_",
)

_FORBIDDEN_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "_label",
)
# Note: ``_rank`` was previously listed here to exclude the legacy
# ``cross_sectional_rank`` / ``cross_sectional_percentile_rank`` output columns
# from being used as features. The ``cross_sectional_`` prefix in
# ``_FORBIDDEN_FEATURE_PREFIXES`` already covers those, and the project's new
# cross-sectional input features intentionally use the ``xs_rank_*`` naming.
# Keeping ``_rank`` in the substring blacklist would silently filter them out.


def compute_forward_return(
    returns: pd.Series,
    *,
    horizon: int = 20,
) -> pd.Series:
    """Compound-return over the next ``horizon`` bars (t+1 … t+horizon).

    Returns a Series aligned to ``returns.index``. Rows without ``horizon``
    forward observations become NaN. The input series is not mutated.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    series = pd.Series(returns).astype(float)
    log_returns = np.log1p(series)
    forward_log_sum = log_returns.shift(-1).rolling(window=horizon).sum().shift(-(horizon - 1))
    forward = np.expm1(forward_log_sum)
    forward.name = f"forward_{horizon}d_return"
    return forward


def compute_trailing_realized_vol(
    returns: pd.Series,
    *,
    window: int = 20,
    annualization: int = 252,
) -> pd.Series:
    """Annualized realized volatility computed from past returns only.

    The rolling result is shifted by one bar so the value at row ``t`` uses
    information available strictly before the close of day ``t``.
    """

    if window <= 1:
        raise ValueError("window must be > 1.")
    series = pd.Series(returns).astype(float)
    rolling = series.rolling(window=window).std(ddof=1)
    trailing = rolling.shift(1) * np.sqrt(annualization)
    trailing.name = f"trailing_{window}d_realized_vol"
    return trailing


def compute_forward_risk_adjusted_return(
    returns: pd.Series,
    *,
    forward_horizon: int = 20,
    vol_window: int = 20,
    annualization: int = 252,
) -> pd.Series:
    """Forward H-day return divided by trailing-window annualized volatility."""

    forward = compute_forward_return(returns, horizon=forward_horizon)
    vol = compute_trailing_realized_vol(returns, window=vol_window, annualization=annualization)
    safe_vol = vol.replace(0.0, np.nan)
    target = forward / safe_vol
    target.name = f"forward_{forward_horizon}d_risk_adjusted_return"
    return target


def build_cross_asset_panel(
    asset_frames: dict[str, pd.DataFrame],
    *,
    date_col: str = "date",
    return_col: str = "return_1d",
    forward_horizon: int = 20,
    vol_window: int = 20,
    annualization: int = 252,
) -> pd.DataFrame:
    """Stack per-asset frames into a (date, asset) long-format panel.

    Each asset frame must contain ``date_col`` and ``return_col``. Any other
    columns are preserved as candidate features. The panel is restricted to
    dates present in *every* asset frame (strict inner-join on date) so the
    cross-sectional rank is well-defined at every row.
    """

    if not asset_frames:
        raise ValueError("asset_frames must not be empty.")

    common_dates: set[pd.Timestamp] | None = None
    normalized: dict[str, pd.DataFrame] = {}
    for asset, frame in asset_frames.items():
        if date_col not in frame.columns:
            raise KeyError(f"Asset {asset!r} frame is missing date column {date_col!r}.")
        if return_col not in frame.columns:
            raise KeyError(f"Asset {asset!r} frame is missing return column {return_col!r}.")
        prepared = frame.copy()
        prepared[date_col] = pd.to_datetime(prepared[date_col]).dt.tz_localize(None).dt.normalize()
        prepared = prepared.sort_values(date_col).drop_duplicates(subset=date_col, keep="last")
        prepared = prepared.reset_index(drop=True)
        normalized[asset] = prepared
        dates = set(prepared[date_col].tolist())
        common_dates = dates if common_dates is None else common_dates & dates

    if not common_dates:
        raise ValueError("No common dates across assets — check input frames or universe.")

    common_index = pd.DatetimeIndex(sorted(common_dates))
    forward_return_col = f"forward_{int(forward_horizon)}d_return"
    trailing_vol_col = f"trailing_{int(vol_window)}d_realized_vol"
    risk_adjusted_col = f"forward_{int(forward_horizon)}d_risk_adjusted_return"
    rows: list[pd.DataFrame] = []
    for asset, prepared in normalized.items():
        prepared = prepared[prepared[date_col].isin(common_index)].copy()
        prepared = prepared.sort_values(date_col).reset_index(drop=True)
        prepared["asset"] = asset
        prepared = prepared.set_index(date_col, drop=False)
        prepared[forward_return_col] = compute_forward_return(
            prepared[return_col],
            horizon=forward_horizon,
        ).values
        prepared[trailing_vol_col] = compute_trailing_realized_vol(
            prepared[return_col],
            window=vol_window,
            annualization=annualization,
        ).values
        prepared[risk_adjusted_col] = compute_forward_risk_adjusted_return(
            prepared[return_col],
            forward_horizon=forward_horizon,
            vol_window=vol_window,
            annualization=annualization,
        ).values
        prepared = prepared.reset_index(drop=True)
        rows.append(prepared)

    panel = pd.concat(rows, axis=0, ignore_index=True)
    panel = panel.sort_values([date_col, "asset"]).reset_index(drop=True)
    return panel


def add_cross_sectional_ranks(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    target_col: str = "forward_20d_risk_adjusted_return",
) -> pd.DataFrame:
    """Add cross-sectional rank, percentile rank, and top-k indicators per date.

    Higher ``target_col`` ranks better. Rows with NaN target are excluded from
    the rank computation and receive ``NaN`` in the rank columns and 0 in the
    indicator columns.
    """

    if date_col not in panel.columns:
        raise KeyError(f"Panel missing {date_col!r} column.")
    if target_col not in panel.columns:
        raise KeyError(f"Panel missing target column {target_col!r}.")

    output = panel.copy()
    grouped = output.groupby(date_col, group_keys=False)
    output["cross_sectional_rank"] = grouped[target_col].rank(method="first", ascending=False)
    output["cross_sectional_percentile_rank"] = grouped[target_col].rank(pct=True, ascending=True)
    output["is_top_1"] = (output["cross_sectional_rank"] == 1).astype("Int64")
    output["is_top_2"] = (output["cross_sectional_rank"] <= 2).astype("Int64")
    nan_target = output[target_col].isna()
    output.loc[nan_target, ["cross_sectional_rank", "cross_sectional_percentile_rank"]] = np.nan
    output.loc[nan_target, ["is_top_1", "is_top_2"]] = pd.NA
    return output


def select_cross_asset_feature_columns(
    panel: pd.DataFrame,
    *,
    target_col: str = "forward_20d_risk_adjusted_return",
) -> list[str]:
    """Return the list of columns safe to use as model inputs for ranking."""

    selected: list[str] = []
    for column in panel.columns:
        column_str = str(column)
        if column_str in _NEVER_FEATURE_COLUMNS:
            continue
        if column_str == target_col:
            continue
        if column_str in _RAW_PRICE_LEVEL_COLUMNS:
            continue
        if any(column_str.startswith(prefix) for prefix in _FORBIDDEN_FEATURE_PREFIXES):
            continue
        if any(token in column_str for token in _FORBIDDEN_FEATURE_SUBSTRINGS):
            continue
        if column_str in {"trailing_20d_realized_vol"}:
            continue
        if not pd.api.types.is_numeric_dtype(panel[column]):
            continue
        selected.append(column_str)
    return selected


def build_top_k_allocations(
    scored_panel: pd.DataFrame,
    *,
    score_col: str,
    k: int,
    date_col: str = "date",
    asset_col: str = "asset",
) -> pd.DataFrame:
    """Equal-weight allocation among the top-``k`` assets per date.

    Returns a long-format DataFrame with rows for every (date, asset) in the
    universe, including non-selected assets at weight 0. Ties are broken by
    asset name (deterministic).
    """

    if k <= 0:
        raise ValueError("k must be positive.")
    if score_col not in scored_panel.columns:
        raise KeyError(f"Scored panel missing score column {score_col!r}.")

    universe = sorted(scored_panel[asset_col].dropna().unique().tolist())
    rows: list[pd.DataFrame] = []
    for date, group in scored_panel.groupby(date_col, sort=True):
        scored = group[[asset_col, score_col]].dropna(subset=[score_col]).copy()
        if scored.empty:
            ranked = pd.DataFrame({asset_col: universe})
            ranked["weight"] = 0.0
            ranked["score"] = np.nan
            ranked["rank"] = np.nan
        else:
            scored = scored.sort_values([score_col, asset_col], ascending=[False, True]).reset_index(drop=True)
            scored["rank"] = np.arange(1, len(scored) + 1, dtype=float)
            top_k = min(k, len(scored))
            scored["weight"] = 0.0
            scored.loc[scored.index[:top_k], "weight"] = 1.0 / float(top_k)
            ranked = pd.DataFrame({asset_col: universe}).merge(
                scored.rename(columns={score_col: "score"}),
                on=asset_col,
                how="left",
            )
            ranked["weight"] = ranked["weight"].fillna(0.0)
        ranked.insert(0, date_col, date)
        rows.append(ranked)
    if not rows:
        return pd.DataFrame(columns=[date_col, asset_col, "weight", "score", "rank"])
    return pd.concat(rows, axis=0, ignore_index=True)


def compute_allocation_returns(
    allocations: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    return_col: str = "return_1d",
    transaction_cost_bps: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Translate (date, asset, weight) allocations into portfolio return streams.

    Lag convention: weights at the end of day ``t`` are applied to the
    next-day return at ``t+1`` via a +1-bar shift on the weight matrix.

    Returns
    -------
    gross_returns, net_returns, turnover : pd.Series indexed by date.
    """

    weight_matrix = allocations.pivot_table(
        index=date_col,
        columns=asset_col,
        values="weight",
        aggfunc="last",
        fill_value=0.0,
    ).sort_index()
    return_matrix = panel.pivot_table(
        index=date_col,
        columns=asset_col,
        values=return_col,
        aggfunc="last",
    ).sort_index()

    union_index = weight_matrix.index.union(return_matrix.index)
    weight_matrix = weight_matrix.reindex(union_index).fillna(0.0)
    return_matrix = return_matrix.reindex(union_index)
    common_assets = sorted(set(weight_matrix.columns) | set(return_matrix.columns))
    weight_matrix = weight_matrix.reindex(columns=common_assets, fill_value=0.0)
    return_matrix = return_matrix.reindex(columns=common_assets)

    lagged_weights = weight_matrix.shift(1).fillna(0.0)
    gross_returns = (lagged_weights * return_matrix.fillna(0.0)).sum(axis=1)

    prev_weights = weight_matrix.shift(1).fillna(0.0)
    weight_changes = (weight_matrix - prev_weights).abs()
    turnover = weight_changes.sum(axis=1)
    transaction_cost = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = gross_returns - transaction_cost

    gross_returns.name = "gross_return"
    net_returns.name = "net_return"
    turnover.name = "turnover"
    return gross_returns, net_returns, turnover


def build_equal_weight_allocations(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
) -> pd.DataFrame:
    """Equal-weight allocation across all assets present at each date."""

    rows: list[pd.DataFrame] = []
    for date, group in panel.groupby(date_col, sort=True):
        assets = sorted(group[asset_col].dropna().unique().tolist())
        if not assets:
            continue
        weight = 1.0 / float(len(assets))
        rows.append(
            pd.DataFrame(
                {
                    date_col: date,
                    asset_col: assets,
                    "weight": weight,
                    "score": np.nan,
                    "rank": np.nan,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=[date_col, asset_col, "weight", "score", "rank"])
    return pd.concat(rows, axis=0, ignore_index=True)


def random_top_k_allocations(
    panel: pd.DataFrame,
    *,
    k: int,
    n_runs: int = 100,
    random_state: int = 42,
    date_col: str = "date",
    asset_col: str = "asset",
) -> dict[str, pd.DataFrame]:
    """Generate ``n_runs`` independent random top-k allocation panels.

    For each run and each date, ``k`` assets are sampled uniformly at random
    from those available, and given equal weight. The output is a dict mapping
    a deterministic run label (``"run_0000"`` … ``"run_<n-1>"``) to its
    allocation DataFrame, suitable for ``compute_allocation_returns``.
    """

    if k <= 0:
        raise ValueError("k must be positive.")
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")

    rng = np.random.default_rng(random_state)
    universe_per_date: list[tuple[pd.Timestamp, list[str]]] = []
    for date, group in panel.groupby(date_col, sort=True):
        assets = sorted(group[asset_col].dropna().unique().tolist())
        if assets:
            universe_per_date.append((date, assets))

    full_universe = sorted({asset for _, assets in universe_per_date for asset in assets})
    runs: dict[str, pd.DataFrame] = {}
    for run_index in range(n_runs):
        date_blocks: list[pd.DataFrame] = []
        for date, assets in universe_per_date:
            top_k = min(k, len(assets))
            chosen = rng.choice(assets, size=top_k, replace=False)
            chosen_set = set(chosen.tolist())
            block = pd.DataFrame({asset_col: full_universe})
            block["weight"] = block[asset_col].apply(
                lambda a: 1.0 / float(top_k) if a in chosen_set else 0.0
            )
            block["score"] = np.nan
            block["rank"] = np.nan
            block.insert(0, date_col, date)
            date_blocks.append(block)
        run_label = f"run_{run_index:04d}"
        runs[run_label] = pd.concat(date_blocks, axis=0, ignore_index=True) if date_blocks else pd.DataFrame()
    return runs


def make_lambdarank_relevance_labels(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    target_col: str = "forward_20d_risk_adjusted_return",
) -> pd.Series:
    """Per-date integer relevance labels for LightGBM LambdaRank.

    Within each date the asset with the highest ``target_col`` value gets the
    largest integer (e.g. 5 with 6 assets present), down to 0 for the worst.
    Ties are broken deterministically (lexicographic on the panel's row order
    after a stable sort by date — i.e. by asset, given the canonical sort).
    Rows with NaN target receive NaN relevance and should be filtered out by
    the caller before fitting; LightGBM does not accept NaN labels.

    Returns a Series aligned to ``panel.index`` named ``"relevance"``.
    """

    if date_col not in panel.columns:
        raise KeyError(f"Panel missing date column {date_col!r}.")
    if target_col not in panel.columns:
        raise KeyError(f"Panel missing target column {target_col!r}.")

    relevance = pd.Series(np.nan, index=panel.index, name="relevance", dtype="float64")
    for _, group in panel.groupby(date_col, sort=False):
        valid = group[target_col].dropna()
        if valid.empty:
            continue
        ranks = valid.rank(method="first", ascending=True).astype(int) - 1
        relevance.loc[ranks.index] = ranks.astype("float64").to_numpy()
    return relevance


def build_lambdarank_groups(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
) -> list[int]:
    """Return contiguous group sizes per unique date.

    The caller is responsible for sorting ``panel`` by ``date_col`` so the
    rows belonging to each date are contiguous in panel-row order. This
    function inspects the column and returns one integer per unique date.
    """

    if date_col not in panel.columns:
        raise KeyError(f"Panel missing date column {date_col!r}.")
    if panel.empty:
        return []
    sizes = panel.groupby(date_col, sort=False).size().tolist()
    return [int(n) for n in sizes]


def normalize_features_per_asset_train_only(
    panel: pd.DataFrame,
    *,
    train_dates: Iterable[pd.Timestamp],
    feature_columns: Iterable[str],
    date_col: str = "date",
    asset_col: str = "asset",
) -> pd.DataFrame:
    """Per-asset z-score of ``feature_columns`` using train-only mean/std.

    For each (asset, feature), the mean and standard deviation are computed
    *only* from rows whose ``date`` is in ``train_dates``. Those train statistics
    are then applied to every row of that asset (train, validation, and test) —
    which preserves leakage discipline because the statistics never see
    validation or test values.

    * Columns not in ``feature_columns`` are left untouched (date, asset,
      target/forward/rank columns, raw OHLCV, etc.).
    * Features with zero or NaN train std are mapped to 0 for that asset (no
      division, no NaN/inf propagation).
    * Row order is preserved.
    """

    feature_list = [c for c in feature_columns if c in panel.columns]
    if not feature_list:
        return panel.copy()
    train_set = set(pd.to_datetime(list(train_dates)))
    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    train_mask = out[date_col].isin(train_set)
    for asset, indices in out.groupby(asset_col).groups.items():
        asset_idx = pd.Index(indices)
        train_idx = asset_idx[train_mask.loc[asset_idx].to_numpy()]
        if len(train_idx) == 0:
            # No training data for this asset under this split — leave untouched.
            continue
        train_block = out.loc[train_idx, feature_list].astype(float)
        means = train_block.mean(axis=0, skipna=True)
        stds = train_block.std(axis=0, skipna=True, ddof=1)
        full_block = out.loc[asset_idx, feature_list].astype(float)
        normalized = full_block.subtract(means, axis=1)
        safe_std = stds.replace(0.0, np.nan)
        normalized = normalized.divide(safe_std, axis=1)
        # Zero-std columns become NaN above; map to 0 (constant features carry no info).
        normalized = normalized.fillna(0.0)
        out.loc[asset_idx, feature_list] = normalized.values
    return out


_CROSS_SECTIONAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "xs_rank_ret_5d",
    "xs_rank_ret_20d",
    "xs_rank_ret_60d",
    "xs_rank_vol_20d",
    "xs_rank_drawdown_60d",
)

_REGIME_INTERACTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "xs_rank_ret_5d_x_vix_z",
    "xs_rank_ret_20d_x_vix_z",
    "xs_rank_ret_60d_x_vix_z",
    "xs_rank_vol_20d_x_vix_z",
    "xs_rank_drawdown_60d_x_vix_z",
)


def _per_date_normalized_rank(series_in_date: pd.Series) -> pd.Series:
    """Rank a per-date Series into [0, 1] — highest value → 1.0, lowest → 0.0.

    NaN inputs receive NaN ranks (``Series.rank`` default). Dates with fewer
    than two valid observations are returned all-NaN.
    """
    ranks = series_in_date.rank(ascending=True, method="average")
    valid = int(series_in_date.notna().sum())
    if valid <= 1:
        return pd.Series(np.nan, index=series_in_date.index, dtype="float64")
    return (ranks - 1.0) / (valid - 1.0)


def add_cross_sectional_features(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    return_col: str = "return_1d",
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """Append per-date cross-sectional rank features to a long-format panel.

    Five features are added (each normalized to [0, 1] per date — highest value
    earning the largest fraction):

    - ``xs_rank_ret_5d`` / ``xs_rank_ret_20d`` / ``xs_rank_ret_60d`` — rank of
      trailing log-return over 5 / 20 / 60 trading days.
    - ``xs_rank_vol_20d`` — rank of trailing 20-day realized return volatility.
    - ``xs_rank_drawdown_60d`` — rank of current drawdown
      (``1 - price / rolling_max(price, 60)``), so the most-drawn-down asset
      on a given date earns rank 1.0.

    The function does NOT z-score these features; they are already in [0, 1]
    by construction and the per-date rank encoding is the entire point of the
    feature. Existing per-asset features in the panel are left untouched.

    The function is leakage-free: every input uses only contemporaneous and
    backward-looking information, and the per-date rank uses only same-day
    cross-section across the universe.
    """

    if return_col not in panel.columns:
        raise KeyError(f"add_cross_sectional_features: missing {return_col!r}.")
    if price_col not in panel.columns:
        raise KeyError(f"add_cross_sectional_features: missing {price_col!r}.")

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([asset_col, date_col]).reset_index(drop=True)

    # Decide once which per-asset inputs need to be computed (must be evaluated
    # *before* the asset loop, otherwise the first asset adds the column and the
    # check returns False for every subsequent asset).
    need_return_5d = "return_5d" not in out.columns
    need_return_20d = "return_20d" not in out.columns
    need_realized_vol_20 = "realized_vol_20" not in out.columns

    for asset, group_idx in out.groupby(asset_col).groups.items():
        asset_idx = pd.Index(group_idx)
        returns = out.loc[asset_idx, return_col].astype(float)
        if need_return_5d:
            out.loc[asset_idx, "return_5d"] = np.expm1(np.log1p(returns).rolling(5).sum()).to_numpy()
        if need_return_20d:
            out.loc[asset_idx, "return_20d"] = np.expm1(np.log1p(returns).rolling(20).sum()).to_numpy()
        # return_60d is not produced by build_feature_set; always compute here.
        out.loc[asset_idx, "return_60d"] = np.expm1(np.log1p(returns).rolling(60).sum()).to_numpy()
        if need_realized_vol_20:
            out.loc[asset_idx, "realized_vol_20"] = returns.rolling(20).std(ddof=1).to_numpy()
        # current drawdown from 60-day rolling peak — always compute here.
        prices = out.loc[asset_idx, price_col].astype(float)
        rolling_peak = prices.rolling(60, min_periods=2).max()
        drawdown = 1.0 - prices / rolling_peak.replace(0.0, np.nan)
        out.loc[asset_idx, "current_drawdown_60d"] = drawdown.to_numpy()

    out = out.sort_values([date_col, asset_col]).reset_index(drop=True)

    rank_specs = (
        ("xs_rank_ret_5d", "return_5d"),
        ("xs_rank_ret_20d", "return_20d"),
        ("xs_rank_ret_60d", "return_60d"),
        ("xs_rank_vol_20d", "realized_vol_20"),
        ("xs_rank_drawdown_60d", "current_drawdown_60d"),
    )
    for new_col, source_col in rank_specs:
        out[new_col] = out.groupby(date_col, group_keys=False)[source_col].transform(_per_date_normalized_rank)
    return out


def add_vix_zscore_to_panel(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    vix_col: str = "VIXClose",
    window: int = 252,
    output_col: str = "vix_zscore_252d",
) -> pd.DataFrame:
    """Append a causal trailing ``window``-day VIX z-score to ``panel``.

    VIX is a market-state series — the same value applies to every asset on a
    given date. The function therefore computes the z-score on the unique
    per-date VIX series and broadcasts back to all (date, asset) rows of the
    long-format panel.

    Causality: ``pandas.Series.rolling(window, min_periods=window)`` uses the
    values at times t-window+1 through t inclusive. VIXClose(t) is known by
    close of t and the one-bar-lag convention applied downstream means weights
    derived at end-of-t affect returns from t+1, so including t in the window
    is leakage-safe.
    """

    if vix_col not in panel.columns:
        raise KeyError(f"add_vix_zscore_to_panel: missing {vix_col!r} column.")

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    per_date = (
        out.sort_values(date_col)[[date_col, vix_col]]
        .drop_duplicates(date_col)
        .set_index(date_col)[vix_col]
        .astype(float)
    )
    rolling_mean = per_date.rolling(window, min_periods=window).mean()
    rolling_std = per_date.rolling(window, min_periods=window).std(ddof=1)
    z = (per_date - rolling_mean) / rolling_std.replace(0.0, np.nan)
    z = z.rename(output_col).reset_index()
    out = out.merge(z, on=date_col, how="left")
    return out


def add_regime_interaction_features(
    panel: pd.DataFrame,
    *,
    base_features: Iterable[str] = _CROSS_SECTIONAL_FEATURE_COLUMNS,
    regime_col: str = "vix_zscore_252d",
    suffix: str = "_x_vix_z",
) -> pd.DataFrame:
    """Append element-wise regime-conditioned interaction features.

    For each ``feature`` in ``base_features`` the output adds a column
    ``f"{feature}{suffix}"`` equal to ``panel[feature] * panel[regime_col]``.
    The result is signed and unbounded; the cross-sectional rank flooring at 0
    means low-rank assets contribute near-zero regardless of regime, while
    top-ranked assets are amplified/inverted by the sign and magnitude of the
    regime indicator. This is intentional — discretizing or re-ranking would
    discard the very information the interaction is encoding.
    """

    if regime_col not in panel.columns:
        raise KeyError(f"add_regime_interaction_features: missing {regime_col!r} column.")
    out = panel.copy()
    regime = out[regime_col].astype(float)
    for feature in base_features:
        if feature not in out.columns:
            raise KeyError(
                f"add_regime_interaction_features: missing base feature {feature!r}."
            )
        out[f"{feature}{suffix}"] = out[feature].astype(float) * regime
    return out


def apply_rebalance_schedule(
    allocations: pd.DataFrame,
    *,
    rebalance_every: int,
    date_col: str = "date",
    asset_col: str = "asset",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Freeze weights between rebalance dates anchored to the first allocation date.

    Day index ``i`` is a rebalance day iff ``i % rebalance_every == 0`` (so the
    very first allocation date is always a rebalance day). Between rebalance
    days, weights are forward-filled from the most recent rebalance day.

    The one-bar lag applied later in :func:`compute_allocation_returns` still
    holds: weights observed at the close of date ``t`` apply to returns on
    ``t+1``. Rebalance scheduling only affects which dates can produce a *new*
    weight; it does not change the lag relationship between weights and returns.

    ``rebalance_every == 1`` is an identity transform.
    """

    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be a positive integer.")
    if rebalance_every == 1 or allocations.empty:
        return allocations.copy()

    sorted_alloc = allocations.sort_values([date_col, asset_col]).copy()
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(sorted_alloc[date_col]).unique()))
    rebalance_mask = pd.Series(
        [(idx % rebalance_every) == 0 for idx in range(len(unique_dates))],
        index=unique_dates,
    )

    weight_matrix = sorted_alloc.pivot_table(
        index=date_col,
        columns=asset_col,
        values=weight_col,
        aggfunc="last",
        fill_value=0.0,
    ).reindex(unique_dates)

    held = weight_matrix.copy().astype(float)
    held.iloc[~rebalance_mask.to_numpy()] = np.nan
    held = held.ffill()
    # Day 0 is always a rebalance day under our anchor, so this is just a safety net.
    held = held.fillna(weight_matrix.iloc[0])

    rebalanced_long = held.stack().rename(weight_col).reset_index()
    rebalanced_long = rebalanced_long.rename(columns={"level_0": date_col})
    if date_col not in rebalanced_long.columns:
        rebalanced_long = rebalanced_long.rename(columns={rebalanced_long.columns[0]: date_col})

    metadata_cols = [c for c in sorted_alloc.columns if c not in {date_col, asset_col, weight_col}]
    if metadata_cols:
        meta = sorted_alloc[[date_col, asset_col, *metadata_cols]].drop_duplicates(subset=[date_col, asset_col], keep="last")
        rebalanced_long = rebalanced_long.merge(meta, on=[date_col, asset_col], how="left")
    rebalanced_long = rebalanced_long.sort_values([date_col, asset_col]).reset_index(drop=True)
    return rebalanced_long


def assets_selected_per_date(allocations: pd.DataFrame, *, asset_col: str = "asset", date_col: str = "date") -> pd.Series:
    """Return per-date count of assets with positive weight."""

    selected = allocations[allocations["weight"] > 0.0]
    return selected.groupby(date_col)[asset_col].nunique()


def asset_selection_counts(allocations: pd.DataFrame, *, asset_col: str = "asset") -> pd.Series:
    """Return per-asset count of dates the asset was selected (weight > 0)."""

    selected = allocations[allocations["weight"] > 0.0]
    return selected.groupby(asset_col).size().sort_values(ascending=False)


def empirical_pvalue_one_sided_ge(observed: float, distribution: Iterable[float]) -> float:
    """Right-tailed empirical p-value: P(null >= observed) with +1 smoothing."""

    values = pd.Series([float(value) for value in distribution if pd.notna(value)])
    if values.empty:
        return float("nan")
    count_ge = int((values >= float(observed)).sum())
    return float((1 + count_ge) / (len(values) + 1))
