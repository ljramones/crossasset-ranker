"""Phase 1 universe diagnostics — documented BEFORE Phase 5 results are known.

Three sub-analyses on the 18-asset cross-asset universe / 5d forward horizon:

  1A. Per-date cross-sectional dispersion of forward 5d returns, plus
      correlation of that dispersion with v1's per-date Spearman from the
      prior scored panel (rank-quality high-water mark).

  1B. Cross-sectional predictability baseline: per-date Spearman between
      cross-sectional ranks of trailing 5d returns and forward 5d returns.
      The simplest possible momentum-based cross-sectional predictor — any
      sophisticated model needs to materially beat this.

  1C. Pairwise correlation distribution across the 18 assets, rolling 60d
      window, to characterise common-factor dominance vs cross-sectional
      structure.

Outputs CSVs and PNG plots under
``results/phase1_universe_diagnostics_<timestamp>/`` plus a short text
summary. No model training, no fetch, cached data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.market_cache import MarketCacheConfig, build_asset_cache_frame


UNIVERSE: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "DBA", "UUP", "VNQ",
    "BTC-USD",
)


def _load_returns_panel(start_date: str = "2010-01-01") -> pd.DataFrame:
    cfg = MarketCacheConfig(start_date=start_date, benchmark_ticker="SPY", vix_ticker="^VIX")
    rets: dict[str, pd.Series] = {}
    for asset in UNIVERSE:
        f = build_asset_cache_frame(asset, config=cfg)
        f = f[f["BenchmarkClose"].notna()]
        f["date"] = pd.to_datetime(f["Date"]).dt.normalize()
        f["return_1d"] = f["Adj Close"].astype(float).pct_change()
        rets[asset] = f.set_index("date")["return_1d"]
    return pd.DataFrame(rets).sort_index()


def _forward_horizon_log(returns: pd.Series, *, horizon: int) -> pd.Series:
    """Forward H-day simple return aligned to row t (uses t+1..t+H)."""
    log_r = np.log1p(returns)
    return np.expm1(log_r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1)))


def _trailing_horizon_log(returns: pd.Series, *, horizon: int) -> pd.Series:
    """Trailing H-day simple return aligned to row t (uses t-H+1..t)."""
    log_r = np.log1p(returns)
    return np.expm1(log_r.rolling(horizon).sum())


def phase1a_dispersion(returns_panel: pd.DataFrame, *, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date cross-sectional std of forward H-day returns."""
    fwd = pd.DataFrame(
        {a: _forward_horizon_log(returns_panel[a], horizon=horizon) for a in returns_panel.columns}
    )
    n_present = fwd.notna().sum(axis=1)
    disp = fwd.std(axis=1, ddof=1)
    col_name = f"fwd_{horizon}d_std"
    rows = pd.DataFrame(
        {"date": fwd.index, col_name: disp.values, "n_assets_present": n_present.values}
    ).dropna(subset=[col_name])
    summary = pd.DataFrame(
        [
            {
                "metric": col_name,
                "n": int(rows[col_name].notna().sum()),
                "mean": float(rows[col_name].mean()),
                "median": float(rows[col_name].median()),
                "p25": float(rows[col_name].quantile(0.25)),
                "p75": float(rows[col_name].quantile(0.75)),
                "p95": float(rows[col_name].quantile(0.95)),
            }
        ]
    )
    return rows, summary


def phase1b_predictability(returns_panel: pd.DataFrame, *, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date Spearman of trailing H-day rank vs forward H-day rank."""
    trailing = pd.DataFrame({a: _trailing_horizon_log(returns_panel[a], horizon=horizon) for a in returns_panel.columns})
    forward = pd.DataFrame({a: _forward_horizon_log(returns_panel[a], horizon=horizon) for a in returns_panel.columns})
    col_name = f"spearman_trailing_vs_forward_{horizon}d"

    rho_rows: list[dict] = []
    for date, fwd_row in forward.iterrows():
        tr_row = trailing.loc[date] if date in trailing.index else None
        if tr_row is None:
            continue
        merged = pd.concat([tr_row.rename("trailing"), fwd_row.rename("forward")], axis=1).dropna()
        if len(merged) < 3:
            continue
        rho = merged["trailing"].rank().corr(merged["forward"].rank(), method="pearson")
        if pd.notna(rho):
            rho_rows.append({"date": date, col_name: float(rho)})
    rows = pd.DataFrame(rho_rows)
    if rows.empty:
        return rows, pd.DataFrame()

    overall_mean = float(rows[col_name].mean())
    overall_median = float(rows[col_name].median())
    summary_rows = [
        {
            "metric": f"{col_name}_overall",
            "n_dates": int(len(rows)),
            "mean": overall_mean,
            "median": overall_median,
            "p25": float(rows[col_name].quantile(0.25)),
            "p75": float(rows[col_name].quantile(0.75)),
            "frac_positive": float((rows[col_name] > 0).mean()),
        }
    ]
    return rows, pd.DataFrame(summary_rows)


def phase1b_per_fold(
    rho_rows: pd.DataFrame,
    *,
    fold_dates: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
    horizon: int,
) -> pd.DataFrame:
    """Aggregate Phase 1B per-date Spearman onto the same per-fold structure."""
    col_name = f"spearman_trailing_vs_forward_{horizon}d"
    fold_rows: list[dict] = []
    for split_id, (start, end) in fold_dates.items():
        sub = rho_rows[(rho_rows["date"] >= start) & (rho_rows["date"] <= end)]
        if sub.empty:
            continue
        fold_rows.append(
            {
                "split_id": int(split_id),
                "n_dates": int(len(sub)),
                f"mean_{col_name}": float(sub[col_name].mean()),
                f"median_{col_name}": float(sub[col_name].median()),
            }
        )
    df = pd.DataFrame(fold_rows)
    if df.empty:
        return df
    mean_col = f"mean_{col_name}"
    mean_overall = float(df[mean_col].mean())
    std_overall = float(df[mean_col].std(ddof=1)) if len(df) > 1 else float("nan")
    icir = mean_overall / std_overall if std_overall and not np.isnan(std_overall) and std_overall > 0 else float("nan")
    df.loc[len(df)] = {
        "split_id": -1,
        "n_dates": int(df["n_dates"].sum()),
        mean_col: mean_overall,
        f"median_{col_name}": float("nan"),
    }
    df.attrs["icir"] = icir
    df.attrs["mean_overall"] = mean_overall
    df.attrs["std_across_folds"] = std_overall
    return df


def phase1c_pairwise_correlation(returns_panel: pd.DataFrame, window: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling pairwise correlation across the universe (mean / median per date)."""
    rows: list[dict] = []
    arr = returns_panel.to_numpy()
    dates = returns_panel.index.to_numpy()
    n_assets = returns_panel.shape[1]
    for i in range(window - 1, len(returns_panel)):
        block = arr[i - window + 1 : i + 1]
        block_df = pd.DataFrame(block, columns=returns_panel.columns)
        block_df = block_df.dropna(axis=1, how="any")
        if block_df.shape[1] < 2:
            continue
        c = block_df.corr().to_numpy()
        upper = c[np.triu_indices_from(c, k=1)]
        rows.append(
            {
                "date": pd.Timestamp(dates[i]),
                "mean_pairwise_corr": float(np.nanmean(upper)),
                "median_pairwise_corr": float(np.nanmedian(upper)),
                "n_assets_in_window": int(block_df.shape[1]),
            }
        )
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        return rows_df, pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "metric": "mean_pairwise_corr_60d",
                "n_dates": int(len(rows_df)),
                "mean": float(rows_df["mean_pairwise_corr"].mean()),
                "median": float(rows_df["mean_pairwise_corr"].median()),
                "p25": float(rows_df["mean_pairwise_corr"].quantile(0.25)),
                "p75": float(rows_df["mean_pairwise_corr"].quantile(0.75)),
                "p95": float(rows_df["mean_pairwise_corr"].quantile(0.95)),
            }
        ]
    )
    return rows_df, summary


def _v1_spearman_per_date(scored_csv: Path, *, horizon: int = 5) -> pd.DataFrame:
    """Compute v1's per-date Spearman from its scored panel against the realized H-day target."""
    if not scored_csv.exists():
        return pd.DataFrame(columns=["date", "v1_per_date_spearman"])
    scored = pd.read_csv(scored_csv, parse_dates=["date"])
    h = scored[scored["model"] == "lambdarank"]
    if h.empty:
        return pd.DataFrame(columns=["date", "v1_per_date_spearman"])
    # Need the target — recompute from raw cache.
    rp = _load_returns_panel()
    fwd = pd.DataFrame({a: _forward_horizon_log(rp[a], horizon=horizon) for a in rp.columns})
    vol = pd.DataFrame({a: rp[a].rolling(20).std(ddof=1).shift(1) * np.sqrt(252) for a in rp.columns})
    target = fwd / vol.replace(0.0, np.nan)
    rows: list[dict] = []
    for date, group in h.groupby("date"):
        if date not in target.index:
            continue
        t = target.loc[date].dropna()
        if len(t) < 3:
            continue
        merged = group.set_index("asset")[["score"]].join(t.rename("target"), how="inner").dropna()
        if len(merged) < 3:
            continue
        rho = merged["score"].rank().corr(merged["target"].rank(), method="pearson")
        if pd.notna(rho):
            rows.append({"date": date, "v1_per_date_spearman": float(rho)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--horizon", type=int, default=5, help="Forward / trailing horizon in trading days.")
    parser.add_argument(
        "--v1-scored-panel",
        default="results/cross_asset_ranking_5d_target_xs_features_lambdarank/cross_asset_ranking_scored_panel_20260511T143612Z.csv",
        help="Optional path to v1's scored panel for the dispersion-vs-Spearman correlation step. "
             "Only meaningful when --horizon matches the v1 run (5).",
    )
    args = parser.parse_args()
    horizon = int(args.horizon)

    ts = strftime("%Y%m%dT%H%M%SZ", gmtime())
    if args.output_dir:
        out = Path(args.output_dir)
    elif horizon == 5:
        out = Path(f"results/phase1_universe_diagnostics_{ts}")
    else:
        out = Path(f"results/phase1_universe_diagnostics_{horizon}d_{ts}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{ts}] Phase 1 universe diagnostics — horizon={horizon}d")
    print(f"  output dir: {out}")

    rp = _load_returns_panel()
    print(f"  returns panel: {rp.shape[0]} dates × {rp.shape[1]} assets")
    print(f"  date range: {rp.index.min().date()} -> {rp.index.max().date()}")

    # 1A
    disp_rows, disp_summary = phase1a_dispersion(rp, horizon=horizon)
    disp_rows.to_csv(out / f"phase1a_dispersion_{horizon}d_per_date.csv", index=False)
    disp_summary.to_csv(out / f"phase1a_dispersion_{horizon}d_summary.csv", index=False)
    print(f"\n[1A] cross-sectional dispersion of forward {horizon}d returns (raw {horizon}d std):")
    print(disp_summary.to_string(index=False))

    # Dispersion-vs-Spearman correlation against v1 per-date Spearman.
    disp_col = f"fwd_{horizon}d_std"
    rho_col = f"spearman_trailing_vs_forward_{horizon}d"
    if horizon == 5:
        v1_spearman = _v1_spearman_per_date(Path(args.v1_scored_panel), horizon=horizon)
        if not v1_spearman.empty:
            merged = disp_rows.merge(v1_spearman, on="date", how="inner")
            if not merged.empty:
                disp_spearman_corr = float(
                    merged[disp_col].corr(merged["v1_per_date_spearman"], method="pearson")
                )
                merged.to_csv(out / f"phase1a_dispersion_{horizon}d_vs_v1_spearman.csv", index=False)
                print(f"  pearson(dispersion, v1 per-date Spearman) = {disp_spearman_corr:+.4f} on {len(merged)} dates")
            else:
                disp_spearman_corr = float("nan")
                print("  v1 Spearman / dispersion merge was empty — skipping correlation")
        else:
            disp_spearman_corr = float("nan")
            print("  v1 scored panel not found; skipping dispersion-vs-Spearman correlation")
    else:
        disp_spearman_corr = float("nan")
        print(f"  (skipping dispersion-vs-Spearman correlation at horizon={horizon}d — no prior model run at this horizon yet)")

    # 1B
    rho_rows, rho_summary = phase1b_predictability(rp, horizon=horizon)
    rho_rows.to_csv(out / f"phase1b_predictability_{horizon}d_per_date.csv", index=False)
    rho_summary.to_csv(out / f"phase1b_predictability_{horizon}d_summary.csv", index=False)
    print(f"\n[1B] per-date Spearman of trailing-{horizon}d-rank vs forward-{horizon}d-rank:")
    print(rho_summary.to_string(index=False))

    fold_dates = {
        0: (pd.Timestamp("2020-02-18"), pd.Timestamp("2021-02-16")),
        1: (pd.Timestamp("2021-02-17"), pd.Timestamp("2022-02-14")),
        2: (pd.Timestamp("2022-02-15"), pd.Timestamp("2023-05-12")),
        3: (pd.Timestamp("2023-05-15"), pd.Timestamp("2024-08-09")),
        4: (pd.Timestamp("2024-08-12"), pd.Timestamp("2025-08-13")),
    }
    per_fold = phase1b_per_fold(rho_rows, fold_dates=fold_dates, horizon=horizon)
    if not per_fold.empty:
        per_fold.to_csv(out / f"phase1b_predictability_{horizon}d_per_fold.csv", index=False)
        print("  per-fold:")
        print(per_fold.to_string(index=False))
        print(f"  ICIR (across the 5 folds): {per_fold.attrs.get('icir', float('nan')):+.4f}")

    # 1C
    pwc_rows, pwc_summary = phase1c_pairwise_correlation(rp, window=60)
    pwc_rows.to_csv(out / "phase1c_pairwise_corr_per_date.csv", index=False)
    pwc_summary.to_csv(out / "phase1c_pairwise_corr_summary.csv", index=False)
    print("\n[1C] rolling 60d pairwise correlation across universe (daily returns):")
    print(pwc_summary.to_string(index=False))

    # Plots (PNG; gitignored)
    try:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        disp_rows.set_index("date")[disp_col].rolling(60).mean().plot(
            ax=axes[0], title=f"Phase 1A — cross-sectional std of forward {horizon}d returns (rolling 60d mean)"
        )
        axes[0].set_ylabel(disp_col)
        rho_rows.set_index("date")[rho_col].rolling(60).mean().plot(
            ax=axes[1], title=f"Phase 1B — trailing-vs-forward {horizon}d cross-sectional Spearman (rolling 60d mean)"
        )
        axes[1].axhline(0.0, color="grey", linewidth=0.5)
        axes[1].axhline(0.05, color="red", linewidth=0.5, linestyle="--", label="decision-grade Spearman threshold")
        axes[1].axhline(0.03, color="orange", linewidth=0.5, linestyle="--", label="directional Spearman threshold")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].set_ylabel("Spearman")
        pwc_rows.set_index("date")["mean_pairwise_corr"].plot(
            ax=axes[2], title="Phase 1C — mean pairwise correlation across 18 assets (60d window)"
        )
        axes[2].set_ylabel("corr")
        plt.tight_layout()
        plt.savefig(out / "phase1_summary_plots.png", dpi=110)
        plt.close()
        print(f"\n  plots saved: {out / 'phase1_summary_plots.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"  plot generation failed: {exc}")

    # Save a metadata JSON
    meta = {
        "generated_at_utc": ts,
        "horizon_days": horizon,
        "universe": list(UNIVERSE),
        "panel_n_dates": int(rp.shape[0]),
        "panel_n_assets": int(rp.shape[1]),
        "panel_date_start": str(rp.index.min().date()),
        "panel_date_end": str(rp.index.max().date()),
        "phase1a_dispersion_summary": disp_summary.to_dict(orient="records"),
        "phase1a_pearson_dispersion_vs_v1_spearman": (
            None if pd.isna(disp_spearman_corr) else float(disp_spearman_corr)
        ),
        "phase1b_predictability_summary": rho_summary.to_dict(orient="records"),
        "phase1b_per_fold_icir": per_fold.attrs.get("icir") if not per_fold.empty else None,
        "phase1b_per_fold_mean_overall": per_fold.attrs.get("mean_overall") if not per_fold.empty else None,
        "phase1c_pairwise_corr_summary": pwc_summary.to_dict(orient="records"),
    }
    (out / "phase1_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
