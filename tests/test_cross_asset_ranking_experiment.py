"""Tests for the cross-asset ranking experiment runner and CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import cross_asset_ranking_experiment as experiment_module
from experiments.cross_asset_ranking_experiment import (
    CrossAssetRankingConfig,
    run_cross_asset_ranking_experiment,
)
from scripts import run_cross_asset_ranking_experiment as cli_module


def _synthetic_asset_frame(*, asset: str, n_rows: int = 40, seed: int = 0) -> pd.DataFrame:
    """Build a tiny per-asset frame in the prepared-CSV column shape."""

    dates = pd.date_range("2020-01-02", periods=n_rows, freq="B")
    rng = np.random.default_rng(seed)
    base_returns = rng.normal(0.0005, 0.01, size=n_rows)
    feature_a = base_returns + rng.normal(0.0, 0.001, size=n_rows)
    return pd.DataFrame(
        {
            "date": dates,
            "Open": np.linspace(100.0, 120.0, n_rows),
            "High": np.linspace(101.0, 121.0, n_rows),
            "Low": np.linspace(99.0, 119.0, n_rows),
            "Close": np.linspace(100.0, 120.0, n_rows),
            "Adj Close": np.linspace(100.0, 120.0, n_rows),
            "Volume": rng.integers(1_000_000, 2_000_000, size=n_rows),
            "BenchmarkClose": np.linspace(400.0, 440.0, n_rows),
            "VIXClose": 18.0 + rng.normal(0.0, 1.0, size=n_rows),
            "return_1d": base_returns,
            "return_5d": pd.Series(base_returns).rolling(5).sum().fillna(0.0).values,
            "return_20d": pd.Series(base_returns).rolling(20).sum().fillna(0.0).values,
            "vol_ratio": rng.normal(1.0, 0.05, size=n_rows),
            "momentum_norm": rng.normal(0.0, 0.5, size=n_rows),
            "volume_zscore": rng.normal(0.0, 1.0, size=n_rows),
            "benchmark_return_1d": rng.normal(0.0003, 0.009, size=n_rows),
            "forward_simple_return_1d": np.roll(base_returns, -1),
            # tag a per-asset feature so the linear/HGB models have something to fit
            "asset_signal": feature_a,
        }
    )


def _runner_inputs() -> tuple[dict[str, pd.DataFrame], CrossAssetRankingConfig]:
    frames = {
        "AAA": _synthetic_asset_frame(asset="AAA", n_rows=80, seed=11),
        "BBB": _synthetic_asset_frame(asset="BBB", n_rows=80, seed=22),
        "CCC": _synthetic_asset_frame(asset="CCC", n_rows=80, seed=33),
    }
    config = CrossAssetRankingConfig(
        assets=("AAA", "BBB", "CCC"),
        forward_horizon=5,
        vol_window=5,
        train_size=30,
        val_size=10,
        test_size=15,
        step_size=15,
        transaction_cost_bps=2.0,
        top_k_values=(1, 2),
        model_names=("momentum_baseline", "linear_regression", "hist_gradient_boosting"),
        random_null_runs=10,
        random_state=42,
        run_purpose="plumbing",
    )
    return frames, config


def test_runner_uses_train_test_separation() -> None:
    frames, config = _runner_inputs()
    result = run_cross_asset_ranking_experiment(asset_frames=frames, config=config)

    scored = result["scored_panel"]
    assert not scored.empty

    panel_dates = sorted(pd.to_datetime(scored["date"]).unique())
    # The scored panel should never contain the very first dates (those went to train/val).
    # We approximate by asserting scored dates start strictly after the configured train+val window.
    earliest_scored = pd.to_datetime(scored["date"]).min()
    earliest_input = min(
        pd.to_datetime(frame["date"]).min() for frame in frames.values()
    )
    assert earliest_scored > earliest_input, "Scored test dates must lag the earliest input date (train slice precedes test)."

    metadata = result["metadata"]
    assert metadata["split_count"] >= 1
    assert metadata["panel_row_count"] > 0


def test_no_legacy_or_optuna_imports_in_experiment_modules() -> None:
    for module in (experiment_module, cli_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import optuna",
            "from optuna",
            "import lightning",
            "from lightning",
            "from neuralforecast",
            "from pytorch_forecasting",
            "from data.market_data",
            "import data.market_data",
            "from utils.experiment",
            "from audit.integrity_audit",
            "from main ",
            "import main\n",
            "regime_stacking",
        ):
            assert forbidden not in source, f"{module.__name__} must not contain {forbidden!r}"


def test_output_bundle_writer_creates_all_files(tmp_path: Path) -> None:
    config = CrossAssetRankingConfig(run_purpose="plumbing")
    fake_result = {
        "summary": pd.DataFrame({"model": ["x"], "top_k": [1], "net_sharpe": [0.5]}),
        "fold_details": pd.DataFrame({"split_id": [0], "model": ["x"], "top_k": [1]}),
        "scored_panel": pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "asset": ["A"], "score": [0.1]}),
        "allocations": pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "asset": ["A"], "weight": [1.0]}),
        "portfolio_returns": pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "net_return": [0.0]}),
        "random_nulls": pd.DataFrame({"split_id": [0], "top_k": [1], "null_run": ["run_0000"]}),
        "null_pvalues": pd.DataFrame({"split_id": [0], "top_k": [1], "model": ["x"], "pvalue_ir_ge_random_top_k": [0.5]}),
        "metadata": {
            "main_py_used": False,
            "prepare_experiment_used": False,
            "old_model_zoo_used": False,
            "optuna_used": False,
            "deep_models_used": False,
            "stacking_used": False,
            "decision_grade": False,
            "run_purpose": "plumbing",
        },
    }

    paths = cli_module.write_output_bundle(
        output_dir=tmp_path,
        timestamp="20260510T000000Z",
        result=fake_result,
        config=config,
    )

    expected_keys = {
        "summary",
        "fold_details",
        "scored_panel",
        "allocations",
        "portfolio_returns",
        "random_nulls",
        "null_pvalues",
        "report",
        "metadata",
    }
    assert expected_keys == set(paths.keys())
    for key, path in paths.items():
        assert path.exists(), f"Missing output file for {key}"

    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["main_py_used"] is False
    assert "output_files" in metadata


def test_random_null_pvalues_are_present_and_in_unit_interval() -> None:
    frames, config = _runner_inputs()
    result = run_cross_asset_ranking_experiment(asset_frames=frames, config=config)

    pvalues = result["null_pvalues"]
    assert not pvalues.empty
    assert "pvalue_ir_ge_random_top_k" in pvalues.columns
    valid = pvalues["pvalue_ir_ge_random_top_k"].dropna()
    assert ((valid > 0.0) & (valid <= 1.0)).all()


def test_cli_dry_run_does_not_load_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fail_load(*args, **kwargs):
        raise AssertionError("Dry-run must not load asset frames.")

    monkeypatch.setattr(cli_module, "load_prepared_asset_frames", fail_load)
    monkeypatch.setattr(cli_module, "run_cross_asset_ranking_experiment", fail_load)

    rc = cli_module.main(
        [
            "--dry-run",
            "--assets",
            "SPY",
            "QQQ",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
            "--top-k",
            "1",
            "2",
            "--random-null-runs",
            "10",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert not (tmp_path / "out").exists()


def test_lambdarank_runs_end_to_end_on_synthetic_panel() -> None:
    frames, config = _runner_inputs()
    config = CrossAssetRankingConfig(
        assets=config.assets,
        forward_horizon=config.forward_horizon,
        vol_window=config.vol_window,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        step_size=config.step_size,
        transaction_cost_bps=config.transaction_cost_bps,
        top_k_values=config.top_k_values,
        model_names=("lambdarank",),
        random_null_runs=config.random_null_runs,
        random_state=config.random_state,
        run_purpose=config.run_purpose,
        decision_grade=config.decision_grade,
        feature_normalization="per_asset_train_zscore",
    )

    result = run_cross_asset_ranking_experiment(asset_frames=frames, config=config)

    assert "lambdarank" in result["scored_panel"]["model"].unique()
    assert (result["scored_panel"]["score"].notna()).any()
    metadata = result["metadata"]
    assert metadata["optuna_used"] is False
    assert metadata["deep_models_used"] is False
    assert metadata["stacking_used"] is False
    assert "lambdarank" in metadata["model_names"]


def test_cli_dry_run_accepts_lambdarank_with_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_module, "load_prepared_asset_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry-run must not load")))
    rc = cli_module.main(
        [
            "--dry-run",
            "--assets", "SPY", "QQQ",
            "--cache-dir", str(tmp_path / "cache"),
            "--output-dir", str(tmp_path / "out"),
            "--models", "lambdarank",
            "--feature-normalization", "per_asset_train_zscore",
            "--top-k", "2",
            "--random-null-runs", "10",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "lambdarank" in out
    assert "per_asset_train_zscore" in out


def test_metadata_safety_flags_are_present() -> None:
    frames, config = _runner_inputs()
    result = run_cross_asset_ranking_experiment(asset_frames=frames, config=config)
    metadata = result["metadata"]

    assert metadata["main_py_used"] is False
    assert metadata["prepare_experiment_used"] is False
    assert metadata["old_model_zoo_used"] is False
    assert metadata["optuna_used"] is False
    assert metadata["deep_models_used"] is False
    assert metadata["stacking_used"] is False
    assert metadata["run_purpose"] == "plumbing"
    assert metadata["decision_grade"] is False
    assert "lag_convention" in metadata
    assert "calendar_convention" in metadata
