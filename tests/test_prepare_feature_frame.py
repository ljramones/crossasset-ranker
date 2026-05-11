"""Tests for the active-track prepare_feature_frame CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import prepare_feature_frame
from scripts.prepare_feature_frame import (
    main as prepare_main,
    prepare_single_asset_feature_frame,
    validate_prepared_feature_frame,
)


def _synthetic_market_frame(rows: int = 600) -> pd.DataFrame:
    dates = pd.date_range("2018-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    asset_returns = rng.normal(0.0004, 0.01, size=rows)
    asset_close = 100.0 * np.exp(np.cumsum(asset_returns))
    bench_returns = rng.normal(0.0003, 0.009, size=rows)
    bench_close = 400.0 * np.exp(np.cumsum(bench_returns))
    vix_close = 18.0 + 4.0 * np.sin(np.linspace(0, 8 * np.pi, rows))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": asset_close * 0.999,
            "High": asset_close * 1.005,
            "Low": asset_close * 0.995,
            "Close": asset_close,
            "Adj Close": asset_close,
            "Volume": rng.integers(1_000_000, 2_000_000, size=rows),
            "BenchmarkClose": bench_close,
            "VIXClose": vix_close,
        }
    )


def test_dry_run_does_not_fetch_or_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fetch_called = {"count": 0}

    def fail_if_called(*args, **kwargs):
        fetch_called["count"] += 1
        raise AssertionError("Dry-run must not invoke build_asset_cache_frame.")

    monkeypatch.setattr(prepare_feature_frame, "build_asset_cache_frame", fail_if_called)

    output_csv = tmp_path / "should_not_exist.csv"
    rc = prepare_main(
        [
            "--dry-run",
            "--ticker",
            "SPY",
            "--start-date",
            "2010-01-01",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-csv",
            str(output_csv),
            "--include-drawdown-labels",
            "--horizons",
            "10",
            "20",
            "--thresholds",
            "-0.02",
            "-0.03",
        ]
    )

    assert rc == 0
    assert fetch_called["count"] == 0
    assert not output_csv.exists()
    assert not (tmp_path / "cache").exists()
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "ticker:" in captured.out


def test_prepared_frame_contains_required_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        prepare_feature_frame,
        "build_asset_cache_frame",
        lambda *args, **kwargs: _synthetic_market_frame(),
    )

    output_csv = tmp_path / "spy_features.csv"
    frame = prepare_single_asset_feature_frame(
        ticker="SPY",
        cache_dir=tmp_path / "cache",
        include_drawdown_labels=True,
        horizons=(10, 20),
        thresholds=(-0.02, -0.03, -0.05),
        output_csv=output_csv,
    )

    required = {
        "date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "BenchmarkClose",
        "return_1d",
        "return_5d",
        "return_20d",
        "vol_ratio",
        "momentum_norm",
        "volume_zscore",
        "range_norm",
        "sma_ratio",
        "realized_vol_20",
        "benchmark_return_1d",
        "forward_simple_return_1d",
    }
    missing = required - set(frame.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"
    assert output_csv.exists()
    on_disk = pd.read_csv(output_csv)
    assert "date" in on_disk.columns
    assert len(on_disk) == len(frame)


def test_drawdown_labels_are_appended(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        prepare_feature_frame,
        "build_asset_cache_frame",
        lambda *args, **kwargs: _synthetic_market_frame(),
    )

    frame = prepare_single_asset_feature_frame(
        ticker="SPY",
        cache_dir=tmp_path / "cache",
        include_drawdown_labels=True,
        horizons=(10, 20),
        thresholds=(-0.02, -0.03, -0.05),
    )

    expected_targets = {
        "target_drawdown_event_10d_2pct",
        "target_drawdown_event_10d_3pct",
        "target_drawdown_event_10d_5pct",
        "target_drawdown_event_20d_2pct",
        "target_drawdown_event_20d_3pct",
        "target_drawdown_event_20d_5pct",
    }
    assert expected_targets.issubset(set(frame.columns))
    valid = frame["target_drawdown_event_20d_3pct"].dropna()
    assert valid.isin([0, 1]).all()


def test_validate_reports_missing_required_columns() -> None:
    bad_frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "return_1d": [0.0, 0.001, -0.002],
        }
    )

    summary = validate_prepared_feature_frame(bad_frame)

    assert summary["is_valid"] is False
    assert "Adj Close" in summary["missing_required_columns"]
    assert "benchmark_return_1d" in summary["missing_required_columns"]
    assert summary["row_count"] == 3


def test_raw_price_columns_are_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        prepare_feature_frame,
        "build_asset_cache_frame",
        lambda *args, **kwargs: _synthetic_market_frame(),
    )

    frame = prepare_single_asset_feature_frame(
        ticker="SPY",
        cache_dir=tmp_path / "cache",
        include_drawdown_labels=False,
    )

    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume", "BenchmarkClose"):
        assert column in frame.columns


def test_target_columns_are_present_when_drawdown_labels_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        prepare_feature_frame,
        "build_asset_cache_frame",
        lambda *args, **kwargs: _synthetic_market_frame(),
    )

    frame = prepare_single_asset_feature_frame(
        ticker="SPY",
        cache_dir=tmp_path / "cache",
        include_drawdown_labels=True,
        horizons=(20,),
        thresholds=(-0.03,),
    )

    summary = validate_prepared_feature_frame(frame)
    assert "target_drawdown_event_20d_3pct" in summary["drawdown_label_columns"]
    assert summary["drawdown_label_columns"], "Drawdown labels must appear in the validation summary."


def test_script_does_not_import_legacy_data_market_data() -> None:
    source = Path(prepare_feature_frame.__file__).read_text(encoding="utf-8")
    assert "from data.market_data" not in source, "Active-track script must not import the legacy module."
    assert "import data.market_data" not in source
    assert "from utils.experiment" not in source
    assert "from audit.integrity_audit" not in source
    assert "import main" not in source
    assert "from main " not in source
