# ml-trading-signal-exploration

Modular research framework for comparing machine learning architectures for trading signals on stationary features. The project starts with daily SPY data, applies strict walk-forward validation, and reports both classification and trading performance.

## What this project does

- builds stationary features such as log returns, volatility ratios, normalized momentum, volume z-scores, range normalization, and SMA ratios,
- runs Augmented Dickey-Fuller checks for each engineered feature,
- evaluates models with time-ordered walk-forward splits,
- compares models using Sharpe, Sortino, Calmar, Information Ratio, directional accuracy, AUC-ROC, max drawdown, and total return.

## Structure

- `config/config.yaml`: pipeline, model, and validation settings
- `data/`: market data loading helpers and optional cache files
- `features/`: stationary feature engineering and ADF checks
- `models/`: common model interface and baseline architectures
- `evaluation/`: metrics and walk-forward validation
- `utils/`: config loading, reporting, and reproducibility
- `notebooks/01_exploration.ipynb`: starter notebook for inspection
- `tests/`: unit tests for core framework logic

## Quick start

```bash
uv sync
uv run python -m pytest
uv run python main.py
```

If you prefer `pip`, install from `requirements.txt`. The default configuration downloads daily `SPY` data with `yfinance` and caches it at `data/spy_daily.csv`.

## Financial ML safeguards

- No raw prices are used as features or targets.
- Features and targets are constructed from lagged or contemporaneous information only.
- Validation is strictly walk-forward, with no shuffled rows or pooled future data.
- Metrics are computed on held-out test windows after model fitting on prior windows only.
