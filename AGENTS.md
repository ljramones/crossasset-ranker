# Repository Guidelines

## Project Structure & Module Organization
This repository is a modular Python research framework managed with `uv`. Runtime configuration lives in [`config/config.yaml`](/Users/larrym/prediction/config/config.yaml), the pipeline entry point is [`main.py`](/Users/larrym/prediction/main.py), and code is split across `data/`, `features/`, `models/`, `evaluation/`, and `utils/`. Keep notebooks in `notebooks/` and unit tests in `tests/`. Avoid putting reusable research logic directly in notebooks or `main.py`.

## Build, Test, and Development Commands
Use `uv` for local setup and execution:

- `uv sync` installs dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py` runs the full signal-exploration pipeline.
- `uv run python -m pytest` runs the unit test suite.
- `uv run python -m pytest tests/test_features.py` targets feature engineering only.
- `uv run jupyter lab` opens the notebook environment for exploration.

Target Python is `3.12` as pinned in `.python-version`.

## Coding Style & Naming Conventions
Follow standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and lowercase module names. Add type hints to public functions and keep feature, validation, and evaluation logic deterministic. Use comments sparingly to explain quant-specific assumptions such as leakage prevention or signal construction.

## Testing Guidelines
Use `pytest` with files named `test_<module>.py` and functions named `test_<behavior>`. Cover new feature engineering, walk-forward logic, and metrics changes with deterministic fixtures. For model code, prefer testing interfaces, shape contracts, and leakage guards rather than brittle performance assertions.

## Commit & Pull Request Guidelines
This repository has no established commit history yet, so start with imperative, concise commit subjects such as `Add walk-forward evaluation` or `Implement LightGBM baseline`. Keep pull requests small and include:

- a short description of the change,
- any setup or verification commands run,
- linked issues if applicable,
- screenshots or sample output when behavior changes are user-visible.

## Environment & Data Notes
Do not commit `.venv/`, cached market data, generated plots, model artifacts, or large datasets. Keep raw and derived datasets out of Git unless they are tiny fixtures needed for tests.
