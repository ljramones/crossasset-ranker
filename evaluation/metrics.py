"""Classification and trading metrics used in walk-forward evaluation.

This module now exposes both legacy summary fields and richer active-return /
turnover diagnostics. The legacy ``trade_frequency`` field is retained for
backward compatibility, but it is semantically the fraction of bars with a
non-zero executed position and should be treated as a deprecated alias for
``fraction_in_market`` in future reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def active_return_series(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """Return the aligned active-return series."""

    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name="active_return")
    return (aligned["strategy"] - aligned["benchmark"]).rename("active_return")


def annualized_active_return(active_returns: pd.Series, annualization: int = 252) -> float:
    """Return the annualized mean active return."""

    if len(active_returns) == 0:
        return 0.0
    return float(active_returns.mean() * annualization)


def tracking_error(active_returns: pd.Series, annualization: int = 252) -> float:
    """Return annualized tracking error from an active-return stream."""

    std = float(active_returns.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(std * np.sqrt(annualization))


def information_ratio_from_active_returns(active_returns: pd.Series, annualization: int = 252) -> float:
    """Return the information ratio from active returns."""

    error = tracking_error(active_returns, annualization=annualization)
    if error == 0.0:
        return 0.0
    return annualized_active_return(active_returns, annualization=annualization) / error


def active_max_drawdown(active_returns: pd.Series) -> float:
    """Return max drawdown of the active equity curve."""

    if len(active_returns) == 0:
        return 0.0
    return float(_max_drawdown(compute_equity_curve(active_returns)))


def active_calmar(active_returns: pd.Series, annualization: int = 252) -> float:
    """Return active Calmar based on active-return equity drawdown."""

    max_dd = abs(active_max_drawdown(active_returns))
    if max_dd == 0.0:
        return 0.0
    return annualized_active_return(active_returns, annualization=annualization) / max_dd


def benchmark_correlation(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Return return-stream correlation to the benchmark."""

    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return 0.0
    strategy_std = float(aligned["strategy"].std())
    benchmark_std = float(aligned["benchmark"].std())
    if strategy_std == 0.0 or benchmark_std == 0.0 or np.isnan(strategy_std) or np.isnan(benchmark_std):
        return 0.0
    correlation = float(aligned["strategy"].corr(aligned["benchmark"]))
    return 0.0 if np.isnan(correlation) else correlation


def beta_to_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Return simple beta estimated from aligned strategy and benchmark returns."""

    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return 0.0
    benchmark_variance = float(aligned["benchmark"].var())
    if benchmark_variance == 0.0 or np.isnan(benchmark_variance):
        return 0.0
    covariance = float(aligned["strategy"].cov(aligned["benchmark"]))
    if np.isnan(covariance):
        return 0.0
    return covariance / benchmark_variance


def alpha_after_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization: int = 252,
) -> float:
    """Return annualized intercept-style alpha after removing benchmark beta."""

    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return 0.0
    beta = beta_to_benchmark(aligned["strategy"], aligned["benchmark"])
    alpha_daily = float(aligned["strategy"].mean() - beta * aligned["benchmark"].mean())
    return alpha_daily * annualization


def compute_classification_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics for binary direction prediction."""

    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc_roc = 0.5

    return {
        "directional_accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": auc_roc,
    }


def compute_trading_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    signal: pd.Series,
    annualization_factor: int,
    transaction_cost_bps: float,
) -> dict[str, float]:
    """Compute gross and net trading metrics from the same signal stream."""

    gross_strategy_returns = compute_strategy_returns(
        returns=returns,
        signal=signal,
        transaction_cost_bps=0.0,
    )
    net_strategy_returns = compute_strategy_returns(
        returns=returns,
        signal=signal,
        transaction_cost_bps=transaction_cost_bps,
    )
    active_returns = active_return_series(net_strategy_returns, benchmark_returns)
    equity_curve = compute_equity_curve(net_strategy_returns)
    gross_equity_curve = compute_equity_curve(gross_strategy_returns)
    executed_position = signal.shift(1).fillna(0.0).rename("position")
    turnover = compute_signal_turnover(executed_position)
    fraction_in_market = float(executed_position.ne(0.0).mean())
    average_long_exposure = float(executed_position.clip(lower=0.0).mean())
    average_short_exposure = float((-executed_position.clip(upper=0.0)).mean())
    average_abs_position = float(executed_position.abs().mean())
    average_position_size = average_abs_position
    fraction_positive_predictions = float(signal.mean())
    benchmark_sharpe = _annualized_sharpe(benchmark_returns, annualization_factor)
    gross_sharpe = _annualized_sharpe(gross_strategy_returns, annualization_factor)
    gross_sortino = _annualized_sortino(gross_strategy_returns, annualization_factor)
    net_sharpe = _annualized_sharpe(net_strategy_returns, annualization_factor)
    net_sortino = _annualized_sortino(net_strategy_returns, annualization_factor)
    annualized_active = annualized_active_return(active_returns, annualization=annualization_factor)
    active_tracking_error = tracking_error(active_returns, annualization=annualization_factor)
    info_ratio = information_ratio_from_active_returns(active_returns, annualization=annualization_factor)
    gross_total_return = float(gross_equity_curve.iloc[-1] - 1.0)
    net_total_return = float(equity_curve.iloc[-1] - 1.0)
    cost_drag = gross_total_return - net_total_return
    turnover_mean = float(turnover.mean())
    active_return_for_cost = abs(annualized_active)
    cost_per_unit_active = float(cost_drag / active_return_for_cost) if active_return_for_cost > 0.0 else float("nan")

    return {
        "sharpe": gross_sharpe,
        "sortino": gross_sortino,
        "net_sharpe": net_sharpe,
        "net_sortino": net_sortino,
        "benchmark_sharpe": benchmark_sharpe,
        "excess_net_sharpe": net_sharpe - benchmark_sharpe,
        "sharpe_cost_delta": gross_sharpe - net_sharpe,
        "sortino_cost_delta": gross_sortino - net_sortino,
        "calmar": _calmar_ratio(net_strategy_returns, annualization_factor),
        "information_ratio": info_ratio,
        "max_drawdown": float(_max_drawdown(equity_curve)),
        "annualized_active_return": annualized_active,
        "active_volatility": active_tracking_error,
        "tracking_error": active_tracking_error,
        "active_max_drawdown": active_max_drawdown(active_returns),
        "active_calmar": active_calmar(active_returns, annualization=annualization_factor),
        "correlation_to_benchmark": benchmark_correlation(net_strategy_returns, benchmark_returns),
        "beta_to_benchmark": beta_to_benchmark(net_strategy_returns, benchmark_returns),
        "alpha_after_beta": alpha_after_beta(
            net_strategy_returns,
            benchmark_returns,
            annualization=annualization_factor,
        ),
        "trade_frequency": fraction_in_market,
        "fraction_in_market": fraction_in_market,
        "average_long_exposure": average_long_exposure,
        "average_short_exposure": average_short_exposure,
        "average_abs_position": average_abs_position,
        "average_position_size": average_position_size,
        "fraction_positive_predictions": fraction_positive_predictions,
        "position_flip_count": float(position_flip_count(executed_position)),
        "daily_turnover": turnover_mean,
        "annualized_turnover": turnover_mean * annualization_factor,
        "average_holding_period_days": average_holding_period(executed_position),
        "round_trip_count": float(round_trip_count(executed_position)),
        "cost_drag": cost_drag,
        "cost_per_unit_active_return": cost_per_unit_active,
        "total_return": net_total_return,
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
    }


def compute_return_stream_metrics(
    net_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int,
    gross_returns: pd.Series | None = None,
    trade_frequency: float | None = None,
    turnover: float | None = None,
    position: pd.Series | None = None,
    prediction: pd.Series | None = None,
) -> dict[str, float]:
    """Compute portfolio metrics from precomputed gross/net return streams."""

    gross_series = gross_returns if gross_returns is not None else net_returns
    active_returns = active_return_series(net_returns, benchmark_returns)
    net_equity_curve = compute_equity_curve(net_returns)
    gross_equity_curve = compute_equity_curve(gross_series)
    benchmark_sharpe = _annualized_sharpe(benchmark_returns, annualization_factor)
    fraction_in_market = float(0.0 if trade_frequency is None else trade_frequency)
    daily_turnover = float(0.0 if turnover is None else turnover)
    annualized_active = annualized_active_return(active_returns, annualization=annualization_factor)
    active_tracking_error = tracking_error(active_returns, annualization=annualization_factor)
    net_total_return = float(net_equity_curve.iloc[-1] - 1.0)
    gross_total_return = float(gross_equity_curve.iloc[-1] - 1.0)
    cost_drag = gross_total_return - net_total_return
    active_return_for_cost = abs(annualized_active)
    cost_per_unit_active = float(cost_drag / active_return_for_cost) if active_return_for_cost > 0.0 else float("nan")

    if position is not None:
        executed_position = position.astype(float)
        average_long_exposure = float(executed_position.clip(lower=0.0).mean())
        average_short_exposure = float((-executed_position.clip(upper=0.0)).mean())
        average_abs_position = float(executed_position.abs().mean())
        flip_count = float(position_flip_count(executed_position))
        holding_period = average_holding_period(executed_position)
        trips = float(round_trip_count(executed_position))
    else:
        average_long_exposure = float("nan")
        average_short_exposure = float("nan")
        average_abs_position = float("nan")
        flip_count = float("nan")
        holding_period = float("nan")
        trips = float("nan")

    fraction_positive_predictions = float(prediction.mean()) if prediction is not None else float("nan")

    return {
        "sharpe": _annualized_sharpe(gross_series, annualization_factor),
        "sortino": _annualized_sortino(gross_series, annualization_factor),
        "net_sharpe": _annualized_sharpe(net_returns, annualization_factor),
        "net_sortino": _annualized_sortino(net_returns, annualization_factor),
        "benchmark_sharpe": benchmark_sharpe,
        "excess_net_sharpe": _annualized_sharpe(net_returns, annualization_factor) - benchmark_sharpe,
        "sharpe_cost_delta": _annualized_sharpe(gross_series, annualization_factor)
        - _annualized_sharpe(net_returns, annualization_factor),
        "sortino_cost_delta": _annualized_sortino(gross_series, annualization_factor)
        - _annualized_sortino(net_returns, annualization_factor),
        "calmar": _calmar_ratio(net_returns, annualization_factor),
        "information_ratio": information_ratio_from_active_returns(active_returns, annualization=annualization_factor),
        "max_drawdown": float(_max_drawdown(net_equity_curve)),
        "annualized_active_return": annualized_active,
        "active_volatility": active_tracking_error,
        "tracking_error": active_tracking_error,
        "active_max_drawdown": active_max_drawdown(active_returns),
        "active_calmar": active_calmar(active_returns, annualization=annualization_factor),
        "correlation_to_benchmark": benchmark_correlation(net_returns, benchmark_returns),
        "beta_to_benchmark": beta_to_benchmark(net_returns, benchmark_returns),
        "alpha_after_beta": alpha_after_beta(net_returns, benchmark_returns, annualization=annualization_factor),
        "trade_frequency": fraction_in_market,
        "fraction_in_market": fraction_in_market,
        "turnover": daily_turnover,
        "average_long_exposure": average_long_exposure,
        "average_short_exposure": average_short_exposure,
        "average_abs_position": average_abs_position,
        "average_position_size": average_abs_position,
        "fraction_positive_predictions": fraction_positive_predictions,
        "position_flip_count": flip_count,
        "daily_turnover": daily_turnover,
        "annualized_turnover": daily_turnover * annualization_factor,
        "average_holding_period_days": holding_period,
        "round_trip_count": trips,
        "cost_drag": cost_drag,
        "cost_per_unit_active_return": cost_per_unit_active,
        "total_return": net_total_return,
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
    }


def compute_strategy_returns(
    returns: pd.Series,
    signal: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    """Translate binary signals into strategy returns with cost on position flips."""

    signal_shifted = signal.shift(1).fillna(0.0).rename("position")
    turnover = compute_signal_turnover(signal_shifted)
    transaction_cost = turnover * (transaction_cost_bps / 10_000.0)
    return signal_shifted * returns - transaction_cost


def compute_signal_turnover(position: pd.Series) -> pd.Series:
    """Compute per-bar turnover from the executed position stream."""

    return position.diff().abs().fillna(position.abs()).rename("turnover")


def position_flip_count(position: pd.Series) -> int:
    """Count position changes after the first non-zero position is established."""

    executed_position = position.fillna(0.0)
    nonzero_positions = np.flatnonzero(executed_position.to_numpy() != 0.0)
    if len(nonzero_positions) == 0:
        return 0
    active_slice = executed_position.iloc[nonzero_positions[0] :]
    return int(active_slice.diff().fillna(0.0).ne(0.0).iloc[1:].sum())


def average_holding_period(position: pd.Series) -> float:
    """Return the mean run length of non-zero executed positions."""

    executed_position = position.fillna(0.0).ne(0.0).astype(int)
    if executed_position.sum() == 0:
        return 0.0
    run_lengths: list[int] = []
    current = 0
    for value in executed_position:
        if value:
            current += 1
        elif current:
            run_lengths.append(current)
            current = 0
    if current:
        run_lengths.append(current)
    if not run_lengths:
        return 0.0
    return float(np.mean(run_lengths))


def round_trip_count(position: pd.Series) -> int:
    """Count completed non-zero exposure runs that return to flat."""

    executed_position = position.fillna(0.0)
    previous_nonzero = executed_position.shift(1).fillna(0.0).ne(0.0)
    current_flat = executed_position.eq(0.0)
    return int((previous_nonzero & current_flat).sum())


def compute_equity_curve(strategy_returns: pd.Series) -> pd.Series:
    """Convert a return stream into a cumulative equity curve."""

    return (1.0 + strategy_returns).cumprod().rename("equity_curve")


def _annualized_sharpe(returns: pd.Series, annualization_factor: int) -> float:
    std = float(returns.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(annualization_factor) * returns.mean() / std)


def _annualized_sortino(returns: pd.Series, annualization_factor: int) -> float:
    downside = returns[returns < 0.0]
    downside_std = float(downside.std())
    if downside_std == 0.0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(annualization_factor) * returns.mean() / downside_std)


def _calmar_ratio(returns: pd.Series, annualization_factor: int) -> float:
    equity_curve = (1.0 + returns).cumprod()
    max_dd = abs(_max_drawdown(equity_curve))
    if max_dd == 0.0:
        return 0.0
    if len(returns) == 0:
        return 0.0
    total_years = max(len(returns) / annualization_factor, 1.0 / annualization_factor)
    terminal_equity = float(equity_curve.iloc[-1])
    if terminal_equity <= 0.0:
        annualized_return = -1.0
    else:
        annualized_return = float(terminal_equity ** (1.0 / total_years) - 1.0)
    return annualized_return / max_dd


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())
