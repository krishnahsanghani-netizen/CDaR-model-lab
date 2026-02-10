"""Portfolio backtesting with optional rebalancing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from enhanced_cdar.metrics.cdar import compute_cdar
from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    compute_portfolio_returns,
)
from enhanced_cdar.metrics.risk_metrics import summarize_core_metrics
from enhanced_cdar.portfolio.weights import validate_weights


@dataclass(frozen=True)
class BacktestResult:
    """Backtest result container."""

    portfolio_returns: pd.Series
    portfolio_values: pd.Series
    drawdown: pd.Series
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float] | None



def run_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    rebalance_calendar: str = "none",
    rebalance_every_n_periods: int | None = None,
    rebalance_mode: str = "fixed",
    benchmark_returns: pd.Series | None = None,
    alpha: float = 0.95,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.0,
    no_short: bool = True,
    gross_exposure_limit: float | None = None,
    dynamic_optimizer: Callable[[pd.DataFrame], np.ndarray] | None = None,
) -> BacktestResult:
    """Run backtest under close-to-close execution assumption."""
    if returns.empty:
        raise ValueError("returns is empty.")

    base_weights = np.asarray(weights, dtype=float)
    validate_weights(
        base_weights,
        no_short=no_short,
        gross_exposure_limit=gross_exposure_limit,
    )

    if rebalance_mode == "dynamic" and dynamic_optimizer is None:
        raise ValueError("dynamic rebalancing requires dynamic_optimizer callback.")

    schedule = _rebalance_flags(
        returns.index,
        rebalance_calendar=rebalance_calendar,
        rebalance_every_n_periods=rebalance_every_n_periods,
    )

    portfolio = []
    current_weights = base_weights.copy()

    for idx, timestamp in enumerate(returns.index):
        if schedule.iloc[idx]:
            if rebalance_mode == "dynamic":
                current_weights = np.asarray(
                    dynamic_optimizer(returns.iloc[: idx + 1]),
                    dtype=float,
                )
                validate_weights(
                    current_weights,
                    no_short=no_short,
                    gross_exposure_limit=gross_exposure_limit,
                )
            else:
                current_weights = base_weights
        row = returns.loc[timestamp].to_numpy(dtype=float)
        portfolio.append(float(np.dot(row, current_weights)))

    portfolio_returns = pd.Series(portfolio, index=returns.index, name="portfolio_return")
    portfolio_values = compute_cumulative_value(portfolio_returns, initial_value=1.0)
    drawdown = compute_drawdown_curve(portfolio_values)

    metrics = summarize_core_metrics(
        portfolio_returns,
        drawdown,
        alpha=alpha,
        annualization_factor=annualization_factor,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    benchmark_metrics: dict[str, float] | None = None
    if benchmark_returns is not None:
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns.rename("benchmark")],
            axis=1,
        ).dropna()
        active_returns = aligned["portfolio_return"] - aligned["benchmark"]
        active_values = compute_cumulative_value(active_returns, initial_value=1.0)
        active_drawdown = compute_drawdown_curve(active_values)
        tracking_error = float(active_returns.std(ddof=1) * np.sqrt(annualization_factor))
        information_ratio = (
            float(
                active_returns.mean()
                / active_returns.std(ddof=1)
                * np.sqrt(annualization_factor)
            )
            if active_returns.std(ddof=1) != 0
            else 0.0
        )
        benchmark_metrics = {
            "active_return_mean": float(active_returns.mean()),
            "active_cdar": compute_cdar(active_drawdown, alpha=alpha),
            "active_max_drawdown": float((-active_drawdown.min()).clip(min=0.0)),
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
        }

    return BacktestResult(
        portfolio_returns=portfolio_returns,
        portfolio_values=portfolio_values,
        drawdown=drawdown,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
    )



def _rebalance_flags(
    index: pd.DatetimeIndex,
    rebalance_calendar: str = "none",
    rebalance_every_n_periods: int | None = None,
) -> pd.Series:
    """Create rebalance flags with first point always true."""
    flags = pd.Series(False, index=index)
    if len(index) == 0:
        return flags

    flags.iloc[0] = True

    if rebalance_every_n_periods is not None:
        if rebalance_every_n_periods <= 0:
            raise ValueError("rebalance_every_n_periods must be positive.")
        flags.iloc[::rebalance_every_n_periods] = True

    cal = rebalance_calendar.upper()
    if cal == "M":
        month_changed = index.to_series().dt.to_period("M").ne(
            index.to_series().dt.to_period("M").shift(1)
        )
        flags |= month_changed
    elif cal == "Q":
        quarter_changed = index.to_series().dt.to_period("Q").ne(
            index.to_series().dt.to_period("Q").shift(1)
        )
        flags |= quarter_changed
    elif cal not in {"NONE", ""}:
        raise ValueError("rebalance_calendar must be one of: none, M, Q")

    return flags
