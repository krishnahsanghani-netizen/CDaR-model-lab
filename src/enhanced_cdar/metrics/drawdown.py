"""Return, portfolio, and drawdown computations."""

from __future__ import annotations

import numpy as np
import pandas as pd



def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute asset returns from price matrix."""
    if prices.empty:
        raise ValueError("Input prices DataFrame is empty.")
    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'.")

    if method == "simple":
        returns = prices.pct_change()
    else:
        returns = np.log(prices / prices.shift(1))

    return returns.dropna(how="all")



def compute_portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute portfolio return time series from asset returns and weights."""
    if returns.empty:
        raise ValueError("Input returns DataFrame is empty.")
    if len(weights) != returns.shape[1]:
        raise ValueError("weights length must match number of return columns.")

    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("weights must contain finite values.")

    series = returns.to_numpy() @ w
    return pd.Series(series, index=returns.index, name="portfolio_return")



def compute_cumulative_value(portfolio_returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Compute cumulative portfolio value series with multiplicative compounding."""
    if portfolio_returns.empty:
        raise ValueError("portfolio_returns is empty.")
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")

    values = (1.0 + portfolio_returns).cumprod() * initial_value
    values.name = "portfolio_value"
    return values



def compute_drawdown_curve(values: pd.Series) -> pd.Series:
    """Compute drawdown series (0 at peak, negative when underwater)."""
    if values.empty:
        raise ValueError("values series is empty.")
    if (values <= 0).any():
        raise ValueError("values must be strictly positive to compute drawdown ratios.")

    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    drawdown.name = "drawdown"
    return drawdown



def max_drawdown(drawdown_series: pd.Series) -> float:
    """Return maximum drawdown as positive magnitude."""
    if drawdown_series.empty:
        raise ValueError("drawdown_series is empty.")
    return float((-drawdown_series.min()).clip(min=0.0))
