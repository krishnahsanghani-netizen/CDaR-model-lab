"""Risk and performance metrics for portfolio returns and drawdowns."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from enhanced_cdar.metrics.cdar import compute_cdar
from enhanced_cdar.metrics.drawdown import max_drawdown



def compute_volatility(portfolio_returns: pd.Series, annualization_factor: int = 252) -> float:
    """Compute annualized volatility."""
    return float(portfolio_returns.std(ddof=1) * np.sqrt(annualization_factor))



def compute_var(
    portfolio_returns: pd.Series,
    alpha: float = 0.95,
    method: str = "historical",
    dist: str = "normal",
    t_df: float | None = None,
) -> float:
    """Compute VaR as a positive loss magnitude."""
    losses = _losses(portfolio_returns)
    if method == "historical":
        q = np.quantile(losses, alpha)
        return float(max(q, 0.0))
    if method == "parametric":
        if dist == "normal":
            mu, sigma = float(losses.mean()), float(losses.std(ddof=1))
            return float(max(stats.norm.ppf(alpha, loc=mu, scale=sigma), 0.0))
        if dist == "student_t":
            df, loc, scale = _fit_student_t(losses, t_df=t_df)
            return float(max(stats.t.ppf(alpha, df=df, loc=loc, scale=scale), 0.0))
    raise ValueError("Unsupported VaR method/dist combination.")



def compute_cvar(
    portfolio_returns: pd.Series,
    alpha: float = 0.95,
    method: str = "historical",
    dist: str = "normal",
    t_df: float | None = None,
) -> float:
    """Compute CVaR (Expected Shortfall) as a positive loss magnitude."""
    losses = _losses(portfolio_returns)
    if method == "historical":
        q = np.quantile(losses, alpha)
        tail = losses[losses >= q]
        return float(max(tail.mean() if tail.size else 0.0, 0.0))

    if method == "parametric":
        if dist == "normal":
            mu, sigma = float(losses.mean()), float(losses.std(ddof=1))
            z = stats.norm.ppf(alpha)
            es = mu + sigma * stats.norm.pdf(z) / (1.0 - alpha)
            return float(max(es, 0.0))
        if dist == "student_t":
            df, loc, scale = _fit_student_t(losses, t_df=t_df)
            q = stats.t.ppf(alpha, df=df)
            numerator = stats.t.pdf(q, df=df) * (df + q**2)
            denominator = (1.0 - alpha) * max(df - 1.0, 1e-9)
            es = loc + scale * numerator / denominator
            return float(max(es, 0.0))

    raise ValueError("Unsupported CVaR method/dist combination.")



def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """Compute annualized Sharpe ratio."""
    rf_period = (1.0 + risk_free_rate_annual) ** (1.0 / annualization_factor) - 1.0
    excess = portfolio_returns - rf_period
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(annualization_factor))



def compute_sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """Compute annualized Sortino ratio."""
    rf_period = (1.0 + risk_free_rate_annual) ** (1.0 / annualization_factor) - 1.0
    excess = portfolio_returns - rf_period
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if np.isnan(downside_std) or downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(annualization_factor))



def compute_calmar_ratio(
    portfolio_returns: pd.Series,
    drawdown_series: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """Compute Calmar ratio using annualized return over max drawdown magnitude."""
    cumulative = (1.0 + portfolio_returns).prod()
    n = len(portfolio_returns)
    if n == 0:
        return 0.0
    annualized_return = cumulative ** (annualization_factor / n) - 1.0
    mdd = max_drawdown(drawdown_series)
    if mdd == 0:
        return 0.0
    return float(annualized_return / mdd)



def compute_cdar_parametric(
    drawdown_series: pd.Series,
    alpha: float = 0.95,
    dist: str = "normal",
    t_df: float | None = None,
) -> float:
    """Compute parametric CDaR approximation using drawdown magnitudes."""
    magnitudes = np.maximum(-drawdown_series.dropna().to_numpy(dtype=float), 0.0)
    if magnitudes.size == 0:
        return 0.0

    if dist == "normal":
        mu, sigma = float(magnitudes.mean()), float(magnitudes.std(ddof=1))
        z = stats.norm.ppf(alpha)
        return float(max(mu + sigma * stats.norm.pdf(z) / (1.0 - alpha), 0.0))

    if dist == "student_t":
        df, loc, scale = _fit_student_t(magnitudes, t_df=t_df)
        q = stats.t.ppf(alpha, df=df)
        numerator = stats.t.pdf(q, df=df) * (df + q**2)
        denominator = (1.0 - alpha) * max(df - 1.0, 1e-9)
        es = loc + scale * numerator / denominator
        return float(max(es, 0.0))

    raise ValueError("dist must be 'normal' or 'student_t'.")



def summarize_core_metrics(
    portfolio_returns: pd.Series,
    drawdown_series: pd.Series,
    alpha: float = 0.95,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> dict[str, float]:
    """Compute a standard report of return and risk metrics."""
    return {
        "volatility": compute_volatility(portfolio_returns, annualization_factor),
        "var": compute_var(portfolio_returns, alpha=alpha),
        "cvar": compute_cvar(portfolio_returns, alpha=alpha),
        "cdar": compute_cdar(drawdown_series, alpha=alpha),
        "max_drawdown": max_drawdown(drawdown_series),
        "sharpe": compute_sharpe_ratio(
            portfolio_returns,
            risk_free_rate_annual=risk_free_rate_annual,
            annualization_factor=annualization_factor,
        ),
        "sortino": compute_sortino_ratio(
            portfolio_returns,
            risk_free_rate_annual=risk_free_rate_annual,
            annualization_factor=annualization_factor,
        ),
        "calmar": compute_calmar_ratio(
            portfolio_returns,
            drawdown_series,
            annualization_factor=annualization_factor,
        ),
    }



def _losses(portfolio_returns: pd.Series) -> npt.NDArray[np.float64]:
    clean = np.asarray(portfolio_returns.dropna().to_numpy(dtype=float), dtype=np.float64)
    if clean.size == 0:
        raise ValueError("portfolio_returns has no valid observations.")
    out = -clean
    return np.asarray(out, dtype=np.float64)



def _fit_student_t(samples: np.ndarray, t_df: float | None = None) -> tuple[float, float, float]:
    if t_df is not None:
        loc, scale = stats.t.fit(samples, f0=t_df)[1:]
        return float(t_df), float(loc), float(scale)
    df, loc, scale = stats.t.fit(samples)
    return float(df), float(loc), float(scale)
