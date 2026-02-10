"""Regime/subperiod analysis utilities for portfolio risk metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    compute_portfolio_returns,
)
from enhanced_cdar.metrics.risk_metrics import summarize_core_metrics

RegimeFrequency = Literal["Y", "Q", "M"]



def compute_regime_metrics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    regime_frequency: RegimeFrequency = "Y",
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.0,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Compute metrics across time regimes (e.g., yearly, quarterly, monthly)."""
    if returns.empty:
        raise ValueError("returns is empty.")
    if len(weights) != returns.shape[1]:
        raise ValueError("weights length must match number of assets.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex for regime analysis.")

    grouped = returns.groupby(returns.index.to_period(regime_frequency))
    rows: list[dict[str, float | str | int]] = []

    for period, block in grouped:
        block_clean = block.dropna(how="any")
        if len(block_clean) < min_periods:
            continue

        port = compute_portfolio_returns(block_clean, weights)
        values = compute_cumulative_value(port)
        drawdown = compute_drawdown_curve(values)
        metrics = summarize_core_metrics(
            port,
            drawdown,
            alpha=alpha,
            annualization_factor=annualization_factor,
            risk_free_rate_annual=risk_free_rate_annual,
        )
        rows.append(
            {
                "regime": str(period),
                "n_periods": int(len(block_clean)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)



def compare_regime_cdar(regime_df: pd.DataFrame) -> dict[str, float | str]:
    """Summarize best/worst CDaR regime from regime metric DataFrame."""
    if regime_df.empty:
        return {
            "best_regime": "",
            "best_cdar": np.nan,
            "worst_regime": "",
            "worst_cdar": np.nan,
        }

    idx_best = regime_df["cdar"].idxmin()
    idx_worst = regime_df["cdar"].idxmax()
    return {
        "best_regime": str(regime_df.loc[idx_best, "regime"]),
        "best_cdar": float(regime_df.loc[idx_best, "cdar"]),
        "worst_regime": str(regime_df.loc[idx_worst, "regime"]),
        "worst_cdar": float(regime_df.loc[idx_worst, "cdar"]),
    }
