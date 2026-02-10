"""Conditional Drawdown at Risk computations."""

from __future__ import annotations

import numpy as np
import pandas as pd



def compute_cdar(drawdown_series: pd.Series, alpha: float = 0.95) -> float:
    """Compute empirical CDaR from drawdown series as positive magnitude."""
    _validate_alpha(alpha)
    x = _to_magnitudes(drawdown_series)
    if x.size == 0:
        raise ValueError("drawdown_series has no valid observations.")

    threshold = float(np.quantile(x, alpha))
    tail = x[x >= threshold]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail))



def compute_rolling_cdar(
    drawdown_series: pd.Series,
    alpha: float = 0.95,
    window: int = 63,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute rolling CDaR; output values are positive magnitudes."""
    _validate_alpha(alpha)
    if window <= 1:
        raise ValueError("window must be > 1.")

    min_obs = min_periods if min_periods is not None else window

    def _rolling(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        threshold = float(np.quantile(arr, alpha))
        tail = arr[arr >= threshold]
        return float(np.mean(tail)) if tail.size else 0.0

    magnitudes = pd.Series(_to_magnitudes(drawdown_series), index=drawdown_series.index)
    rolling = magnitudes.rolling(window=window, min_periods=min_obs).apply(_rolling, raw=True)
    rolling.name = f"rolling_cdar_{alpha:.2f}"
    return rolling



def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")



def _to_magnitudes(drawdown_series: pd.Series) -> np.ndarray:
    if drawdown_series.empty:
        raise ValueError("drawdown_series is empty.")
    clean = drawdown_series.dropna().to_numpy(dtype=float)
    return np.maximum(-clean, 0.0)
