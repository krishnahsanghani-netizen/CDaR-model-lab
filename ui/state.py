"""Typed Streamlit session state models for CDaR Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class UIState:
    """Session-backed state container for UI workflow."""

    prices_df: pd.DataFrame | None = None
    returns_df: pd.DataFrame | None = None
    benchmark_prices: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    weights: np.ndarray | None = None
    analysis_results: dict[str, Any] = field(default_factory=dict)
    equity_curve: pd.Series | None = None
    drawdown_series: pd.Series | None = None
    rolling_cdar: pd.Series | None = None
    frontier_df: pd.DataFrame | None = None
    surface_df: pd.DataFrame | None = None
    run_dir: str | None = None
