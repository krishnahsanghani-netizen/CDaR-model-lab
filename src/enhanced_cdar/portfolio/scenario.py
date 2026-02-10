"""Scenario analysis utilities for stress-testing portfolios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    compute_portfolio_returns,
)
from enhanced_cdar.metrics.risk_metrics import summarize_core_metrics



def build_scenario_returns(
    returns: pd.DataFrame,
    asset_shocks: dict[str, float] | None = None,
    global_shift: float = 0.0,
) -> pd.DataFrame:
    """Apply additive shocks to returns to construct a scenario return matrix."""
    scenario = returns.copy()
    if global_shift != 0.0:
        scenario = scenario + global_shift

    if asset_shocks:
        missing = [asset for asset in asset_shocks if asset not in scenario.columns]
        if missing:
            raise ValueError(f"Scenario assets not found in returns: {missing}")
        for asset, shock in asset_shocks.items():
            scenario[asset] = scenario[asset] + float(shock)

    return scenario



def evaluate_portfolio_scenarios(
    returns: pd.DataFrame,
    weights: np.ndarray,
    scenarios: dict[str, dict[str, Any]],
    alpha: float = 0.95,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> pd.DataFrame:
    """Evaluate baseline and shocked scenarios for one portfolio."""
    if returns.empty:
        raise ValueError("returns is empty.")
    if len(weights) != returns.shape[1]:
        raise ValueError("weights length must match number of assets.")

    rows: list[dict[str, Any]] = []

    baseline_port = compute_portfolio_returns(returns, weights)
    baseline_values = compute_cumulative_value(baseline_port)
    baseline_dd = compute_drawdown_curve(baseline_values)
    baseline_metrics = summarize_core_metrics(
        baseline_port,
        baseline_dd,
        alpha=alpha,
        annualization_factor=annualization_factor,
        risk_free_rate_annual=risk_free_rate_annual,
    )
    rows.append({"scenario": "baseline", **baseline_metrics})

    for scenario_name, spec in scenarios.items():
        shock_returns = build_scenario_returns(
            returns,
            asset_shocks=spec.get("asset_shocks"),
            global_shift=float(spec.get("global_shift", 0.0)),
        )
        port = compute_portfolio_returns(shock_returns, weights)
        values = compute_cumulative_value(port)
        dd = compute_drawdown_curve(values)
        metrics = summarize_core_metrics(
            port,
            dd,
            alpha=alpha,
            annualization_factor=annualization_factor,
            risk_free_rate_annual=risk_free_rate_annual,
        )
        rows.append({"scenario": scenario_name, **metrics})

    return pd.DataFrame(rows)



def preset_scenarios() -> dict[str, dict[str, Any]]:
    """Return built-in scenario presets."""
    return {
        "global_down_2pct": {"global_shift": -0.02},
        "global_down_5pct": {"global_shift": -0.05},
    }
