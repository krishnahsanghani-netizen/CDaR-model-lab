"""CDaR-focused portfolio optimization routines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import cvxpy as cp
import joblib
import numpy as np
import pandas as pd

from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    compute_portfolio_returns,
)
from enhanced_cdar.metrics.risk_metrics import summarize_core_metrics
from enhanced_cdar.portfolio.weights import default_bounds

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    cdar: float
    expected_return: float
    other_metrics: dict[str, float]
    status: str



def optimize_portfolio_cdar(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    no_short: bool = True,
    weight_bounds: tuple[float, float] | None = (0.0, 1.0),
    target_return: float | None = None,
    target_cdar: float | None = None,
    solver: str | None = None,
    gross_exposure_limit: float | None = None,
    per_asset_lower: np.ndarray | None = None,
    per_asset_upper: np.ndarray | None = None,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> dict[str, Any]:
    """Minimize CDaR via LP-style formulation with linearized drawdown magnitudes."""
    if returns.empty:
        raise ValueError("returns DataFrame is empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    t_count, n_assets = returns.shape
    r = returns.to_numpy(dtype=float)
    mu = returns.mean().to_numpy(dtype=float)

    w = cp.Variable(n_assets)
    c = cp.Variable(t_count)  # proxy cumulative return path
    p = cp.Variable(t_count)  # running peak proxy
    x = cp.Variable(t_count)  # drawdown magnitude proxy
    eta = cp.Variable()
    xi = cp.Variable(t_count)

    constraints = [cp.sum(w) == 1]

    if per_asset_lower is not None and per_asset_upper is not None:
        constraints += [w >= per_asset_lower, w <= per_asset_upper]
    elif weight_bounds is not None:
        lb, ub = weight_bounds
        constraints += [w >= lb, w <= ub]
    else:
        lb_default, ub_default = default_bounds(n_assets, allow_short=not no_short)
        constraints += [w >= lb_default, w <= ub_default]

    if no_short:
        constraints.append(w >= 0)

    if gross_exposure_limit is not None:
        constraints.append(cp.norm1(w) <= gross_exposure_limit)

    # Cumulative proxy: c_t = sum_{s<=t} r_s^T w
    for t in range(t_count):
        constraints.append(c[t] == cp.sum(r[: t + 1] @ w))

    # Running peak constraints and drawdown magnitudes.
    constraints += [p[0] >= c[0]]
    for t in range(1, t_count):
        constraints += [p[t] >= p[t - 1], p[t] >= c[t]]
    constraints += [x == p - c, x >= 0]

    # CVaR-on-drawdown proxy objective.
    constraints += [xi >= x - eta, xi >= 0]

    if target_return is not None:
        constraints.append(mu @ w >= target_return)

    objective = cp.Minimize(eta + (1.0 / ((1.0 - alpha) * t_count)) * cp.sum(xi))
    problem = cp.Problem(objective, constraints)

    status = _solve_with_fallback(problem, primary=solver or "ECOS", fallback="SCS")

    if w.value is None:
        return {
            "weights": np.full(n_assets, np.nan),
            "cdar": np.nan,
            "expected_return": np.nan,
            "other_metrics": {},
            "status": status,
        }

    weights = np.asarray(w.value, dtype=float)
    port = compute_portfolio_returns(returns, weights)
    values = compute_cumulative_value(port)
    drawdown = compute_drawdown_curve(values)
    metrics = summarize_core_metrics(
        port,
        drawdown,
        alpha=alpha,
        annualization_factor=annualization_factor,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    if target_cdar is not None and metrics["cdar"] > target_cdar:
        status = "target_cdar_not_met"

    return {
        "weights": weights,
        "cdar": float(metrics["cdar"]),
        "expected_return": float(port.mean()),
        "other_metrics": metrics,
        "status": status,
    }



def compute_cdar_efficient_frontier(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    no_short: bool = True,
    n_points: int = 20,
    return_range: tuple[float, float] | None = None,
    gross_exposure_limit: float | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute a Mean-CDaR frontier over target return grid."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    if return_range is None:
        lo, hi = _adaptive_return_range(returns, no_short=no_short)
    else:
        lo, hi = return_range

    targets = np.linspace(lo, hi, n_points)

    def _solve(target: float) -> dict[str, Any]:
        result = optimize_portfolio_cdar(
            returns,
            alpha=alpha,
            no_short=no_short,
            target_return=float(target),
            gross_exposure_limit=gross_exposure_limit,
            weight_bounds=None,
        )
        row = {
            "target_return": float(target),
            "achieved_return": (
                float(result["expected_return"])
                if np.isfinite(result["expected_return"])
                else np.nan
            ),
            "cdar": float(result["cdar"]) if np.isfinite(result["cdar"]) else np.nan,
            "volatility": float(result["other_metrics"].get("volatility", np.nan)),
            "max_drawdown": float(result["other_metrics"].get("max_drawdown", np.nan)),
            "status": result["status"],
            "weights": result["weights"],
        }
        return row

    if parallel and n_points >= 20:
        rows = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(_solve)(t) for t in targets
        )
    else:
        rows = [_solve(t) for t in targets]

    return pd.DataFrame(rows)



def compute_mean_var_cdar_surface(
    returns: pd.DataFrame,
    alpha: float,
    lambda_grid: list[tuple[float, float, float]],
    no_short: bool = True,
    gross_exposure_limit: float | None = None,
    parallel: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute mean-variance-CDaR trade-off surface via scalarized optimization."""
    if not lambda_grid:
        raise ValueError("lambda_grid must not be empty.")
    if len(lambda_grid) > 200:
        LOGGER.warning("Large lambda_grid (%d). Runtime may be high.", len(lambda_grid))

    t_count, n_assets = returns.shape
    r = returns.to_numpy(dtype=float)
    mu = returns.mean().to_numpy(dtype=float)
    cov = np.cov(r, rowvar=False)

    def _solve_one(lambdas: tuple[float, float, float]) -> dict[str, Any]:
        lambda_cdar, lambda_var, lambda_ret = lambdas

        w = cp.Variable(n_assets)
        c = cp.Variable(t_count)
        p = cp.Variable(t_count)
        x = cp.Variable(t_count)
        eta = cp.Variable()
        xi = cp.Variable(t_count)

        constraints = [cp.sum(w) == 1]
        if no_short:
            constraints.append(w >= 0)
        else:
            constraints += [w >= -1.0, w <= 1.0]

        if gross_exposure_limit is not None:
            constraints.append(cp.norm1(w) <= gross_exposure_limit)

        for t in range(t_count):
            constraints.append(c[t] == cp.sum(r[: t + 1] @ w))
        constraints += [p[0] >= c[0]]
        for t in range(1, t_count):
            constraints += [p[t] >= p[t - 1], p[t] >= c[t]]
        constraints += [x == p - c, x >= 0, xi >= x - eta, xi >= 0]

        cdar_expr = eta + (1.0 / ((1.0 - alpha) * t_count)) * cp.sum(xi)
        var_expr = cp.quad_form(w, cov)
        ret_expr = mu @ w
        objective = cp.Minimize(
            lambda_cdar * cdar_expr + lambda_var * var_expr - lambda_ret * ret_expr
        )

        problem = cp.Problem(objective, constraints)
        status = _solve_with_fallback(problem, primary="ECOS", fallback="SCS")

        if w.value is None:
            return {
                "lambda_cdar": lambda_cdar,
                "lambda_var": lambda_var,
                "lambda_return": lambda_ret,
                "expected_return": np.nan,
                "volatility": np.nan,
                "cdar": np.nan,
                "max_drawdown": np.nan,
                "status": status,
                "weights": np.full(n_assets, np.nan),
            }

        weights = np.asarray(w.value, dtype=float)
        port = compute_portfolio_returns(returns, weights)
        values = compute_cumulative_value(port)
        drawdown = compute_drawdown_curve(values)
        metrics = summarize_core_metrics(port, drawdown, alpha=alpha)

        return {
            "lambda_cdar": lambda_cdar,
            "lambda_var": lambda_var,
            "lambda_return": lambda_ret,
            "expected_return": float(port.mean()),
            "volatility": float(metrics["volatility"]),
            "cdar": float(metrics["cdar"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "status": status,
            "weights": weights,
        }

    if parallel and len(lambda_grid) >= 40:
        rows = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(_solve_one)(l) for l in lambda_grid
        )
    else:
        rows = [_solve_one(l) for l in lambda_grid]

    return pd.DataFrame(rows)



def _adaptive_return_range(
    returns: pd.DataFrame,
    no_short: bool,
) -> tuple[float, float]:
    """Estimate frontier return range from equal-weight and corner portfolios."""
    n_assets = returns.shape[1]
    means = returns.mean().to_numpy(dtype=float)

    equal_w = np.full(n_assets, 1.0 / n_assets)
    candidates = [float(means @ equal_w)]

    for i in range(n_assets):
        w = np.zeros(n_assets)
        w[i] = 1.0
        candidates.append(float(means @ w))

    return float(np.nanmin(candidates)), float(np.nanmax(candidates))



def _solve_with_fallback(
    problem: cp.Problem,
    primary: str,
    fallback: str,
) -> str:
    """Solve optimization with primary solver and fallback solver."""
    try:
        problem.solve(solver=primary)
        status = str(problem.status)
        if status in {"optimal", "optimal_inaccurate"}:
            LOGGER.info("Optimization solved with solver=%s status=%s", primary, status)
            return status
        LOGGER.warning(
            "Primary solver %s returned status=%s. Trying fallback %s.",
            primary,
            status,
            fallback,
        )
    except Exception as exc:  # pragma: no cover - solver-specific runtime paths
        LOGGER.warning("Primary solver %s failed: %s. Trying fallback %s.", primary, exc, fallback)

    try:
        problem.solve(solver=fallback)
        status = str(problem.status)
        LOGGER.info("Fallback solver=%s status=%s", fallback, status)
        return status
    except Exception as exc:  # pragma: no cover - solver-specific runtime paths
        LOGGER.error("Fallback solver %s failed: %s", fallback, exc)
        return "solver_error"
