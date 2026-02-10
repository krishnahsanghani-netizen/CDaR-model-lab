import numpy as np
import pandas as pd
import pytest

cvxpy = pytest.importorskip("cvxpy")

from enhanced_cdar.portfolio.optimization import (  # noqa: E402
    compute_cdar_efficient_frontier,
    optimize_portfolio_cdar,
)


def test_optimize_portfolio_cdar_weights_sum_to_one():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.005, -0.01, 0.003, 0.004],
            "B": [0.004, 0.003, -0.005, 0.002, 0.002],
            "C": [0.02, -0.01, 0.015, -0.005, 0.01],
        }
    )
    result = optimize_portfolio_cdar(returns, alpha=0.9)
    if result["status"] in {"optimal", "optimal_inaccurate"}:
        assert np.isclose(np.sum(result["weights"]), 1.0, atol=1e-6)


def test_frontier_outputs_status_column():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.005, -0.01, 0.003, 0.004],
            "B": [0.004, 0.003, -0.005, 0.002, 0.002],
            "C": [0.02, -0.01, 0.015, -0.005, 0.01],
        }
    )
    frontier = compute_cdar_efficient_frontier(returns, n_points=5, parallel=False)
    assert "status" in frontier.columns
    assert "message" in frontier.columns
    assert len(frontier) == 5


def test_optimize_portfolio_cdar_infeasible_has_message():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.005, -0.01, 0.003, 0.004],
            "B": [0.004, 0.003, -0.005, 0.002, 0.002],
            "C": [0.02, -0.01, 0.015, -0.005, 0.01],
        }
    )
    result = optimize_portfolio_cdar(
        returns,
        alpha=0.9,
        target_return=10.0,
        no_short=True,
        weight_bounds=(0.0, 1.0),
    )
    if result["status"] != "optimal":
        assert "message" in result
        assert isinstance(result["message"], str)
