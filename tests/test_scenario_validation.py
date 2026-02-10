import numpy as np
import pandas as pd
import pytest

from enhanced_cdar.portfolio.scenario import build_scenario_returns, evaluate_portfolio_scenarios


def test_build_scenario_returns_missing_asset_raises():
    returns = pd.DataFrame({"A": [0.01, 0.02]})
    with pytest.raises(ValueError):
        build_scenario_returns(returns, asset_shocks={"B": -0.01})


def test_evaluate_portfolio_scenarios_requires_matching_weights():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, -0.01]})
    with pytest.raises(ValueError):
        evaluate_portfolio_scenarios(
            returns,
            np.array([1.0]),
            scenarios={"shock": {"global_shift": -0.01}},
        )
