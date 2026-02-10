import numpy as np
import pandas as pd

from enhanced_cdar.portfolio.scenario import (
    build_scenario_returns,
    evaluate_portfolio_scenarios,
    preset_scenarios,
)


def test_build_scenario_returns_global_shift_and_asset_shock():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.0, -0.01],
        }
    )
    shocked = build_scenario_returns(
        returns,
        asset_shocks={"A": -0.02},
        global_shift=-0.01,
    )
    assert np.isclose(shocked.loc[0, "A"], -0.02)
    assert np.isclose(shocked.loc[0, "B"], -0.01)


def test_evaluate_portfolio_scenarios_includes_baseline_and_presets():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.0],
            "B": [0.0, -0.01, 0.005, 0.004],
        }
    )
    weights = np.array([0.5, 0.5])
    scenarios = preset_scenarios()
    result = evaluate_portfolio_scenarios(returns, weights, scenarios, alpha=0.9)
    assert "baseline" in result["scenario"].tolist()
    assert len(result) == len(scenarios) + 1
