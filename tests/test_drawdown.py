import numpy as np
import pandas as pd

from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    compute_portfolio_returns,
    compute_returns,
    max_drawdown,
)


def test_compute_returns_simple():
    prices = pd.DataFrame({"A": [100, 110, 121], "B": [200, 220, 198]})
    returns = compute_returns(prices, method="simple")
    assert np.isclose(returns.iloc[0, 0], 0.10)
    assert np.isclose(returns.iloc[1, 1], -0.10)


def test_portfolio_drawdown_and_max_drawdown():
    returns = pd.DataFrame(
        {
            "A": [0.10, -0.20, 0.05],
            "B": [0.00, -0.10, 0.02],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    weights = np.array([0.5, 0.5])
    port = compute_portfolio_returns(returns, weights)
    values = compute_cumulative_value(port)
    dd = compute_drawdown_curve(values)
    mdd = max_drawdown(dd)

    assert dd.max() <= 0
    assert mdd >= 0
