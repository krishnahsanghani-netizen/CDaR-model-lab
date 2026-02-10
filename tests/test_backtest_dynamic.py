import numpy as np
import pandas as pd

from enhanced_cdar.portfolio.backtest import run_backtest


def test_run_backtest_dynamic_rebalance_invokes_optimizer():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.005, 0.004, 0.003, 0.002, -0.001, 0.005, 0.001],
            "B": [0.004, -0.002, 0.003, 0.002, 0.001, 0.0, 0.002, 0.001],
        },
        index=idx,
    )

    calls: list[int] = []

    def dyn_opt(history: pd.DataFrame) -> np.ndarray:
        calls.append(len(history))
        return np.array([0.6, 0.4], dtype=float)

    result = run_backtest(
        returns=returns,
        weights=np.array([0.5, 0.5]),
        rebalance_every_n_periods=2,
        rebalance_mode="dynamic",
        dynamic_optimizer=dyn_opt,
        no_short=True,
    )

    assert len(calls) >= 2
    assert len(result.portfolio_returns) == len(returns)
