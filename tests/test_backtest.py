import numpy as np
import pandas as pd

from enhanced_cdar.portfolio.backtest import run_backtest


def test_run_backtest_basic_and_benchmark_metrics():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.005, 0.01, -0.01, 0.002],
            "B": [0.004, -0.01, 0.003, 0.009, -0.004, 0.001],
        },
        index=idx,
    )
    benchmark = pd.Series([0.006, -0.012, 0.004, 0.008, -0.006, 0.001], index=idx)

    result = run_backtest(
        returns=returns,
        weights=np.array([0.6, 0.4]),
        rebalance_calendar="M",
        benchmark_returns=benchmark,
    )

    assert len(result.portfolio_returns) == len(returns)
    assert "cdar" in result.metrics
    assert result.benchmark_metrics is not None
    assert "tracking_error" in result.benchmark_metrics
