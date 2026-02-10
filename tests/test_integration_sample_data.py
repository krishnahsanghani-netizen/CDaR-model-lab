from pathlib import Path

import numpy as np
import pandas as pd

from enhanced_cdar.metrics.drawdown import compute_returns
from enhanced_cdar.portfolio.optimization import optimize_portfolio_cdar


def test_integration_sample_dataset_runs_optimization():
    path = Path("tests/data/sample_prices_spy_agg_gld_qqq.csv")
    prices = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    returns = compute_returns(prices)

    result = optimize_portfolio_cdar(returns, alpha=0.9)
    assert "status" in result
    if result["status"] in {"optimal", "optimal_inaccurate"}:
        assert np.isclose(result["weights"].sum(), 1.0, atol=1e-6)
