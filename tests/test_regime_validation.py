import numpy as np
import pandas as pd
import pytest

from enhanced_cdar.portfolio.regime import compute_regime_metrics


def test_regime_requires_datetime_index():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, -0.01]})
    weights = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        compute_regime_metrics(returns, weights)


def test_regime_min_periods_can_filter_all_rows():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = pd.DataFrame(
        {"A": [0.01, 0.0, 0.01, 0.0, 0.01, 0.0], "B": [0.0, 0.01, 0.0, 0.01, 0.0, 0.01]},
        index=idx,
    )
    weights = np.array([0.5, 0.5])
    regime_df = compute_regime_metrics(returns, weights, regime_frequency="M", min_periods=100)
    assert regime_df.empty
