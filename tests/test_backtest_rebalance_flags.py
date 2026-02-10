import pandas as pd
import pytest

from enhanced_cdar.portfolio.backtest import _rebalance_flags


def test_rebalance_flags_every_n_periods():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    flags = _rebalance_flags(idx, rebalance_calendar="none", rebalance_every_n_periods=2)
    assert flags.iloc[0]
    assert flags.iloc[2]
    assert flags.iloc[4]


def test_rebalance_flags_monthly_boundary():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    flags = _rebalance_flags(idx, rebalance_calendar="M")
    assert flags.iloc[0]
    assert flags.iloc[2]


def test_rebalance_flags_invalid_every_n_raises():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.raises(ValueError):
        _rebalance_flags(idx, rebalance_every_n_periods=0)
