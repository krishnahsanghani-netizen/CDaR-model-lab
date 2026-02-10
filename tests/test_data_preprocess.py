import pandas as pd
import pytest

from enhanced_cdar.data.preprocess import align_and_clean_prices


def test_align_and_clean_prices_default_ffill_then_drop():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            "A": [100.0, None, 102.0, 103.0],
            "B": [50.0, 50.2, None, 50.4],
        },
        index=idx,
    )
    cleaned = align_and_clean_prices(prices)
    assert not cleaned.isna().any().any()
    assert len(cleaned) >= 1


def test_align_and_clean_prices_raise_policy_errors_on_missing():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, None, 102.0]}, index=idx)
    with pytest.raises(ValueError):
        align_and_clean_prices(prices, missing_data_policy="raise")
