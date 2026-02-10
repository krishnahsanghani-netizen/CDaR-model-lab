import pandas as pd
import pytest

from ui.utils import autodetect_csv_columns, load_example_prices, parse_uploaded_prices


def test_autodetect_csv_columns_finds_date_and_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "SPY": [100.0, 101.0],
            "AGG": [80.0, 80.2],
            "Label": ["x", "y"],
        }
    )
    date_col, price_cols = autodetect_csv_columns(frame)
    assert date_col == "Date"
    assert price_cols == ["SPY", "AGG"]


def test_parse_uploaded_prices_validates_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "SPY": [100.0, 101.0],
        }
    )
    out = parse_uploaded_prices(frame, "date", ["SPY"])
    assert list(out.columns) == ["SPY"]
    assert out.index.name == "date"

    with pytest.raises(ValueError):
        parse_uploaded_prices(frame, "missing", ["SPY"])


def test_load_example_prices_not_empty() -> None:
    prices = load_example_prices()
    assert not prices.empty
    assert prices.index.is_monotonic_increasing
