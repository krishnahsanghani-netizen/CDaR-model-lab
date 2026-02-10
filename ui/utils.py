"""UI helper utilities (pure logic, testable without Streamlit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd



def load_example_prices() -> pd.DataFrame:
    """Load bundled example prices for offline UI usage."""
    path = Path("tests/data/sample_prices_spy_agg_gld_qqq.csv")
    if not path.exists():
        raise FileNotFoundError(
            "Example dataset not found at tests/data/sample_prices_spy_agg_gld_qqq.csv"
        )
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame.sort_index()



def autodetect_csv_columns(frame: pd.DataFrame) -> tuple[str | None, list[str]]:
    """Infer likely date column and numeric price columns from uploaded CSV."""
    date_col: str | None = None
    for column in frame.columns:
        if "date" in column.lower():
            date_col = column
            break

    numeric_cols = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and column != date_col
    ]
    return date_col, numeric_cols



def parse_uploaded_prices(
    frame: pd.DataFrame,
    date_col: str,
    price_cols: list[str],
) -> pd.DataFrame:
    """Normalize uploaded CSV data into datetime-indexed price matrix."""
    if date_col not in frame.columns:
        raise ValueError(f"date column '{date_col}' not found in uploaded file")
    missing = [col for col in price_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"price columns missing from uploaded file: {missing}")

    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    out = out.loc[:, price_cols]
    if out.empty:
        raise ValueError("Uploaded data has no rows after parsing selected date/price columns")
    return out
