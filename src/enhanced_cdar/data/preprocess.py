"""Preprocessing utilities for price data."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

LOGGER = logging.getLogger(__name__)

MissingDataPolicy = Literal["ffill_then_drop", "drop", "raise"]


def align_and_clean_prices(
    prices_df: pd.DataFrame, missing_data_policy: MissingDataPolicy = "ffill_then_drop"
) -> pd.DataFrame:
    """Align and clean price matrix according to configured NA policy."""
    if prices_df.empty:
        raise ValueError("Input prices DataFrame is empty.")

    cleaned = prices_df.copy().sort_index()
    cleaned = cleaned.dropna(how="all")

    if missing_data_policy == "ffill_then_drop":
        LOGGER.warning("Applying missing-data policy: forward-fill then drop remaining NA rows.")
        cleaned = cleaned.ffill().dropna(how="any")
    elif missing_data_policy == "drop":
        LOGGER.warning("Applying missing-data policy: drop rows with any NA values.")
        cleaned = cleaned.dropna(how="any")
    elif missing_data_policy == "raise":
        if cleaned.isna().any().any():
            raise ValueError("Missing values present in prices and policy='raise'.")
    else:
        raise ValueError(f"Unsupported missing_data_policy='{missing_data_policy}'.")

    if cleaned.empty:
        raise ValueError("No rows remain after cleaning price data.")
    return cleaned


def normalize_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each asset series to start at 1.0."""
    if prices_df.empty:
        raise ValueError("Input prices DataFrame is empty.")
    first_row = prices_df.iloc[0]
    if (first_row <= 0).any():
        raise ValueError("Normalization requires strictly positive initial prices.")
    return prices_df / first_row
