import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(42)
    a = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
    b = 80 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(idx)))
    c = 120 * np.cumprod(1 + rng.normal(0.0007, 0.012, len(idx)))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
