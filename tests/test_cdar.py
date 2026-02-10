import numpy as np
import pandas as pd

from enhanced_cdar.metrics.cdar import compute_cdar, compute_rolling_cdar


def test_compute_cdar_non_negative():
    dd = pd.Series([0.0, -0.05, -0.10, -0.03, 0.0])
    value = compute_cdar(dd, alpha=0.8)
    assert value >= 0


def test_compute_rolling_cdar_length_and_nans():
    dd = pd.Series([0.0, -0.01, -0.03, -0.02, -0.04])
    roll = compute_rolling_cdar(dd, alpha=0.8, window=3)
    assert len(roll) == len(dd)
    assert np.isnan(roll.iloc[0])
