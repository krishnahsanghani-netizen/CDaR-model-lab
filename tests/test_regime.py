import numpy as np
import pandas as pd

from enhanced_cdar.portfolio.regime import compare_regime_cdar, compute_regime_metrics


def test_compute_regime_metrics_yearly():
    idx = pd.date_range("2022-01-01", periods=520, freq="D")
    returns = pd.DataFrame(
        {
            "A": np.full(len(idx), 0.0005),
            "B": np.full(len(idx), 0.0002),
        },
        index=idx,
    )
    weights = np.array([0.5, 0.5])
    regime_df = compute_regime_metrics(
        returns=returns,
        weights=weights,
        regime_frequency="Y",
        min_periods=50,
    )
    assert not regime_df.empty
    assert "cdar" in regime_df.columns


def test_compare_regime_cdar_returns_summary_fields():
    regime_df = pd.DataFrame(
        {
            "regime": ["2022", "2023"],
            "cdar": [0.05, 0.1],
        }
    )
    summary = compare_regime_cdar(regime_df)
    assert summary["best_regime"] == "2022"
    assert summary["worst_regime"] == "2023"
