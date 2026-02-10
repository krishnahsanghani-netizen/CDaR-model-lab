import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

from enhanced_cdar.metrics.cdar import compute_cdar
from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    max_drawdown,
)


@given(
    st.lists(
        st.floats(
            min_value=-0.2,
            max_value=0.2,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=5,
        max_size=50,
    )
)
def test_cdar_is_non_negative(returns_list):
    returns = pd.Series(returns_list)
    values = compute_cumulative_value(returns)
    dd = compute_drawdown_curve(values)
    cdar = compute_cdar(dd, alpha=0.9)
    assert cdar >= 0.0


@given(
    st.lists(
        st.floats(
            min_value=-0.2,
            max_value=0.2,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=5,
        max_size=50,
    )
)
def test_max_drawdown_is_non_negative(returns_list):
    returns = pd.Series(returns_list)
    values = compute_cumulative_value(returns)
    dd = compute_drawdown_curve(values)
    mdd = max_drawdown(dd)
    assert mdd >= 0.0
    assert np.isfinite(mdd)
