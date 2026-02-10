import numpy as np
import pandas as pd
import pytest

from enhanced_cdar.portfolio.weights import (
    default_bounds,
    load_asset_bounds_csv,
    validate_weights,
)


def test_default_bounds_long_only_and_long_short():
    lo, hi = default_bounds(3, allow_short=False)
    assert np.allclose(lo, [0.0, 0.0, 0.0])
    assert np.allclose(hi, [1.0, 1.0, 1.0])

    lo_s, hi_s = default_bounds(2, allow_short=True)
    assert np.allclose(lo_s, [-1.0, -1.0])
    assert np.allclose(hi_s, [1.0, 1.0])


def test_validate_weights_gross_exposure_violation():
    with pytest.raises(ValueError):
        validate_weights(
            np.array([1.5, -0.5]),
            no_short=False,
            gross_exposure_limit=1.8,
        )


def test_load_asset_bounds_csv(tmp_path):
    frame = pd.DataFrame(
        {
            "asset": ["A", "B"],
            "lower": [0.0, -0.2],
            "upper": [0.8, 1.0],
        }
    )
    path = tmp_path / "bounds.csv"
    frame.to_csv(path, index=False)

    lower, upper = load_asset_bounds_csv(path, ["A", "B"])
    assert np.allclose(lower, [0.0, -0.2])
    assert np.allclose(upper, [0.8, 1.0])
