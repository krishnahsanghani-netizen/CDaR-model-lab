import pandas as pd

from enhanced_cdar.portfolio.optimization import compute_cdar_efficient_frontier


def test_frontier_includes_status_for_all_rows():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.005, -0.01, 0.003, 0.004],
            "B": [0.004, 0.003, -0.005, 0.002, 0.002],
            "C": [0.02, -0.01, 0.015, -0.005, 0.01],
        }
    )

    frontier = compute_cdar_efficient_frontier(
        returns,
        alpha=0.9,
        n_points=7,
        parallel=False,
        return_range=(0.5, 1.0),
    )

    assert len(frontier) == 7
    assert "status" in frontier.columns
    assert "message" in frontier.columns
