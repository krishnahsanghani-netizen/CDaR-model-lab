from pathlib import Path

import numpy as np
import pandas as pd

from enhanced_cdar.viz import animation as anim


def test_select_frame_indices_caps_and_sorts() -> None:
    idx = anim.select_frame_indices(1000, max_frames=123)
    assert len(idx) <= 123
    assert idx[0] == 0
    assert idx[-1] == 999
    assert np.all(np.diff(idx) >= 0)


def test_generate_frontier_snapshots_custom_step(monkeypatch) -> None:
    returns = pd.DataFrame(
        {
            "A": np.linspace(0.001, 0.002, 260),
            "B": np.linspace(0.0005, 0.0015, 260),
        },
        index=pd.date_range("2024-01-01", periods=260, freq="D"),
    )

    def _fake_frontier(**_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target_return": [0.01, 0.02],
                "achieved_return": [0.011, 0.019],
                "cdar": [0.03, 0.04],
            }
        )

    monkeypatch.setattr(anim, "compute_cdar_efficient_frontier", _fake_frontier)

    snapshots = anim.generate_frontier_snapshots(
        returns=returns,
        lookback=100,
        step_mode="custom",
        step_n=40,
        max_snapshots=10,
    )
    assert snapshots
    assert isinstance(snapshots[0][0], str)
    assert not snapshots[0][1].empty


def test_animate_underwater_smoke_writes_artifact(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    returns = pd.Series(np.sin(np.linspace(0, 3, 40)) * 0.002, index=idx)
    values = (1 + returns).cumprod()
    drawdown = values / values.cummax() - 1.0

    out = anim.animate_underwater(
        values=values,
        drawdown=drawdown,
        fps=6,
        max_frames=20,
        save_path=str(tmp_path / "underwater.mp4"),
    )

    out_path = Path(out)
    assert out_path.exists()
    assert out_path.suffix in {".mp4", ".gif"}
