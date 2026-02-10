"""Mean-variance-CDaR surface visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px



def plot_mean_variance_cdar_surface(
    surface_df: pd.DataFrame,
    mode: str = "3d",
    show: bool = True,
    save_path: str | None = None,
    backend: str = "plotly",
):
    """Plot mean-variance-CDaR surface in 3D or 2D projection mode."""
    frame = surface_df.dropna(subset=["volatility", "cdar", "expected_return"]).copy()
    if frame.empty:
        raise ValueError("No feasible points available to plot surface.")

    if backend == "matplotlib":
        fig = plt.figure(figsize=(8, 6))
        if mode == "3d":
            ax = fig.add_subplot(111, projection="3d")
            points = ax.scatter(
                frame["volatility"],
                frame["cdar"],
                frame["expected_return"],
                c=frame.get("max_drawdown", frame["cdar"]),
            )
            ax.set_xlabel("Volatility")
            ax.set_ylabel("CDaR")
            ax.set_zlabel("Expected Return")
            fig.colorbar(points)
        else:
            ax = fig.add_subplot(111)
            points = ax.scatter(
                frame["volatility"], frame["expected_return"], c=frame["cdar"]
            )
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Expected Return")
            fig.colorbar(points, label="CDaR")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        return fig

    if mode == "3d":
        fig = px.scatter_3d(
            frame,
            x="volatility",
            y="cdar",
            z="expected_return",
            color="max_drawdown" if "max_drawdown" in frame.columns else "cdar",
            title="Mean-Variance-CDaR Surface",
        )
    elif mode == "2d-projections":
        fig = px.scatter(
            frame,
            x="volatility",
            y="expected_return",
            color="cdar",
            title="Mean-Variance Projection Colored by CDaR",
        )
    else:
        raise ValueError("mode must be '3d' or '2d-projections'.")

    if save_path:
        _save_plotly(fig, save_path)
    if show:
        fig.show()
    return fig



def _save_plotly(fig, save_path: str) -> None:
    out = Path(save_path)
    if out.suffix.lower() == ".html":
        fig.write_html(str(out))
    else:
        fig.write_image(str(out))
