"""Underwater plot utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from enhanced_cdar.metrics.drawdown import compute_drawdown_curve



def plot_underwater(
    values: pd.Series,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    backend: str = "plotly",
):
    """Plot portfolio value and underwater drawdown."""
    drawdown = compute_drawdown_curve(values)
    chart_title = title or "Portfolio Value and Underwater Drawdown"

    if backend == "matplotlib":
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(values.index, values.values, label="Portfolio Value")
        axes[0].set_title(chart_title)
        axes[0].set_ylabel("Value")
        axes[0].legend()

        axes[1].plot(drawdown.index, drawdown.values, color="crimson", label="Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        return fig

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=values.index, y=values.values, mode="lines", name="Portfolio Value"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, mode="lines", name="Drawdown"),
        row=2,
        col=1,
    )
    fig.update_layout(title=chart_title, height=700)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)

    if save_path:
        _save_plotly(fig, save_path)
    if show:
        fig.show()
    return fig



def _save_plotly(fig: go.Figure, save_path: str) -> None:
    out = Path(save_path)
    if out.suffix.lower() == ".html":
        fig.write_html(str(out))
    else:
        fig.write_image(str(out))
