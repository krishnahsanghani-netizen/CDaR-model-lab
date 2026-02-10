"""Underwater plot utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from enhanced_cdar.metrics.drawdown import compute_drawdown_curve


def make_underwater_figure(
    values: pd.Series,
    drawdown: pd.Series,
    benchmark_values: pd.Series | None = None,
    rolling_cdar: pd.Series | None = None,
    theme: str = "plotly_white",
) -> go.Figure:
    """Create an interactive underwater figure for UI rendering."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Portfolio Value", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(x=values.index, y=values.values, mode="lines", name="Portfolio"),
        row=1,
        col=1,
    )
    if benchmark_values is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_values.index,
                y=benchmark_values.values,
                mode="lines",
                name="Benchmark",
                line=dict(color="gray"),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown",
            line=dict(color="crimson"),
        ),
        row=2,
        col=1,
    )
    if rolling_cdar is not None:
        fig.add_trace(
            go.Scatter(
                x=rolling_cdar.index,
                y=-rolling_cdar.values,
                mode="lines",
                name="Rolling CDaR",
                line=dict(color="darkorange", dash="dash"),
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title="Portfolio Value and Underwater Drawdown",
        template=theme,
        height=720,
    )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    return fig


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

    fig = make_underwater_figure(values=values, drawdown=drawdown)
    fig.update_layout(title=chart_title)

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
