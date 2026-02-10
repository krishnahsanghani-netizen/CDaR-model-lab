"""Efficient frontier visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def make_cdar_frontier_figure(
    frontier_df: pd.DataFrame,
    current_point: dict[str, Any] | None = None,
    theme: str = "plotly_white",
) -> go.Figure:
    """Build interactive Mean-CDaR frontier figure for UI usage."""
    valid = frontier_df.dropna(subset=["cdar", "achieved_return"]).copy()
    if valid.empty:
        raise ValueError("No feasible points available to plot frontier.")

    fig = px.line(
        valid,
        x="cdar",
        y="achieved_return",
        markers=True,
        title="Mean-CDaR Efficient Frontier",
        color_discrete_sequence=["#1f77b4"],
        template=theme,
    )
    fig.update_traces(
        hovertemplate="CDaR=%{x:.4f}<br>Return=%{y:.4f}<extra></extra>",
    )
    if "volatility" in valid.columns:
        fig.update_traces(marker=dict(color=valid["volatility"], colorscale="Viridis"))

    if current_point is not None:
        cdar = current_point.get("cdar")
        ret = current_point.get("expected_return")
        if cdar is not None and ret is not None:
            fig.add_scatter(
                x=[float(cdar)],
                y=[float(ret)],
                mode="markers",
                name="Selected Portfolio",
                marker=dict(size=12, color="#d62728", symbol="x"),
            )

    min_cdar_idx = valid["cdar"].idxmin()
    max_return_idx = valid["achieved_return"].idxmax()
    fig.add_scatter(
        x=[valid.loc[min_cdar_idx, "cdar"]],
        y=[valid.loc[min_cdar_idx, "achieved_return"]],
        mode="markers",
        name="Min CDaR",
        marker=dict(size=10, color="#2ca02c"),
    )
    fig.add_scatter(
        x=[valid.loc[max_return_idx, "cdar"]],
        y=[valid.loc[max_return_idx, "achieved_return"]],
        mode="markers",
        name="Max Return",
        marker=dict(size=10, color="#9467bd"),
    )
    fig.update_xaxes(title_text="CDaR")
    fig.update_yaxes(title_text="Expected Return")
    return fig


def plot_cdar_efficient_frontier(
    frontier_df: pd.DataFrame,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    backend: str = "plotly",
):
    """Plot Mean-CDaR efficient frontier."""
    frame = frontier_df.copy()
    plot_title = title or "Mean-CDaR Efficient Frontier"

    valid = frame.dropna(subset=["cdar", "achieved_return"]).copy()
    if valid.empty:
        raise ValueError("No feasible points available to plot frontier.")

    min_cdar_idx = valid["cdar"].idxmin()
    max_return_idx = valid["achieved_return"].idxmax()

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(valid["cdar"], valid["achieved_return"], marker="o")
        ax.scatter(
            valid.loc[min_cdar_idx, "cdar"],
            valid.loc[min_cdar_idx, "achieved_return"],
            label="Min CDaR",
        )
        ax.scatter(
            valid.loc[max_return_idx, "cdar"],
            valid.loc[max_return_idx, "achieved_return"],
            label="Max Return",
        )
        ax.set_xlabel("CDaR")
        ax.set_ylabel("Expected Return")
        ax.set_title(plot_title)
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        return fig

    fig = make_cdar_frontier_figure(valid)
    fig.update_layout(title=plot_title)

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
