"""Mean-variance-CDaR surface visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _build_reactive_surface(
    phase: float,
    intensity: float,
    n_grid: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deformed polar mesh used by the reactive 3D model."""
    u = np.linspace(0.0, 2.0 * np.pi, n_grid)
    v = np.linspace(0.0, 1.0, n_grid)
    uu, vv = np.meshgrid(u, v)

    radial_warp = 1.0 + 0.22 * intensity * np.sin(3.0 * uu + 2.5 * phase)
    radial_warp += 0.08 * np.cos(8.0 * vv - 1.7 * phase)

    xx = vv * radial_warp * np.cos(uu)
    yy = vv * radial_warp * np.sin(uu)

    ridge = np.sin(8.0 * uu + 3.5 * vv + phase)
    ripple = np.cos(12.0 * vv - 2.0 * phase)
    bowl = -0.6 * vv**2
    zz = 0.36 * intensity * ridge + 0.18 * (1.0 + intensity) * ripple + bowl

    color_field = np.sqrt(xx**2 + yy**2) + 0.45 * zz
    return xx, yy, zz, color_field


def _reactive_time_signal(
    drawdown: pd.Series,
    returns: pd.Series | None = None,
    max_frames: int = 260,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Create frame phases/intensities from drawdown and optional return volatility."""
    if drawdown.empty:
        raise ValueError("drawdown series is empty")

    frame_count = min(max_frames, len(drawdown))
    idx = np.linspace(0, len(drawdown) - 1, frame_count).astype(int)
    idx = np.unique(idx)

    dd_mag = (-drawdown.iloc[idx]).clip(lower=0.0).to_numpy(dtype=float)
    if dd_mag.max() > 0:
        dd_norm = dd_mag / dd_mag.max()
    else:
        dd_norm = np.zeros_like(dd_mag)

    if returns is not None and not returns.empty:
        ret_aligned = returns.reindex(drawdown.index).fillna(0.0)
        vol = ret_aligned.rolling(21).std().iloc[idx].fillna(0.0).to_numpy(dtype=float)
        if vol.max() > 0:
            vol = vol / vol.max()
        intensity = np.clip(0.15 + 0.75 * dd_norm + 0.25 * vol, 0.05, 1.25)
    else:
        intensity = np.clip(0.15 + 0.85 * dd_norm, 0.05, 1.1)

    phases = np.linspace(0.0, 2.0 * np.pi, len(idx), endpoint=False)
    labels = [str(drawdown.index[i].date()) for i in idx]
    return phases, labels, intensity


def make_cdar_reactive_model_figure(
    drawdown: pd.Series,
    returns: pd.Series | None = None,
    theme: str = "plotly_dark",
    max_frames: int = 260,
) -> go.Figure:
    """Create an animated, deforming 3D model driven by drawdown dynamics."""
    phases, labels, intensity = _reactive_time_signal(
        drawdown=drawdown,
        returns=returns,
        max_frames=max_frames,
    )
    x0, y0, z0, c0 = _build_reactive_surface(phases[0], float(intensity[0]))

    colorscale = [
        [0.0, "#001219"],
        [0.2, "#005f73"],
        [0.45, "#0a9396"],
        [0.7, "#ee9b00"],
        [1.0, "#ca6702"],
    ]

    lighting = {
        "ambient": 0.32,
        "diffuse": 0.78,
        "specular": 1.15,
        "roughness": 0.22,
        "fresnel": 0.18,
    }
    contours = {
        "z": {
            "show": True,
            "usecolormap": False,
            "color": "#d4d7dd",
            "width": 1,
        }
    }

    frames: list[go.Frame] = []
    for i in range(len(phases)):
        xx, yy, zz, cc = _build_reactive_surface(phases[i], float(intensity[i]))
        frames.append(
            go.Frame(
                data=[
                    go.Surface(
                        x=xx,
                        y=yy,
                        z=zz,
                        surfacecolor=cc,
                        cmin=float(c0.min()),
                        cmax=float(c0.max() + 0.8),
                        colorscale=colorscale,
                        opacity=0.98,
                        showscale=False,
                        contours=contours,
                        lighting=lighting,
                        lightposition={"x": 120, "y": 35, "z": 140},
                    )
                ],
                name=str(i),
                traces=[0],
            )
        )

    fig = go.Figure(
        data=[
            go.Surface(
                x=x0,
                y=y0,
                z=z0,
                surfacecolor=c0,
                cmin=float(c0.min()),
                cmax=float(c0.max() + 0.8),
                colorscale=colorscale,
                opacity=0.98,
                showscale=False,
                contours=contours,
                lighting=lighting,
                lightposition={"x": 120, "y": 35, "z": 140},
            )
        ],
        frames=frames,
    )

    frame_ms = 130
    slider_steps = [
        {
            "args": [
                [str(i)],
                {"frame": {"duration": frame_ms, "redraw": True}, "mode": "immediate"},
            ],
            "label": labels[i],
            "method": "animate",
        }
        for i in range(len(labels))
    ]

    fig.update_layout(
        title="Reactive 3D Risk Model (CDaR-Driven Deformation)",
        template=theme,
        height=760,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.8)),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.03,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "y": 0.03,
                "len": 0.84,
                "steps": slider_steps,
            }
        ],
    )
    return fig


def make_mean_variance_cdar_surface_figure(
    surface_df: pd.DataFrame,
    theme: str = "plotly_white",
) -> go.Figure:
    """Create interactive 3D mean-variance-CDaR surface for UI rendering."""
    frame = surface_df.dropna(subset=["volatility", "cdar", "expected_return"]).copy()
    if frame.empty:
        raise ValueError("No feasible points available to plot surface.")

    color_col = "max_drawdown" if "max_drawdown" in frame.columns else "cdar"
    fig = px.scatter_3d(
        frame,
        x="volatility",
        y="cdar",
        z="expected_return",
        color=color_col,
        title="Mean-Variance-CDaR Surface",
        template=theme,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.85))
    fig.update_layout(
        scene=dict(
            xaxis_title="Volatility",
            yaxis_title="CDaR",
            zaxis_title="Expected Return",
        )
    )
    return fig


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
        fig = make_mean_variance_cdar_surface_figure(frame)
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


def _save_plotly(fig: go.Figure, save_path: str) -> None:
    out = Path(save_path)
    if out.suffix.lower() == ".html":
        fig.write_html(str(out))
    else:
        fig.write_image(str(out))
