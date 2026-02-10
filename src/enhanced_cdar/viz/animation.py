"""Animation utilities for CDaR analytics exports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd

from enhanced_cdar.portfolio.optimization import (
    compute_cdar_efficient_frontier,
    compute_mean_var_cdar_surface,
)

LOGGER = logging.getLogger(__name__)

StepMode = Literal["monthly", "quarterly", "custom"]


def select_frame_indices(n_points: int, max_frames: int = 300) -> np.ndarray:
    """Return monotonic frame indices capped at max_frames."""
    if n_points <= 0:
        return np.array([], dtype=int)
    frame_count = min(n_points, max_frames)
    idx = np.linspace(0, n_points - 1, frame_count).astype(int)
    return np.unique(idx)


def animate_underwater(
    values: pd.Series,
    drawdown: pd.Series,
    benchmark_values: pd.Series | None = None,
    fps: int = 24,
    dpi: int = 120,
    save_path: str = "underwater.mp4",
    max_frames: int = 300,
) -> str:
    """Create underwater animation and save to MP4, fallback to GIF if needed."""
    idx = select_frame_indices(len(values), max_frames=max_frames)
    if idx.size == 0:
        raise ValueError("No frames to animate.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1, ax2 = axes
    ax1.set_title("Portfolio Value Over Time")
    ax1.set_ylabel("Value")
    ax2.set_title("Underwater (Drawdown)")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")

    line_val, = ax1.plot([], [], color="tab:blue", label="Portfolio")
    line_bench = None
    if benchmark_values is not None:
        line_bench, = ax1.plot([], [], color="gray", label="Benchmark")
    line_dd, = ax2.plot([], [], color="crimson", label="Drawdown")

    ax1.legend(loc="upper left")
    ax2.legend(loc="lower left")

    x = values.index

    def _update(frame_idx: int):
        i = idx[frame_idx]
        line_val.set_data(x[: i + 1], values.iloc[: i + 1])
        if line_bench is not None and benchmark_values is not None:
            line_bench.set_data(
                benchmark_values.index[: i + 1],
                benchmark_values.iloc[: i + 1],
            )
        line_dd.set_data(drawdown.index[: i + 1], drawdown.iloc[: i + 1])

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        return (line_val, line_dd) if line_bench is None else (line_val, line_bench, line_dd)

    anim = animation.FuncAnimation(fig, _update, frames=len(idx), blit=False)
    out = _save_animation_with_fallback(anim, save_path, fps=fps, dpi=dpi)
    plt.close(fig)
    return out


def animate_frontier_over_time(
    frontier_snapshots: list[tuple[str, pd.DataFrame]],
    fps: int = 12,
    dpi: int = 120,
    save_path: str = "frontier_over_time.mp4",
    max_frames: int = 300,
) -> str:
    """Animate frontier snapshots over time."""
    if not frontier_snapshots:
        raise ValueError("frontier_snapshots is empty.")

    idx = select_frame_indices(len(frontier_snapshots), max_frames=max_frames)
    fig, ax = plt.subplots(figsize=(8, 5))

    def _update(frame_idx: int):
        ax.clear()
        label, frame = frontier_snapshots[int(idx[frame_idx])]
        valid = frame.dropna(subset=["cdar", "achieved_return"])
        ax.plot(valid["cdar"], valid["achieved_return"], marker="o", color="tab:blue")
        ax.set_xlabel("CDaR")
        ax.set_ylabel("Expected Return")
        ax.set_title(f"Mean-CDaR Frontier ({label})")
        ax.grid(alpha=0.2)
        return []

    anim = animation.FuncAnimation(fig, _update, frames=len(idx), blit=False)
    out = _save_animation_with_fallback(anim, save_path, fps=fps, dpi=dpi)
    plt.close(fig)
    return out


def animate_surface_over_time(
    surface_snapshots: list[tuple[str, pd.DataFrame]],
    fps: int = 8,
    dpi: int = 120,
    save_path: str = "surface_over_time.mp4",
    max_frames: int = 300,
) -> str:
    """Animate mean-variance-CDaR surface snapshots in 3D."""
    if not surface_snapshots:
        raise ValueError("surface_snapshots is empty.")

    idx = select_frame_indices(len(surface_snapshots), max_frames=max_frames)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _update(frame_idx: int):
        ax.clear()
        label, frame = surface_snapshots[int(idx[frame_idx])]
        valid = frame.dropna(subset=["volatility", "cdar", "expected_return"])
        ax.scatter(
            valid["volatility"],
            valid["cdar"],
            valid["expected_return"],
            c=valid.get("max_drawdown", valid["cdar"]),
            cmap="viridis",
        )
        ax.set_xlabel("Volatility")
        ax.set_ylabel("CDaR")
        ax.set_zlabel("Expected Return")
        ax.set_title(f"Mean-Variance-CDaR Surface ({label})")
        ax.view_init(elev=22, azim=45 + frame_idx)
        return []

    anim = animation.FuncAnimation(fig, _update, frames=len(idx), blit=False)
    out = _save_animation_with_fallback(anim, save_path, fps=fps, dpi=dpi)
    plt.close(fig)
    return out


def animate_reactive_cdar_model(
    drawdown: pd.Series,
    returns: pd.Series | None = None,
    fps: int = 10,
    dpi: int = 180,
    save_path: str = "reactive_cdar_model.mp4",
    max_frames: int = 360,
) -> str:
    """Render a cinematic deforming 3D model driven by drawdown dynamics."""
    if drawdown.empty:
        raise ValueError("drawdown series is empty.")

    idx = select_frame_indices(len(drawdown), max_frames=max_frames)
    if idx.size == 0:
        raise ValueError("No frames to animate.")
    idx = np.repeat(idx, 2)

    dd_mag = (-drawdown.iloc[idx]).clip(lower=0.0).to_numpy(dtype=float)
    if dd_mag.max() > 0:
        dd_norm = dd_mag / dd_mag.max()
    else:
        dd_norm = np.zeros_like(dd_mag)

    vol_norm = np.zeros_like(dd_norm)
    if returns is not None and not returns.empty:
        aligned = returns.reindex(drawdown.index).fillna(0.0)
        rolling = aligned.rolling(21).std().iloc[idx].fillna(0.0).to_numpy(dtype=float)
        if rolling.max() > 0:
            vol_norm = rolling / rolling.max()

    intensity = np.clip(0.15 + 0.75 * dd_norm + 0.25 * vol_norm, 0.05, 1.25)
    phase = np.linspace(0.0, 2.0 * np.pi, len(idx), endpoint=False)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#08090f")
    ax.set_facecolor("#08090f")

    u = np.linspace(0.0, 2.0 * np.pi, 96)
    v = np.linspace(0.0, 1.0, 96)
    uu, vv = np.meshgrid(u, v)

    def _field(phase_val: float, amp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radial = 1.0 + 0.22 * amp * np.sin(3.0 * uu + 2.5 * phase_val)
        radial += 0.08 * np.cos(8.0 * vv - 1.7 * phase_val)
        xx = vv * radial * np.cos(uu)
        yy = vv * radial * np.sin(uu)
        zz = 0.36 * amp * np.sin(8.0 * uu + 3.5 * vv + phase_val)
        zz += 0.18 * (1.0 + amp) * np.cos(12.0 * vv - 2.0 * phase_val)
        zz += -0.6 * vv**2
        return xx, yy, zz

    def _update(frame_idx: int):
        ax.clear()
        i = int(frame_idx)
        xx, yy, zz = _field(float(phase[i]), float(intensity[i]))
        ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="turbo",
            linewidth=0.08,
            edgecolor=(0.03, 0.04, 0.05, 0.35),
            antialiased=True,
            rcount=96,
            ccount=96,
            alpha=0.97,
            shade=True,
        )
        ax.set_axis_off()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.4, 0.85)
        ax.set_title(
            f"Reactive CDaR Model | {drawdown.index[idx[i]].date()}",
            color="#f4f4f8",
            fontsize=13,
            pad=16,
        )
        ax.view_init(elev=24 + 6 * np.sin(i / 18.0), azim=20 + 0.78 * i)
        ax.set_box_aspect((1.0, 1.0, 0.58))
        return []

    anim = animation.FuncAnimation(fig, _update, frames=len(idx), blit=False)
    out = _save_animation_with_fallback(anim, save_path, fps=fps, dpi=dpi)
    plt.close(fig)
    return out


def generate_frontier_snapshots(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    no_short: bool = True,
    lookback: int = 252,
    step_mode: StepMode = "monthly",
    step_n: int = 21,
    max_snapshots: int = 300,
) -> list[tuple[str, pd.DataFrame]]:
    """Generate rolling frontier snapshots from return history."""
    windows = _rolling_windows(returns.index, lookback, step_mode, step_n)
    windows = windows[:max_snapshots]
    snapshots: list[tuple[str, pd.DataFrame]] = []
    for end_idx in windows:
        block = returns.iloc[end_idx - lookback : end_idx]
        if len(block) < lookback:
            continue
        frontier = compute_cdar_efficient_frontier(
            returns=block,
            alpha=alpha,
            no_short=no_short,
            n_points=20,
            parallel=False,
        )
        label = str(block.index[-1].date())
        snapshots.append((label, frontier))
    return snapshots


def generate_surface_snapshots(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    no_short: bool = True,
    lookback: int = 252,
    step_mode: StepMode = "monthly",
    step_n: int = 21,
    max_snapshots: int = 300,
) -> list[tuple[str, pd.DataFrame]]:
    """Generate rolling mean-variance-CDaR surface snapshots."""
    windows = _rolling_windows(returns.index, lookback, step_mode, step_n)
    windows = windows[:max_snapshots]
    grid = [(1.0, 0.1, 0.1), (1.0, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 1.0, 1.0)]
    snapshots: list[tuple[str, pd.DataFrame]] = []
    for end_idx in windows:
        block = returns.iloc[end_idx - lookback : end_idx]
        if len(block) < lookback:
            continue
        surface = compute_mean_var_cdar_surface(
            returns=block,
            alpha=alpha,
            lambda_grid=grid,
            no_short=no_short,
            parallel=False,
        )
        label = str(block.index[-1].date())
        snapshots.append((label, surface))
    return snapshots


def _rolling_windows(
    index: pd.DatetimeIndex,
    lookback: int,
    step_mode: StepMode,
    step_n: int,
) -> list[int]:
    if len(index) < lookback:
        return []
    if step_mode == "custom":
        step = max(1, step_n)
        return list(range(lookback, len(index) + 1, step))

    periods = index.to_period("M" if step_mode == "monthly" else "Q")
    windows: list[int] = []
    last_period = None
    for i in range(lookback, len(index) + 1):
        p = periods[i - 1]
        if p != last_period:
            windows.append(i)
            last_period = p
    return windows


def _save_animation_with_fallback(
    anim: animation.FuncAnimation,
    save_path: str,
    fps: int,
    dpi: int,
) -> str:
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(str(out), writer=writer, dpi=dpi)
        return str(out)
    except Exception as exc:
        LOGGER.warning("FFmpeg save failed (%s). Falling back to GIF.", exc)
        gif_path = out.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(gif_path), writer=writer, dpi=dpi)
        return str(gif_path)
