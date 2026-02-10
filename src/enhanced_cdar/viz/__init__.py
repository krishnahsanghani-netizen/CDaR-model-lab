"""Visualization subpackage exports."""

from enhanced_cdar.viz.animation import (
    animate_frontier_over_time,
    animate_reactive_cdar_model,
    animate_surface_over_time,
    animate_underwater,
    generate_frontier_snapshots,
    generate_surface_snapshots,
    select_frame_indices,
)
from enhanced_cdar.viz.frontier import (
    make_cdar_frontier_figure,
    plot_cdar_efficient_frontier,
)
from enhanced_cdar.viz.surfaces import (
    make_cdar_reactive_model_figure,
    make_mean_variance_cdar_surface_figure,
    plot_mean_variance_cdar_surface,
)
from enhanced_cdar.viz.underwater import make_underwater_figure, plot_underwater

__all__ = [
    "plot_underwater",
    "make_underwater_figure",
    "plot_cdar_efficient_frontier",
    "make_cdar_frontier_figure",
    "plot_mean_variance_cdar_surface",
    "make_cdar_reactive_model_figure",
    "make_mean_variance_cdar_surface_figure",
    "animate_underwater",
    "animate_frontier_over_time",
    "animate_surface_over_time",
    "animate_reactive_cdar_model",
    "generate_frontier_snapshots",
    "generate_surface_snapshots",
    "select_frame_indices",
]
