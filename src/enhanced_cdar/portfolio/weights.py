"""Weight validation and bounds utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd



def validate_weights(
    weights: np.ndarray,
    no_short: bool = True,
    gross_exposure_limit: float | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    atol: float = 1e-8,
) -> None:
    """Validate portfolio weights against configured constraints."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D vector.")
    if not np.isfinite(w).all():
        raise ValueError("weights must contain finite values.")
    if not np.isclose(w.sum(), 1.0, atol=atol):
        raise ValueError("weights must sum to 1.")

    if no_short and (w < -atol).any():
        raise ValueError("No-short constraint violated by negative weight.")

    if lower_bounds is not None and (w < np.asarray(lower_bounds, dtype=float) - atol).any():
        raise ValueError("Lower bound constraint violated.")
    if upper_bounds is not None and (w > np.asarray(upper_bounds, dtype=float) + atol).any():
        raise ValueError("Upper bound constraint violated.")

    if gross_exposure_limit is not None:
        gross = float(np.abs(w).sum())
        if gross > gross_exposure_limit + atol:
            raise ValueError(
                f"Gross exposure constraint violated: {gross:.6f} > {gross_exposure_limit:.6f}."
            )



def default_bounds(n_assets: int, allow_short: bool) -> tuple[np.ndarray, np.ndarray]:
    """Build default per-asset bounds based on long-only vs long-short mode."""
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")
    if allow_short:
        lower = np.full(n_assets, -1.0)
        upper = np.full(n_assets, 1.0)
    else:
        lower = np.zeros(n_assets)
        upper = np.ones(n_assets)
    return lower, upper



def load_asset_bounds_csv(path: str | Path, assets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load per-asset bounds from CSV with columns: asset,lower,upper."""
    frame = pd.read_csv(path)
    required = {"asset", "lower", "upper"}
    if not required.issubset(frame.columns):
        raise ValueError("Bounds CSV must include columns: asset, lower, upper")

    bounds = frame.set_index("asset")
    lower = []
    upper = []
    for asset in assets:
        if asset not in bounds.index:
            raise ValueError(f"Missing bounds for asset '{asset}'")
        lo = float(bounds.loc[asset, "lower"])
        hi = float(bounds.loc[asset, "upper"])
        if lo > hi:
            raise ValueError(f"Invalid bounds for asset '{asset}': lower > upper")
        lower.append(lo)
        upper.append(hi)

    return np.array(lower, dtype=float), np.array(upper, dtype=float)
