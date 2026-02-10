"""Configuration models and helpers for enhanced CDaR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


Frequency = Literal["daily", "weekly", "monthly"]
ReturnMethod = Literal["simple", "log"]
MissingDataPolicy = Literal["ffill_then_drop", "drop", "raise"]


def infer_annualization_factor(frequency: Frequency) -> int:
    """Infer annualization factor from series frequency."""
    if frequency == "daily":
        return 252
    if frequency == "weekly":
        return 52
    return 12


class DataConfig(BaseModel):
    """Data loading and cleaning configuration."""

    model_config = ConfigDict(extra="forbid")

    frequency: Frequency = "daily"
    cache_dir: str = "cache"
    use_cache: bool = True
    refresh_cache: bool = False
    missing_data_policy: MissingDataPolicy = "ffill_then_drop"


class MetricsConfig(BaseModel):
    """Metrics computation configuration."""

    model_config = ConfigDict(extra="forbid")

    return_method: ReturnMethod = "simple"
    default_alpha: float = 0.95
    risk_free_rate_annual: float = 0.0
    rolling_cdar_window: int = 63


class OptimizationConfig(BaseModel):
    """Optimization configuration."""

    model_config = ConfigDict(extra="forbid")

    no_short: bool = True
    gross_exposure_limit: float = 2.0
    default_solver: str = "ECOS"
    fallback_solver: str = "SCS"


class BacktestConfig(BaseModel):
    """Backtest behavior configuration."""

    model_config = ConfigDict(extra="forbid")

    rebalance_calendar: Literal["none", "M", "Q"] = "none"
    rebalance_every_n_periods: int | None = None
    rebalance_mode: Literal["fixed", "dynamic"] = "fixed"


class AppConfig(BaseModel):
    """Top-level package configuration."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig = Field(default_factory=DataConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @property
    def annualization_factor(self) -> int:
        """Annualization factor inferred from configured frequency."""
        return infer_annualization_factor(self.data.frequency)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load raw YAML config into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML configuration must decode to a mapping object.")
    return data


def build_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Build application config with precedence: overrides > YAML > defaults."""
    merged: dict[str, Any] = {}
    if config_path is not None:
        merged.update(load_yaml_config(config_path))
    if overrides:
        merged = deep_merge(merged, overrides)
    return AppConfig.model_validate(merged)


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries recursively."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
