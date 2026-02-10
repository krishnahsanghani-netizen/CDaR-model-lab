"""Command line interface for enhanced CDaR toolkit."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import typer

from enhanced_cdar.config import AppConfig, build_config, merge_config
from enhanced_cdar.data.loaders import load_from_yfinance
from enhanced_cdar.data.preprocess import align_and_clean_prices
from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    compute_portfolio_returns,
    compute_returns,
)
from enhanced_cdar.metrics.cdar import compute_rolling_cdar
from enhanced_cdar.metrics.risk_metrics import (
    compute_cdar_parametric,
    compute_cvar,
    compute_var,
    summarize_core_metrics,
)
from enhanced_cdar.portfolio.backtest import run_backtest
from enhanced_cdar.portfolio.optimization import (
    compute_cdar_efficient_frontier,
    compute_mean_var_cdar_surface,
    optimize_portfolio_cdar,
)
from enhanced_cdar.portfolio.regime import (
    RegimeFrequency,
    compare_regime_cdar,
    compute_regime_metrics,
)
from enhanced_cdar.portfolio.scenario import evaluate_portfolio_scenarios, preset_scenarios
from enhanced_cdar.portfolio.weights import load_asset_bounds_csv
from enhanced_cdar.viz.frontier import plot_cdar_efficient_frontier
from enhanced_cdar.viz.surfaces import plot_mean_variance_cdar_surface
from enhanced_cdar.viz.underwater import plot_underwater
from enhanced_cdar.viz.animation import (
    StepMode,
    animate_frontier_over_time,
    animate_reactive_cdar_model,
    animate_surface_over_time,
    animate_underwater,
    generate_frontier_snapshots,
    generate_surface_snapshots,
)

app = typer.Typer(help="Enhanced CDaR modeling toolkit")
LOGGER = logging.getLogger(__name__)



def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")



def _load_config(config_path: str | None) -> AppConfig:
    return build_config(config_path=config_path) if config_path else AppConfig()



def _apply_overrides(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    return merge_config(config, overrides)



def _load_prices_csv(prices_csv: str) -> pd.DataFrame:
    frame = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    if frame.empty:
        raise typer.BadParameter("Input prices CSV is empty.")
    return frame



def _parse_weights(weights: str) -> np.ndarray:
    try:
        arr = np.array([float(x.strip()) for x in weights.split(",")], dtype=float)
    except ValueError as exc:
        raise typer.BadParameter("Failed parsing --weights as comma-separated floats.") from exc
    return arr



def _emit(payload: Any, out_format: str) -> None:
    if out_format == "json":
        typer.echo(json.dumps(payload, indent=2, default=_json_default))
    else:
        if isinstance(payload, dict):
            for key, value in payload.items():
                typer.echo(f"{key}: {value}")
        else:
            typer.echo(str(payload))



def _resolve_lambda_grid(
    lambda_grid_json: str | None,
    lambda_preset: str,
) -> list[tuple[float, float, float]]:
    """Resolve lambda-grid input from JSON or named preset."""
    presets: dict[str, list[tuple[float, float, float]]] = {
        "small": [
            (1.0, 0.1, 0.1),
            (1.0, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (0.5, 1.0, 1.0),
        ],
        "medium": [
            (1.0, 0.1, 0.1),
            (1.0, 0.3, 0.3),
            (1.0, 0.5, 0.5),
            (1.0, 0.8, 0.8),
            (1.0, 1.0, 1.0),
            (0.5, 1.0, 1.0),
            (0.3, 1.0, 1.2),
            (0.1, 1.2, 1.4),
        ],
    }
    if lambda_grid_json:
        grid = json.loads(lambda_grid_json)
        lambda_grid: list[tuple[float, float, float]] = []
        for row in grid:
            if not isinstance(row, list | tuple) or len(row) != 3:
                raise typer.BadParameter(
                    "Each lambda grid entry must be a 3-item list: "
                    "[lambda_cdar, lambda_var, lambda_return]."
                )
            lambda_grid.append((float(row[0]), float(row[1]), float(row[2])))
        return lambda_grid
    if lambda_preset not in presets:
        raise typer.BadParameter("lambda preset must be one of: small, medium")
    return presets[lambda_preset]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _resolve_prices_for_animation(
    run_dir: str | None,
    prices_csv: str | None,
) -> Path:
    """Resolve source prices path with run-dir precedence."""
    if run_dir:
        candidate = Path(run_dir) / "data" / "prices.csv"
        if candidate.exists():
            return candidate
        raise typer.BadParameter(f"Could not find prices file in run directory: {candidate}")
    if prices_csv:
        path = Path(prices_csv)
        if path.exists():
            return path
        raise typer.BadParameter(f"prices CSV not found: {path}")
    raise typer.BadParameter("Provide either --run-dir or --prices-csv.")


def _new_videos_dir(output_root: str = "runs") -> Path:
    run_dir = Path(output_root) / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    return videos_dir


def _write_video_manifest(
    videos_dir: Path,
    files: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "files": files,
    }
    if extra:
        payload.update(extra)
    manifest_path = videos_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return manifest_path


@app.command("fetch-data")
def fetch_data(
    tickers: str = typer.Option(..., help="Comma-separated list of symbols."),
    start: str = typer.Option(..., help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option(..., help="End date (YYYY-MM-DD)."),
    output: str = typer.Option(..., help="Output CSV path."),
    frequency: str = typer.Option("daily", help="daily|weekly|monthly"),
    refresh: bool = typer.Option(False, help="Refresh cache instead of using cached file."),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Download and save historical prices."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    cfg = _apply_overrides(cfg, {"data": {"frequency": frequency, "refresh_cache": refresh}})

    result = load_from_yfinance(
        tickers=tickers.split(","),
        start=start,
        end=end,
        frequency=cfg.data.frequency,
        cache_dir=cfg.data.cache_dir,
        use_cache=cfg.data.use_cache,
        refresh=cfg.data.refresh_cache,
    )
    cleaned = align_and_clean_prices(result.prices, cfg.data.missing_data_policy)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(result.metadata.__dict__, indent=2), encoding="utf-8")
    typer.echo(f"Saved data: {out_path}")
    typer.echo(f"Saved metadata: {meta_path}")


@app.command("analyze-portfolio")
def analyze_portfolio(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    weights: str = typer.Option(..., help="Comma-separated weights."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    frequency: str = typer.Option("daily", help="daily|weekly|monthly"),
    risk_free_rate: float = typer.Option(0.0, help="Annual risk-free rate."),
    benchmark_csv: str | None = typer.Option(
        None,
        help="Optional benchmark prices CSV (single-column price series).",
    ),
    rolling_cdar: bool = typer.Option(
        False,
        help="If set, compute rolling CDaR using --rolling-window.",
    ),
    rolling_window: int = typer.Option(
        63,
        help="Rolling CDaR window (periods); used when --rolling-cdar is enabled.",
    ),
    metric_mode: str = typer.Option(
        "historical",
        help="historical|parametric metric estimation mode.",
    ),
    parametric_dist: str = typer.Option(
        "normal",
        help="normal|student_t distribution for parametric mode.",
    ),
    student_t_df: float | None = typer.Option(
        None,
        help="Optional fixed Student-t degrees of freedom.",
    ),
    out_format: str = typer.Option("text", "--format", help="text|json"),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Compute portfolio risk/return summary from prices and weights."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    cfg = _apply_overrides(
        cfg,
        {
            "data": {"frequency": frequency},
            "metrics": {
                "default_alpha": alpha,
                "risk_free_rate_annual": risk_free_rate,
            },
        },
    )

    prices = _load_prices_csv(prices_csv)
    prices = align_and_clean_prices(prices, cfg.data.missing_data_policy)
    rets = compute_returns(prices, method=cfg.metrics.return_method)

    w = _parse_weights(weights)
    if len(w) != rets.shape[1]:
        raise typer.BadParameter("weights length does not match number of assets in prices file.")

    port = compute_portfolio_returns(rets, w)
    values = compute_cumulative_value(port)
    dd = compute_drawdown_curve(values)
    metrics = summarize_core_metrics(
        port,
        dd,
        alpha=cfg.metrics.default_alpha,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
    )
    if metric_mode == "parametric":
        metrics["var_parametric"] = compute_var(
            port,
            alpha=cfg.metrics.default_alpha,
            method="parametric",
            dist=parametric_dist,
            t_df=student_t_df,
        )
        metrics["cvar_parametric"] = compute_cvar(
            port,
            alpha=cfg.metrics.default_alpha,
            method="parametric",
            dist=parametric_dist,
            t_df=student_t_df,
        )
        metrics["cdar_parametric"] = compute_cdar_parametric(
            dd,
            alpha=cfg.metrics.default_alpha,
            dist=parametric_dist,
            t_df=student_t_df,
        )

    if rolling_cdar:
        rolling = compute_rolling_cdar(
            dd,
            alpha=cfg.metrics.default_alpha,
            window=rolling_window,
        )
        rolling_clean = rolling.dropna()
        metrics["rolling_cdar_last"] = (
            float(rolling_clean.iloc[-1]) if rolling_clean.size else 0.0
        )

    if benchmark_csv:
        benchmark_prices = _load_prices_csv(benchmark_csv)
        benchmark_prices = align_and_clean_prices(
            benchmark_prices,
            cfg.data.missing_data_policy,
        )
        if benchmark_prices.shape[1] != 1:
            raise typer.BadParameter("benchmark CSV must contain exactly one price column.")
        benchmark_returns = compute_returns(
            benchmark_prices,
            method=cfg.metrics.return_method,
        ).iloc[:, 0]
        backtest = run_backtest(
            returns=rets,
            weights=w,
            benchmark_returns=benchmark_returns,
            alpha=cfg.metrics.default_alpha,
            annualization_factor=cfg.annualization_factor,
            risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
        )
        if backtest.benchmark_metrics is not None:
            metrics.update(backtest.benchmark_metrics)

    metrics["drawdown_display_min_pct"] = float(dd.min() * 100.0)
    _emit(metrics, out_format)


@app.command("backtest")
def backtest_portfolio(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    weights: str = typer.Option(..., help="Comma-separated weights."),
    benchmark_csv: str | None = typer.Option(
        None,
        help="Optional benchmark prices CSV (single-column).",
    ),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    frequency: str = typer.Option("daily", help="daily|weekly|monthly"),
    risk_free_rate: float = typer.Option(0.0, help="Annual risk-free rate."),
    rebalance_calendar: str = typer.Option("none", help="none|M|Q"),
    rebalance_every_n_periods: int | None = typer.Option(
        None,
        help="Optional periodic rebalance every N periods.",
    ),
    rebalance_mode: str = typer.Option("fixed", help="fixed|dynamic"),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    gross_limit: float | None = typer.Option(
        None,
        help="Optional gross exposure limit for dynamic mode.",
    ),
    output_returns_csv: str | None = typer.Option(
        None,
        help="Optional output path to save backtest return series.",
    ),
    out_format: str = typer.Option("text", "--format", help="text|json"),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Run portfolio backtest with optional rebalancing and benchmark analytics."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    cfg = _apply_overrides(
        cfg,
        {
            "data": {"frequency": frequency},
            "metrics": {
                "default_alpha": alpha,
                "risk_free_rate_annual": risk_free_rate,
            },
        },
    )

    prices = align_and_clean_prices(_load_prices_csv(prices_csv), cfg.data.missing_data_policy)
    rets = compute_returns(prices, method=cfg.metrics.return_method)

    w = _parse_weights(weights)
    if len(w) != rets.shape[1]:
        raise typer.BadParameter("weights length does not match number of assets in prices file.")

    benchmark_returns: pd.Series | None = None
    if benchmark_csv:
        benchmark_prices = align_and_clean_prices(
            _load_prices_csv(benchmark_csv),
            cfg.data.missing_data_policy,
        )
        if benchmark_prices.shape[1] != 1:
            raise typer.BadParameter("benchmark CSV must contain exactly one price column.")
        benchmark_returns = compute_returns(
            benchmark_prices,
            method=cfg.metrics.return_method,
        ).iloc[:, 0]

    dynamic_optimizer = None
    if rebalance_mode == "dynamic":
        effective_gross = gross_limit
        if effective_gross is None and not no_short:
            effective_gross = cfg.optimization.gross_exposure_limit

        def _dynamic_optimizer(history_returns: pd.DataFrame) -> np.ndarray:
            result = optimize_portfolio_cdar(
                returns=history_returns,
                alpha=cfg.metrics.default_alpha,
                no_short=no_short,
                weight_bounds=None,
                gross_exposure_limit=effective_gross,
                solver=cfg.optimization.default_solver,
                annualization_factor=cfg.annualization_factor,
                risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
            )
            if not np.isfinite(result["weights"]).all():
                raise ValueError(
                    "Dynamic optimizer failed to produce valid weights. "
                    "Try relaxing constraints."
                )
            return np.asarray(result["weights"], dtype=float)

        dynamic_optimizer = _dynamic_optimizer

    result = run_backtest(
        returns=rets,
        weights=w,
        rebalance_calendar=rebalance_calendar,
        rebalance_every_n_periods=rebalance_every_n_periods,
        rebalance_mode=rebalance_mode,
        benchmark_returns=benchmark_returns,
        alpha=cfg.metrics.default_alpha,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
        no_short=no_short,
        gross_exposure_limit=gross_limit,
        dynamic_optimizer=dynamic_optimizer,
    )

    if output_returns_csv:
        out = pd.DataFrame(
            {
                "portfolio_return": result.portfolio_returns,
                "portfolio_value": result.portfolio_values,
                "drawdown": result.drawdown,
            }
        )
        Path(output_returns_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_returns_csv, index=True)

    payload: dict[str, Any] = {"metrics": result.metrics}
    if result.benchmark_metrics is not None:
        payload["benchmark_metrics"] = result.benchmark_metrics
    if output_returns_csv:
        payload["returns_csv"] = output_returns_csv

    _emit(payload, out_format)


@app.command("optimize-cdar")
def optimize_cdar(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    target_return: float | None = typer.Option(None, help="Optional target return."),
    bounds_csv: str | None = typer.Option(
        None,
        help="Optional per-asset bounds CSV with columns asset,lower,upper.",
    ),
    gross_limit: float | None = typer.Option(
        None,
        help="Optional gross exposure limit. Defaults to config in long-short mode.",
    ),
    solver: str | None = typer.Option(
        None,
        help="Optional solver override (e.g., ECOS, SCS).",
    ),
    out_format: str = typer.Option("text", "--format", help="text|json"),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Optimize portfolio for CDaR subject to constraints."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    prices = align_and_clean_prices(_load_prices_csv(prices_csv), cfg.data.missing_data_policy)
    rets = compute_returns(prices, method=cfg.metrics.return_method)

    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    if bounds_csv:
        lower_bounds, upper_bounds = load_asset_bounds_csv(bounds_csv, list(rets.columns))

    effective_gross_limit = gross_limit
    if effective_gross_limit is None and not no_short:
        effective_gross_limit = cfg.optimization.gross_exposure_limit

    result = optimize_portfolio_cdar(
        returns=rets,
        alpha=alpha,
        no_short=no_short,
        target_return=target_return,
        per_asset_lower=lower_bounds,
        per_asset_upper=upper_bounds,
        weight_bounds=None,
        gross_exposure_limit=effective_gross_limit,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
        solver=solver or cfg.optimization.default_solver,
    )
    _emit(result, out_format)


@app.command("frontier")
def frontier(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    n_points: int = typer.Option(20, help="Number of frontier points."),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    gross_limit: float | None = typer.Option(
        None,
        help="Optional gross exposure limit for long-short mode.",
    ),
    return_min: float | None = typer.Option(
        None,
        help="Optional minimum target return for frontier grid.",
    ),
    return_max: float | None = typer.Option(
        None,
        help="Optional maximum target return for frontier grid.",
    ),
    parallel: bool = typer.Option(True, "--parallel/--no-parallel"),
    n_jobs: int = typer.Option(-1, help="Parallel workers for frontier solve."),
    output_csv: str | None = typer.Option(None, help="Optional output CSV path."),
    plot_path: str | None = typer.Option(None, help="Optional plot output path (.html/.png/.svg)."),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Compute and optionally plot the Mean-CDaR frontier."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    prices = align_and_clean_prices(_load_prices_csv(prices_csv), cfg.data.missing_data_policy)
    rets = compute_returns(prices, method=cfg.metrics.return_method)
    return_range = (
        (return_min, return_max)
        if return_min is not None and return_max is not None
        else None
    )
    effective_gross = gross_limit
    if effective_gross is None and not no_short:
        effective_gross = cfg.optimization.gross_exposure_limit

    frontier_df = compute_cdar_efficient_frontier(
        returns=rets,
        alpha=alpha,
        no_short=no_short,
        n_points=n_points,
        return_range=return_range,
        gross_exposure_limit=effective_gross,
        parallel=parallel,
        n_jobs=n_jobs,
    )

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        frontier_df.to_csv(output_csv, index=False)
        typer.echo(f"Saved frontier CSV: {output_csv}")

    if plot_path:
        plot_cdar_efficient_frontier(frontier_df, show=False, save_path=plot_path)
        typer.echo(f"Saved frontier plot: {plot_path}")

    frontier_preview = frontier_df[
        ["target_return", "achieved_return", "cdar", "status"]
    ].to_string(index=False)
    typer.echo(frontier_preview)


@app.command("surface")
def surface(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    lambda_grid_json: str | None = typer.Option(
        None,
        help="Optional JSON list of [lambda_cdar, lambda_var, lambda_return].",
    ),
    lambda_preset: str = typer.Option("small", help="small|medium preset grid."),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    gross_limit: float | None = typer.Option(
        None,
        help="Optional gross exposure limit for long-short mode.",
    ),
    parallel: bool = typer.Option(True, "--parallel/--no-parallel"),
    n_jobs: int = typer.Option(-1, help="Parallel workers for surface solve."),
    output_csv: str | None = typer.Option(None, help="Optional output CSV path."),
    plot_path: str | None = typer.Option(None, help="Optional plot output path (.html/.png/.svg)."),
    mode: str = typer.Option("3d", help="3d|2d-projections"),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Compute and optionally plot mean-variance-CDaR surface."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    prices = align_and_clean_prices(_load_prices_csv(prices_csv), cfg.data.missing_data_policy)
    rets = compute_returns(prices, method=cfg.metrics.return_method)
    lambda_grid = _resolve_lambda_grid(lambda_grid_json, lambda_preset)
    effective_gross = gross_limit
    if effective_gross is None and not no_short:
        effective_gross = cfg.optimization.gross_exposure_limit

    surface_df = compute_mean_var_cdar_surface(
        returns=rets,
        alpha=alpha,
        lambda_grid=lambda_grid,
        no_short=no_short,
        gross_exposure_limit=effective_gross,
        parallel=parallel,
        n_jobs=n_jobs,
    )

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        surface_df.to_csv(output_csv, index=False)
        typer.echo(f"Saved surface CSV: {output_csv}")

    if plot_path:
        plot_mean_variance_cdar_surface(surface_df, mode=mode, show=False, save_path=plot_path)
        typer.echo(f"Saved surface plot: {plot_path}")

    surface_preview = surface_df[
        ["expected_return", "volatility", "cdar", "status"]
    ].head().to_string(index=False)
    typer.echo(surface_preview)


@app.command("run-pipeline")
def run_pipeline(
    tickers: str = typer.Option("SPY,AGG,GLD,QQQ", help="Comma-separated list of symbols."),
    years: int = typer.Option(5, help="Lookback years for default date range."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    output_root: str = typer.Option("runs", help="Base output directory."),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Fetch data, optimize CDaR, and generate core artifacts."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)

    end_date = date.today()
    start_date = date(end_date.year - years, end_date.month, max(1, min(end_date.day, 28)))

    run_dir = Path(output_root) / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    data_dir = run_dir / "data"
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"
    for d in (data_dir, results_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    result = load_from_yfinance(
        tickers=tickers.split(","),
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        frequency=cfg.data.frequency,
        cache_dir=cfg.data.cache_dir,
        use_cache=cfg.data.use_cache,
        refresh=cfg.data.refresh_cache,
    )
    prices = align_and_clean_prices(result.prices, cfg.data.missing_data_policy)
    prices_path = data_dir / "prices.csv"
    prices.to_csv(prices_path)
    (data_dir / "metadata.json").write_text(
        json.dumps(result.metadata.__dict__, indent=2),
        encoding="utf-8",
    )

    rets = compute_returns(prices, method=cfg.metrics.return_method)
    opt = optimize_portfolio_cdar(
        returns=rets,
        alpha=alpha,
        no_short=no_short,
        gross_exposure_limit=cfg.optimization.gross_exposure_limit,
        weight_bounds=None,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
    )

    frontier_df = compute_cdar_efficient_frontier(
        rets,
        alpha=alpha,
        n_points=20,
        no_short=no_short,
    )
    lambda_grid = _resolve_lambda_grid(None, "small")
    surface_df = compute_mean_var_cdar_surface(
        rets,
        alpha=alpha,
        lambda_grid=lambda_grid,
        no_short=no_short,
        gross_exposure_limit=cfg.optimization.gross_exposure_limit,
    )

    port = compute_portfolio_returns(rets, np.asarray(opt["weights"], dtype=float))
    values = compute_cumulative_value(port)

    frontier_df.to_csv(results_dir / "frontier.csv", index=False)
    surface_df.to_csv(results_dir / "surface.csv", index=False)
    (results_dir / "optimization.json").write_text(
        json.dumps(opt, indent=2, default=_json_default),
        encoding="utf-8",
    )
    summary = {
        "run_dir": str(run_dir),
        "alpha": alpha,
        "no_short": no_short,
        "n_assets": int(rets.shape[1]),
        "n_periods": int(rets.shape[0]),
        "opt_status": opt.get("status"),
        "opt_cdar": opt.get("cdar"),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )

    plot_underwater(values, show=False, save_path=str(plots_dir / "underwater.html"))
    plot_cdar_efficient_frontier(
        frontier_df,
        show=False,
        save_path=str(plots_dir / "frontier.html"),
    )
    plot_mean_variance_cdar_surface(
        surface_df,
        mode="3d",
        show=False,
        save_path=str(plots_dir / "surface_3d.html"),
    )
    plot_mean_variance_cdar_surface(
        surface_df,
        mode="2d-projections",
        show=False,
        save_path=str(plots_dir / "surface_2d.html"),
    )

    typer.echo(f"Run artifacts saved to: {run_dir}")


@app.command("ui")
def launch_ui(
    port: int = typer.Option(8501, help="Port for Streamlit app."),
    server_headless: str = typer.Option(
        "true",
        help="Run Streamlit in headless mode (true|false).",
    ),
    streamlit_args: list[str] | None = typer.Argument(
        None,
        help="Additional args forwarded to Streamlit (e.g. --browser.gatherUsageStats false).",
    ),
) -> None:
    """Launch CDaR Lab Streamlit UI."""
    if server_headless.lower() not in {"true", "false"}:
        raise typer.BadParameter("--server-headless must be true or false.")
    repo_root = Path(__file__).resolve().parents[2]
    app_path = repo_root / "ui" / "streamlit_app.py"
    if not app_path.exists():
        raise typer.BadParameter(f"Streamlit app not found at: {app_path}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        server_headless.lower(),
    ]
    cmd.extend(streamlit_args or [])
    raise typer.Exit(subprocess.call(cmd))


@app.command("animate-underwater")
def animate_underwater_cmd(
    run_dir: str | None = typer.Option(
        None,
        help="Existing run directory (preferred). If provided, data/prices.csv is used.",
    ),
    prices_csv: str | None = typer.Option(None, help="Input prices CSV."),
    weights: str | None = typer.Option(
        None,
        help=(
            "Optional comma-separated weights. "
            "Defaults to run optimization weights or equal-weight."
        ),
    ),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    fps: int = typer.Option(12, help="Animation frames per second."),
    dpi: int = typer.Option(120, help="Render DPI."),
    max_frames: int = typer.Option(300, help="Maximum frames after downsampling."),
    output_root: str = typer.Option("runs", help="Root directory for generated video runs."),
    config: str | None = typer.Option(None, help="Path to YAML config."),
) -> None:
    """Generate underwater animation MP4/GIF with ffmpeg fallback."""
    cfg = _load_config(config)
    prices_path = _resolve_prices_for_animation(run_dir, prices_csv)
    prices = align_and_clean_prices(
        _load_prices_csv(str(prices_path)),
        cfg.data.missing_data_policy,
    )
    returns = compute_returns(prices, method=cfg.metrics.return_method)

    if weights:
        weight_vec = _parse_weights(weights)
    elif run_dir and (Path(run_dir) / "results" / "optimization.json").exists():
        opt = json.loads(
            (Path(run_dir) / "results" / "optimization.json").read_text(
                encoding="utf-8"
            )
        )
        weight_vec = np.asarray(opt.get("weights", []), dtype=float)
    else:
        weight_vec = np.full(returns.shape[1], 1.0 / returns.shape[1])
    if len(weight_vec) != returns.shape[1]:
        raise typer.BadParameter("Weights length does not match number of assets.")

    port_returns = compute_portfolio_returns(returns, weight_vec)
    values = compute_cumulative_value(port_returns)
    drawdown = compute_drawdown_curve(values)

    videos_dir = _new_videos_dir(output_root=output_root)
    output_path = videos_dir / "underwater.mp4"
    generated = animate_underwater(
        values=values,
        drawdown=drawdown,
        fps=fps,
        dpi=dpi,
        save_path=str(output_path),
        max_frames=max_frames,
    )
    manifest = _write_video_manifest(
        videos_dir,
        files=[{"type": "underwater", "path": generated}],
        extra={"alpha": alpha, "fps": fps, "dpi": dpi, "max_frames": max_frames},
    )
    typer.echo(f"Generated underwater animation: {generated}")
    typer.echo(f"Manifest: {manifest}")


@app.command("animate-frontier")
def animate_frontier_cmd(
    run_dir: str | None = typer.Option(
        None,
        help="Existing run directory (preferred). If provided, data/prices.csv is used.",
    ),
    prices_csv: str | None = typer.Option(None, help="Input prices CSV."),
    snapshots_json: str | None = typer.Option(
        None,
        help="Optional JSON file with [{\"label\": ..., \"csv\": ...}] precomputed snapshots.",
    ),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    lookback: int = typer.Option(252, help="Rolling lookback window (periods)."),
    step_mode: str = typer.Option("monthly", help="monthly|quarterly|custom"),
    step_n: int = typer.Option(21, help="Step size used when step-mode=custom."),
    fps: int = typer.Option(10, help="Animation frames per second."),
    dpi: int = typer.Option(120, help="Render DPI."),
    max_frames: int = typer.Option(300, help="Maximum frames after downsampling."),
    output_root: str = typer.Option("runs", help="Root directory for generated video runs."),
    config: str | None = typer.Option(None, help="Path to YAML config."),
) -> None:
    """Generate efficient frontier evolution animation over rolling windows."""
    if step_mode not in {"monthly", "quarterly", "custom"}:
        raise typer.BadParameter("step-mode must be one of monthly|quarterly|custom.")
    cfg = _load_config(config)
    prices_path = _resolve_prices_for_animation(run_dir, prices_csv)
    prices = align_and_clean_prices(
        _load_prices_csv(str(prices_path)),
        cfg.data.missing_data_policy,
    )
    returns = compute_returns(prices, method=cfg.metrics.return_method)

    snapshots: list[tuple[str, pd.DataFrame]]
    if snapshots_json:
        spec = json.loads(Path(snapshots_json).read_text(encoding="utf-8"))
        snapshots = []
        for row in spec:
            label = str(row["label"])
            csv_path = Path(str(row["csv"]))
            snapshots.append((label, pd.read_csv(csv_path)))
    else:
        snapshots = generate_frontier_snapshots(
            returns=returns,
            alpha=alpha,
            no_short=no_short,
            lookback=lookback,
            step_mode=cast(StepMode, step_mode),
            step_n=step_n,
            max_snapshots=max_frames,
        )
    if not snapshots:
        raise typer.BadParameter("No frontier snapshots were produced.")

    videos_dir = _new_videos_dir(output_root=output_root)
    output_path = videos_dir / "frontier_over_time.mp4"
    generated = animate_frontier_over_time(
        frontier_snapshots=snapshots,
        fps=fps,
        dpi=dpi,
        save_path=str(output_path),
        max_frames=max_frames,
    )
    manifest = _write_video_manifest(
        videos_dir,
        files=[{"type": "frontier", "path": generated}],
        extra={
            "alpha": alpha,
            "fps": fps,
            "dpi": dpi,
            "max_frames": max_frames,
            "lookback": lookback,
            "step_mode": step_mode,
            "step_n": step_n,
        },
    )
    typer.echo(f"Generated frontier animation: {generated}")
    typer.echo(f"Manifest: {manifest}")


@app.command("animate-surface")
def animate_surface_cmd(
    run_dir: str | None = typer.Option(
        None,
        help="Existing run directory (preferred). If provided, data/prices.csv is used.",
    ),
    prices_csv: str | None = typer.Option(None, help="Input prices CSV."),
    snapshots_json: str | None = typer.Option(
        None,
        help="Optional JSON file with [{\"label\": ..., \"csv\": ...}] precomputed snapshots.",
    ),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    lookback: int = typer.Option(252, help="Rolling lookback window (periods)."),
    step_mode: str = typer.Option("monthly", help="monthly|quarterly|custom"),
    step_n: int = typer.Option(21, help="Step size used when step-mode=custom."),
    fps: int = typer.Option(8, help="Animation frames per second."),
    dpi: int = typer.Option(120, help="Render DPI."),
    max_frames: int = typer.Option(300, help="Maximum frames after downsampling."),
    output_root: str = typer.Option("runs", help="Root directory for generated video runs."),
    config: str | None = typer.Option(None, help="Path to YAML config."),
) -> None:
    """Generate mean-variance-CDaR surface evolution animation over rolling windows."""
    if step_mode not in {"monthly", "quarterly", "custom"}:
        raise typer.BadParameter("step-mode must be one of monthly|quarterly|custom.")
    cfg = _load_config(config)
    prices_path = _resolve_prices_for_animation(run_dir, prices_csv)
    prices = align_and_clean_prices(
        _load_prices_csv(str(prices_path)),
        cfg.data.missing_data_policy,
    )
    returns = compute_returns(prices, method=cfg.metrics.return_method)

    snapshots: list[tuple[str, pd.DataFrame]]
    if snapshots_json:
        spec = json.loads(Path(snapshots_json).read_text(encoding="utf-8"))
        snapshots = []
        for row in spec:
            label = str(row["label"])
            csv_path = Path(str(row["csv"]))
            snapshots.append((label, pd.read_csv(csv_path)))
    else:
        snapshots = generate_surface_snapshots(
            returns=returns,
            alpha=alpha,
            no_short=no_short,
            lookback=lookback,
            step_mode=cast(StepMode, step_mode),
            step_n=step_n,
            max_snapshots=max_frames,
        )
    if not snapshots:
        raise typer.BadParameter("No surface snapshots were produced.")

    videos_dir = _new_videos_dir(output_root=output_root)
    output_path = videos_dir / "surface_over_time.mp4"
    generated = animate_surface_over_time(
        surface_snapshots=snapshots,
        fps=fps,
        dpi=dpi,
        save_path=str(output_path),
        max_frames=max_frames,
    )
    manifest = _write_video_manifest(
        videos_dir,
        files=[{"type": "surface", "path": generated}],
        extra={
            "alpha": alpha,
            "fps": fps,
            "dpi": dpi,
            "max_frames": max_frames,
            "lookback": lookback,
            "step_mode": step_mode,
            "step_n": step_n,
        },
    )
    typer.echo(f"Generated surface animation: {generated}")
    typer.echo(f"Manifest: {manifest}")


@app.command("animate-model")
def animate_model_cmd(
    run_dir: str | None = typer.Option(
        None,
        help="Existing run directory (preferred). If provided, data/prices.csv is used.",
    ),
    prices_csv: str | None = typer.Option(None, help="Input prices CSV."),
    weights: str | None = typer.Option(
        None,
        help="Optional comma-separated weights. Defaults to equal-weight.",
    ),
    fps: int = typer.Option(16, help="Animation frames per second."),
    dpi: int = typer.Option(140, help="Render DPI."),
    max_frames: int = typer.Option(220, help="Maximum frames after downsampling."),
    output_root: str = typer.Option("runs", help="Root directory for generated video runs."),
    config: str | None = typer.Option(None, help="Path to YAML config."),
) -> None:
    """Generate cinematic reactive 3D risk model animation (MP4/GIF)."""
    cfg = _load_config(config)
    prices_path = _resolve_prices_for_animation(run_dir, prices_csv)
    prices = align_and_clean_prices(
        _load_prices_csv(str(prices_path)),
        cfg.data.missing_data_policy,
    )
    returns = compute_returns(prices, method=cfg.metrics.return_method)

    if weights:
        weight_vec = _parse_weights(weights)
    else:
        weight_vec = np.full(returns.shape[1], 1.0 / returns.shape[1])
    if len(weight_vec) != returns.shape[1]:
        raise typer.BadParameter("Weights length does not match number of assets.")

    portfolio_returns = compute_portfolio_returns(returns, weight_vec)
    values = compute_cumulative_value(portfolio_returns)
    drawdown = compute_drawdown_curve(values)

    videos_dir = _new_videos_dir(output_root=output_root)
    output_path = videos_dir / "reactive_cdar_model.mp4"
    generated = animate_reactive_cdar_model(
        drawdown=drawdown,
        returns=portfolio_returns,
        fps=fps,
        dpi=dpi,
        save_path=str(output_path),
        max_frames=max_frames,
    )
    manifest = _write_video_manifest(
        videos_dir,
        files=[{"type": "reactive_model", "path": generated}],
        extra={"fps": fps, "dpi": dpi, "max_frames": max_frames},
    )
    typer.echo(f"Generated reactive model animation: {generated}")
    typer.echo(f"Manifest: {manifest}")


@app.command("scenario")
def scenario_analysis(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    weights: str = typer.Option(..., help="Comma-separated weights."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    scenarios_json: str | None = typer.Option(
        None,
        help=(
            "Optional scenario spec JSON mapping names to "
            "{asset_shocks, global_shift}."
        ),
    ),
    scenario_preset: str = typer.Option(
        "basic",
        help="basic preset or custom (requires --scenarios-json).",
    ),
    output_csv: str | None = typer.Option(
        None,
        help="Optional output CSV path for scenario metrics.",
    ),
    out_format: str = typer.Option("text", "--format", help="text|json"),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Run portfolio scenario/stress analysis and report metrics by scenario."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    prices = align_and_clean_prices(_load_prices_csv(prices_csv), cfg.data.missing_data_policy)
    rets = compute_returns(prices, method=cfg.metrics.return_method)

    w = _parse_weights(weights)
    if len(w) != rets.shape[1]:
        raise typer.BadParameter("weights length does not match number of assets in prices file.")

    if scenarios_json:
        scenarios = json.loads(scenarios_json)
        if not isinstance(scenarios, dict):
            raise typer.BadParameter("--scenarios-json must decode to an object mapping.")
    else:
        if scenario_preset != "basic":
            raise typer.BadParameter("Only 'basic' preset is supported without --scenarios-json.")
        scenarios = preset_scenarios()

    scenario_df = evaluate_portfolio_scenarios(
        returns=rets,
        weights=w,
        scenarios=scenarios,
        alpha=alpha,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
    )

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        scenario_df.to_csv(output_csv, index=False)

    payload = {
        "scenario_count": int(len(scenario_df)),
        "results": scenario_df.to_dict(orient="records"),
    }
    if output_csv:
        payload["output_csv"] = output_csv
    _emit(payload, out_format)


@app.command("regime")
def regime_analysis(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    weights: str = typer.Option(..., help="Comma-separated weights."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    regime_frequency: str = typer.Option("Y", help="Y|Q|M regime split."),
    min_periods: int = typer.Option(10, help="Minimum observations per regime."),
    output_csv: str | None = typer.Option(
        None,
        help="Optional output CSV path for regime metrics.",
    ),
    out_format: str = typer.Option("text", "--format", help="text|json"),
    config: str | None = typer.Option(None, help="Path to YAML config."),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Compare CDaR and related metrics across time regimes."""
    _configure_logging(verbose, quiet)
    cfg = _load_config(config)
    prices = align_and_clean_prices(_load_prices_csv(prices_csv), cfg.data.missing_data_policy)
    rets = compute_returns(prices, method=cfg.metrics.return_method)

    w = _parse_weights(weights)
    if len(w) != rets.shape[1]:
        raise typer.BadParameter("weights length does not match number of assets in prices file.")
    if regime_frequency not in {"Y", "Q", "M"}:
        raise typer.BadParameter("regime_frequency must be one of Y, Q, M.")

    regime_freq = cast(RegimeFrequency, regime_frequency)
    regime_df = compute_regime_metrics(
        returns=rets,
        weights=w,
        alpha=alpha,
        regime_frequency=regime_freq,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
        min_periods=min_periods,
    )
    comparison = compare_regime_cdar(regime_df)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        regime_df.to_csv(output_csv, index=False)

    payload = {
        "regime_count": int(len(regime_df)),
        "comparison": comparison,
        "results": regime_df.to_dict(orient="records"),
    }
    if output_csv:
        payload["output_csv"] = output_csv
    _emit(payload, out_format)


if __name__ == "__main__":
    app()
