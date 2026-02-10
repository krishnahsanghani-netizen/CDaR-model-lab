"""Command line interface for enhanced CDaR toolkit."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from enhanced_cdar.config import AppConfig, build_config
from enhanced_cdar.data.loaders import load_from_yfinance
from enhanced_cdar.data.preprocess import align_and_clean_prices
from enhanced_cdar.metrics.drawdown import (
    compute_cumulative_value,
    compute_drawdown_curve,
    compute_portfolio_returns,
    compute_returns,
)
from enhanced_cdar.metrics.risk_metrics import summarize_core_metrics
from enhanced_cdar.portfolio.optimization import (
    compute_cdar_efficient_frontier,
    compute_mean_var_cdar_surface,
    optimize_portfolio_cdar,
)
from enhanced_cdar.viz.frontier import plot_cdar_efficient_frontier
from enhanced_cdar.viz.surfaces import plot_mean_variance_cdar_surface
from enhanced_cdar.viz.underwater import plot_underwater

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
    if not overrides:
        return config
    return config.model_copy(update=overrides)



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



def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


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
        {"data": {"frequency": frequency}, "metrics": {"default_alpha": alpha, "risk_free_rate_annual": risk_free_rate}},
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
    metrics["drawdown_display_min_pct"] = float(dd.min() * 100.0)
    _emit(metrics, out_format)


@app.command("optimize-cdar")
def optimize_cdar(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    no_short: bool = typer.Option(True, "--no-short/--allow-short"),
    target_return: float | None = typer.Option(None, help="Optional target return."),
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

    gross_limit = None if no_short else cfg.optimization.gross_exposure_limit
    result = optimize_portfolio_cdar(
        returns=rets,
        alpha=alpha,
        no_short=no_short,
        target_return=target_return,
        weight_bounds=None,
        gross_exposure_limit=gross_limit,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
    )
    _emit(result, out_format)


@app.command("frontier")
def frontier(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    n_points: int = typer.Option(20, help="Number of frontier points."),
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

    frontier_df = compute_cdar_efficient_frontier(
        returns=rets,
        alpha=alpha,
        no_short=cfg.optimization.no_short,
        n_points=n_points,
        parallel=True,
    )

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        frontier_df.to_csv(output_csv, index=False)
        typer.echo(f"Saved frontier CSV: {output_csv}")

    if plot_path:
        plot_cdar_efficient_frontier(frontier_df, show=False, save_path=plot_path)
        typer.echo(f"Saved frontier plot: {plot_path}")

    typer.echo(frontier_df[["target_return", "achieved_return", "cdar", "status"]].to_string(index=False))


@app.command("surface")
def surface(
    prices_csv: str = typer.Option(..., help="Input prices CSV."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
    lambda_grid_json: str = typer.Option(..., help="JSON list of [lambda_cdar, lambda_var, lambda_return]."),
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

    grid = json.loads(lambda_grid_json)
    lambda_grid = [tuple(map(float, row)) for row in grid]

    surface_df = compute_mean_var_cdar_surface(
        returns=rets,
        alpha=alpha,
        lambda_grid=lambda_grid,
        no_short=cfg.optimization.no_short,
        gross_exposure_limit=cfg.optimization.gross_exposure_limit,
    )

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        surface_df.to_csv(output_csv, index=False)
        typer.echo(f"Saved surface CSV: {output_csv}")

    if plot_path:
        plot_mean_variance_cdar_surface(surface_df, mode=mode, show=False, save_path=plot_path)
        typer.echo(f"Saved surface plot: {plot_path}")

    typer.echo(surface_df[["expected_return", "volatility", "cdar", "status"]].head().to_string(index=False))


@app.command("run-pipeline")
def run_pipeline(
    tickers: str = typer.Option("SPY,AGG,GLD,QQQ", help="Comma-separated list of symbols."),
    years: int = typer.Option(5, help="Lookback years for default date range."),
    alpha: float = typer.Option(0.95, help="CDaR alpha."),
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
    (data_dir / "metadata.json").write_text(json.dumps(result.metadata.__dict__, indent=2), encoding="utf-8")

    rets = compute_returns(prices, method=cfg.metrics.return_method)
    opt = optimize_portfolio_cdar(
        returns=rets,
        alpha=alpha,
        no_short=cfg.optimization.no_short,
        gross_exposure_limit=cfg.optimization.gross_exposure_limit,
        weight_bounds=None,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=cfg.metrics.risk_free_rate_annual,
    )

    frontier_df = compute_cdar_efficient_frontier(rets, alpha=alpha, n_points=20, no_short=cfg.optimization.no_short)
    lambda_grid = [(1.0, 0.1, 0.1), (1.0, 0.5, 0.5), (1.0, 1.0, 1.0), (0.5, 1.0, 1.0)]
    surface_df = compute_mean_var_cdar_surface(
        rets,
        alpha=alpha,
        lambda_grid=lambda_grid,
        no_short=cfg.optimization.no_short,
        gross_exposure_limit=cfg.optimization.gross_exposure_limit,
    )

    port = compute_portfolio_returns(rets, np.asarray(opt["weights"], dtype=float))
    values = compute_cumulative_value(port)

    frontier_df.to_csv(results_dir / "frontier.csv", index=False)
    surface_df.to_csv(results_dir / "surface.csv", index=False)
    (results_dir / "optimization.json").write_text(json.dumps(opt, indent=2, default=_json_default), encoding="utf-8")

    plot_underwater(values, show=False, save_path=str(plots_dir / "underwater.html"))
    plot_cdar_efficient_frontier(frontier_df, show=False, save_path=str(plots_dir / "frontier.html"))
    plot_mean_variance_cdar_surface(surface_df, mode="3d", show=False, save_path=str(plots_dir / "surface_3d.html"))
    plot_mean_variance_cdar_surface(
        surface_df,
        mode="2d-projections",
        show=False,
        save_path=str(plots_dir / "surface_2d.html"),
    )

    typer.echo(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    app()
