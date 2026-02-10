from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from enhanced_cdar.cli import app


runner = CliRunner()


def test_analyze_portfolio_command(tmp_path: Path):
    prices = pd.DataFrame(
        {
            "A": [100, 101, 99, 102],
            "B": [50, 50.5, 50.2, 51],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    path = tmp_path / "prices.csv"
    prices.to_csv(path)

    result = runner.invoke(
        app,
        [
            "analyze-portfolio",
            "--prices-csv",
            str(path),
            "--weights",
            "0.6,0.4",
            "--format",
            "json",
            "--rolling-cdar",
            "--rolling-window",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "cdar" in result.stdout
    assert "rolling_cdar_last" in result.stdout


def test_analyze_portfolio_with_benchmark(tmp_path: Path):
    prices = pd.DataFrame(
        {
            "A": [100, 101, 99, 102],
            "B": [50, 50.5, 50.2, 51],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    benchmark = pd.DataFrame(
        {"SPY": [470, 472, 468, 474]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )

    prices_path = tmp_path / "prices.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    prices.to_csv(prices_path)
    benchmark.to_csv(benchmark_path)

    result = runner.invoke(
        app,
        [
            "analyze-portfolio",
            "--prices-csv",
            str(prices_path),
            "--weights",
            "0.6,0.4",
            "--benchmark-csv",
            str(benchmark_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert "tracking_error" in result.stdout
    assert "information_ratio" in result.stdout

def test_backtest_command_with_benchmark_and_rebalance(tmp_path: Path):
    prices = pd.DataFrame(
        {
            "A": [100, 101, 99, 102, 103, 104],
            "B": [50, 50.5, 50.2, 51, 51.2, 51.6],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )
    benchmark = pd.DataFrame(
        {"SPY": [470, 472, 468, 474, 475, 476]},
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    prices_path = tmp_path / "prices.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    out_path = tmp_path / "backtest.csv"
    prices.to_csv(prices_path)
    benchmark.to_csv(benchmark_path)

    result = runner.invoke(
        app,
        [
            "backtest",
            "--prices-csv",
            str(prices_path),
            "--weights",
            "0.5,0.5",
            "--benchmark-csv",
            str(benchmark_path),
            "--rebalance-calendar",
            "M",
            "--rebalance-every-n-periods",
            "2",
            "--output-returns-csv",
            str(out_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert "metrics" in result.stdout
    assert "benchmark_metrics" in result.stdout
    assert out_path.exists()


def test_frontier_command_with_advanced_flags(tmp_path: Path):
    prices = pd.DataFrame(
        {
            "A": [100, 101, 99, 102, 103, 104],
            "B": [50, 50.5, 50.2, 51, 51.2, 51.6],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )
    prices_path = tmp_path / "prices.csv"
    frontier_out = tmp_path / "frontier.csv"
    prices.to_csv(prices_path)

    result = runner.invoke(
        app,
        [
            "frontier",
            "--prices-csv",
            str(prices_path),
            "--n-points",
            "5",
            "--allow-short",
            "--gross-limit",
            "2.0",
            "--no-parallel",
            "--output-csv",
            str(frontier_out),
        ],
    )

    assert result.exit_code == 0
    assert frontier_out.exists()


def test_surface_command_with_preset(tmp_path: Path):
    prices = pd.DataFrame(
        {
            "A": [100, 101, 99, 102, 103, 104],
            "B": [50, 50.5, 50.2, 51, 51.2, 51.6],
            "C": [80, 81, 80.5, 82, 82.2, 83],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )
    prices_path = tmp_path / "prices.csv"
    surface_out = tmp_path / "surface.csv"
    prices.to_csv(prices_path)

    result = runner.invoke(
        app,
        [
            "surface",
            "--prices-csv",
            str(prices_path),
            "--lambda-preset",
            "small",
            "--no-parallel",
            "--output-csv",
            str(surface_out),
        ],
    )

    assert result.exit_code == 0
    assert surface_out.exists()
