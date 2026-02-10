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
