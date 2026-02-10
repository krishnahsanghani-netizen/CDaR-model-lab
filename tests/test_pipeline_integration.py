from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import enhanced_cdar.cli as cli_mod
from enhanced_cdar.data.loaders import DataLoadResult, DataMetadata


runner = CliRunner()


def test_run_pipeline_writes_expected_artifacts(tmp_path: Path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.2 for i in range(30)],
            "AGG": [80 + i * 0.05 for i in range(30)],
            "GLD": [60 + i * 0.1 for i in range(30)],
            "QQQ": [120 + i * 0.3 for i in range(30)],
        },
        index=idx,
    )

    def fake_load_from_yfinance(*args, **kwargs):
        return DataLoadResult(
            prices=prices,
            metadata=DataMetadata(
                source="test",
                downloaded_at_utc="2026-02-10T00:00:00+00:00",
                tickers=["SPY", "AGG", "GLD", "QQQ"],
                start="2021-01-01",
                end="2026-01-01",
                frequency="daily",
            ),
        )

    monkeypatch.setattr(cli_mod, "load_from_yfinance", fake_load_from_yfinance)

    result = runner.invoke(
        cli_mod.app,
        [
            "run-pipeline",
            "--output-root",
            str(tmp_path / "runs"),
            "--years",
            "2",
        ],
    )

    assert result.exit_code == 0

    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs, "Expected one run directory"
    run_dir = run_dirs[0]

    assert (run_dir / "data" / "prices.csv").exists()
    assert (run_dir / "data" / "metadata.json").exists()
    assert (run_dir / "results" / "optimization.json").exists()
    assert (run_dir / "results" / "summary.json").exists()
    assert (run_dir / "results" / "frontier.csv").exists()
    assert (run_dir / "results" / "surface.csv").exists()
    assert (run_dir / "plots" / "underwater.html").exists()
    assert (run_dir / "plots" / "frontier.html").exists()
    assert (run_dir / "plots" / "surface_3d.html").exists()
    assert (run_dir / "plots" / "surface_2d.html").exists()
