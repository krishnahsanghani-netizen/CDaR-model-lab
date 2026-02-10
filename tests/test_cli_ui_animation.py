from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from enhanced_cdar.cli import app

runner = CliRunner()


def test_ui_command_invokes_streamlit(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr("enhanced_cdar.cli.subprocess.call", _fake_call)

    result = runner.invoke(app, ["ui", "--port", "8502", "--server-headless", "true"])
    assert result.exit_code == 0
    assert calls
    assert "streamlit" in calls[0]
    assert "--server.port" in calls[0]


def test_animate_underwater_prefers_run_dir_over_prices_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "existing-run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)

    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 101.5],
            "B": [80.0, 80.2, 80.4, 80.1],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    (data_dir / "prices.csv").write_text(prices.to_csv(), encoding="utf-8")

    def _fake_animate(**kwargs):
        out = Path(kwargs["save_path"]).with_suffix(".gif")
        out.write_bytes(b"GIF89a")
        return str(out)

    monkeypatch.setattr("enhanced_cdar.cli.animate_underwater", _fake_animate)

    result = runner.invoke(
        app,
        [
            "animate-underwater",
            "--run-dir",
            str(run_dir),
            "--prices-csv",
            str(tmp_path / "missing.csv"),
            "--output-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    assert "Generated underwater animation" in result.stdout
    manifests = list((tmp_path / "runs").rglob("manifest.json"))
    assert manifests
