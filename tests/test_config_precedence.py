from pathlib import Path

from enhanced_cdar.config import build_config


def test_build_config_yaml_then_overrides_precedence(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
data:
  frequency: monthly
metrics:
  default_alpha: 0.9
  risk_free_rate_annual: 0.01
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = build_config(
        config_path=yaml_path,
        overrides={
            "data": {"frequency": "weekly"},
            "metrics": {"default_alpha": 0.95},
        },
    )

    assert cfg.data.frequency == "weekly"
    assert cfg.metrics.default_alpha == 0.95
    assert cfg.metrics.risk_free_rate_annual == 0.01
