from enhanced_cdar.config import AppConfig, merge_config


def test_merge_config_nested_override():
    cfg = AppConfig()
    merged = merge_config(
        cfg,
        {
            "data": {"frequency": "weekly"},
            "metrics": {"risk_free_rate_annual": 0.04},
        },
    )
    assert merged.data.frequency == "weekly"
    assert merged.metrics.risk_free_rate_annual == 0.04
