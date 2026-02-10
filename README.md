# enhanced-cdar-model

Enhanced Conditional Drawdown at Risk (CDaR) modeling suite for quantitative portfolio research.

## What It Is
This project provides a Python package and CLI for:
- Historical data ingestion (CSV and yfinance).
- Portfolio return, drawdown, CDaR, VaR/CVaR, and performance metrics.
- CDaR-aware optimization and frontier/surface generation.
- Underwater, frontier, and mean-variance-CDaR visualization.

The design targets quant developers and portfolio analysts first, with defaults that remain usable for motivated beginners.

## Install
```bash
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## Quick Start (Python API)
```python
import numpy as np
from enhanced_cdar.data.loaders import load_from_yfinance
from enhanced_cdar.data.preprocess import align_and_clean_prices
from enhanced_cdar.metrics.drawdown import compute_returns
from enhanced_cdar.portfolio.optimization import optimize_portfolio_cdar

result = load_from_yfinance(["SPY", "AGG", "GLD", "QQQ"], "2021-01-01", "2026-01-01")
prices = align_and_clean_prices(result.prices)
returns = compute_returns(prices)
opt = optimize_portfolio_cdar(returns, alpha=0.95, no_short=True)
print(opt["weights"], opt["cdar"], opt["status"])
```

## API Overview
- Data
  - `enhanced_cdar.data.load_from_csv`
  - `enhanced_cdar.data.load_from_yfinance`
  - `enhanced_cdar.data.align_and_clean_prices`
- Metrics
  - `enhanced_cdar.metrics.compute_returns`
  - `enhanced_cdar.metrics.compute_drawdown_curve`
  - `enhanced_cdar.metrics.compute_cdar`
  - `enhanced_cdar.metrics.summarize_core_metrics`
- Portfolio / Optimization
  - `enhanced_cdar.portfolio.run_backtest`
  - `enhanced_cdar.portfolio.optimize_portfolio_cdar`
  - `enhanced_cdar.portfolio.compute_cdar_efficient_frontier`
  - `enhanced_cdar.portfolio.compute_mean_var_cdar_surface`
  - `enhanced_cdar.portfolio.evaluate_portfolio_scenarios`
  - `enhanced_cdar.portfolio.compute_regime_metrics`
- Visualization
  - `enhanced_cdar.viz.plot_underwater`
  - `enhanced_cdar.viz.plot_cdar_efficient_frontier`
  - `enhanced_cdar.viz.plot_mean_variance_cdar_surface`

## Mathematical Notes
- Drawdown is computed as `value / running_peak - 1`, internally non-positive.
- CDaR reports positive drawdown magnitude in tail.
- Optimization uses a convex proxy with auxiliary variables (`eta`, `xi`) and supports:
  - return target constraint,
  - explicit CDaR target constraint,
  - bounds and gross exposure constraints.

## CLI Overview
Entry point: `enhanced-cdar`

Commands:
- `fetch-data`
- `analyze-portfolio`
- `backtest`
- `optimize-cdar`
- `frontier`
- `surface`
- `scenario`
- `regime`
- `run-pipeline`

Examples:
```bash
enhanced-cdar fetch-data --tickers SPY,AGG,GLD,QQQ --start 2021-01-01 --end 2026-01-01 --output data/prices.csv
enhanced-cdar analyze-portfolio --prices-csv data/prices.csv --weights 0.25,0.25,0.25,0.25 --format json
enhanced-cdar backtest --prices-csv data/prices.csv --weights 0.25,0.25,0.25,0.25 --rebalance-calendar M
enhanced-cdar optimize-cdar --prices-csv data/prices.csv --alpha 0.95 --no-short
enhanced-cdar frontier --prices-csv data/prices.csv --n-points 20 --allow-short --gross-limit 2.0
enhanced-cdar surface --prices-csv data/prices.csv --lambda-preset medium --no-parallel
enhanced-cdar scenario --prices-csv data/prices.csv --weights 0.25,0.25,0.25,0.25 --scenario-preset basic
enhanced-cdar regime --prices-csv data/prices.csv --weights 0.25,0.25,0.25,0.25 --regime-frequency Q
enhanced-cdar run-pipeline
```

Python script example:
```bash
python examples/example_basic_pipeline.py
```

## Configuration
- YAML supported via `--config`.
- Precedence: CLI flags > YAML config > package defaults.
- Frequency-aware annualization: daily=252, weekly=52, monthly=12.
- Example YAML config: `examples/config.example.yaml`.

## Notes on Conventions
- Drawdown is stored internally as negative values (`0` at peaks, negative underwater).
- User-facing summaries report positive risk magnitudes for CDaR, max drawdown, VaR, and CVaR.
- Data cache defaults to `./cache` and is reused until `--refresh` is passed.

## Development Quality Gates
- CI lint gate: `ruff`
- Type checks: `mypy`
- Tests: `pytest --cov`
- Build check: `python -m build`
- CI matrix: Python 3.10 and 3.11

## License
MIT. See `LICENSE`.
