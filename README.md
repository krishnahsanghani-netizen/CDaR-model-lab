# enhanced-cdar-model

A Python toolkit and interactive lab for Conditional Drawdown at Risk (CDaR) portfolio analysis.

## What this project provides

- Historical data ingestion (CSV and yfinance)
- Portfolio metrics (CDaR, max drawdown, VaR/CVaR, Sharpe, Sortino, Calmar)
- CDaR-focused optimization and efficient frontier tooling
- Interactive Streamlit UI (CDaR Lab)
- Animation exports (underwater, frontier, surface, reactive model)

## Quick start

See [`QUICKSTART.md`](QUICKSTART.md).

Fastest path:

```bash
npm run dev
```

Then open `http://localhost:8501`.

## Streamlit UI workflow

1. Choose a data source in the sidebar.
2. Click `Load data`.
3. Click `Run analysis`.
4. Explore tabs:
   - Overview
   - Underwater
   - Frontier
   - 3D Surface
   - Reactive 3D Model
   - Animations / Export

For long date ranges, keep `Fast mode` enabled while iterating.

## CLI

Entry point:

```bash
enhanced-cdar --help
```

Core commands:

- `fetch-data`
- `analyze-portfolio`
- `backtest`
- `optimize-cdar`
- `frontier`
- `surface`
- `run-pipeline`
- `ui`
- `animate-underwater`
- `animate-frontier`
- `animate-surface`
- `animate-model`
- `scenario`
- `regime`

## Python API example

```python
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

## Notes

- Internal drawdown sign convention is non-positive (`0` at peaks, negative underwater).
- User-facing risk summaries display positive magnitudes.
- Config precedence: CLI flags > YAML config > package defaults.
- Animation export prefers MP4; GIF is used as fallback when ffmpeg is unavailable.

## License

MIT. See `LICENSE`.
