# PROJECT PRD – ENHANCED CONDITIONAL DRAWDOWN AT RISK (CDaR) MODELING SUITE

File name suggestion: ENHANCED_CDAR_MODEL_PRD.txt

## 1. Project Overview
Build a production-quality, open-source quantitative finance toolkit focused on Conditional Drawdown at Risk (CDaR) and “enhanced” drawdown-based portfolio models (in the spirit of KERR/SCHW-style enhanced CDaR).

The toolkit must:

- Ingest historical price data for multiple assets.
- Compute portfolio returns, drawdowns, CDaR, and related risk metrics.
- Optimize portfolios using CDaR-based risk objectives and constraints.
- Visualize:
  - Underwater (drawdown) curves.
  - Mean–CDaR efficient frontiers.
  - Enhanced multi-dimensional risk surfaces (e.g., Mean–Variance–CDaR).
- Be written in Python with clean, modular architecture.
- Be usable as:
  - A Python package (importable in other projects).
  - A CLI tool for non-programmers.
- Be ready for GitHub open-source release with:
  - LICENSE (permissive).
  - README.md (detailed).
  - .gitignore.
  - Examples and tests.

Aim for clarity such that a high-finance quant engineer would consider it a serious starting point.

Assume the end user may have minimal math/programming background; provide documentation and defaults that “just work.”

## 2. Tech Stack and Repository Layout

### 2.1 Language and Core Libraries
Use:

- Python 3.11+.

Core libraries:

- numpy
- pandas
- scipy (optimization, distributions)
- cvxpy (convex optimization for CDaR problems)
- matplotlib or plotly (visualization; prefer plotly for interactivity)
- pydantic (for typed configs, optional but preferred)
- typer or click (for CLI)
- pytest (for tests)
- yfinance (for data download convenience)

No hard dependency on heavyweight frameworks; keep it pure Python + scientific stack.

### 2.2 Repository Structure
Create a repo structure like:

```text
enhanced-cdar-model/
  README.md
  LICENSE
  .gitignore
  pyproject.toml  or  setup.cfg + setup.py
  requirements.txt
  src/
    enhanced_cdar/
      __init__.py
      config.py
      data/
        __init__.py
        loaders.py
        pocess.py
      metrics/
        __init__.py
        drawdown.py
        cdar.py
        risk_metrics.py
      portfolio/
        __init__.py
        weights.py
        optimization.py
        backtest.py
      viz/
        __init__.py
        underwater.py
        frontier.py
        surfaces.py
      cli.py
  examples/
    example_basic_pipeline.ipynb
    example_cli_usage.md
  tests/
    test_drawdown.py
    test_cdar.py
    test_optimization.py
    test_backtest.py
    test_cli.py
```

## 3. Mathematical Definitions and Strategy
Implement all math explicitly and clearly documented in-code with docstrings. Use standard definitions where possible.

### 3.1 Data Model
Assume N assets and T time steps.

Let:

- P(t, i): price of asset i at time t.
- R(t, i): return of asset i at time t, computed as simple or log return (configurable).

### 3.1.1 Returns
Default: simple returns.

Simple return:

R(t, i) = (P(t, i) − P(t−1, i)) / P(t−1, i)

Log return (if configured):

R(t, i) = ln(P(t, i) / P(t−1, i))

Implement:

- A configurable function `compute_returns(prices, method="simple"|"log")`.

### 3.2 Portfolio Weights and Returns
Let:

- w ∈ R^N: portfolio weight vector (weights sum to 1, optionally no shorting).

Portfolio return at time t:

R_p(t) = w^T R_t

where R_t is the vector of asset returns at time t.

Implement:

- `compute_portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series`.

### 3.3 Cumulative Value and Drawdown
Let:

- Initial portfolio value V_0 = 1 (configurable).

Portfolio value at time t:

V_t = V_{t−1} · (1 + R_p(t))

Define:

- Running peak:

Peak_t = max_{0≤s≤t} V_s

- Drawdown at time t:

D_t = (V_t − Peak_t) / Peak_t

This is negative or zero; often reported as a positive magnitude: −D_t.

Implement:

- `compute_cumulative_value(portfolio_returns) -> pd.Series`.
- `compute_drawdown_curve(values) -> pd.Series` of drawdown values.
- `max_drawdown(drawdown_series)`.

### 3.4 Conditional Drawdown at Risk (CDaR)
Define CDaR analogous to Conditional Value at Risk (CVaR) but on drawdowns rather than returns.

Let:

- Drawdown series D_t (negative or zero).
- Consider magnitudes X_t = −D_t ≥ 0 as positive drawdown sizes.
- Choose confidence level α ∈ (0,1), e.g., α = 0.95.

Steps:

1. Sort drawdown magnitudes X_t descending (largest drawdowns first).
2. Identify the quantile threshold q_α so that P(X_t ≥ q_α) = 1 − α.
3. CDaR is the conditional expectation of X_t given X_t ≥ q_α:

CDaR_α = E[X_t | X_t ≥ q_α]

Implement:

- `compute_cdar(drawdown_series, alpha=0.95)` returning a positive number (magnitude).
- Optionally, implement an LP-based definition via Roc; see below).

### 3.5 Other Risk Metrics (for comparison)
Implement at least:

- Volatility of portfolio returns (annualized, configurable frequency).
- Value at Risk (VaR) at level α (historical method).
- CVaR (Expected Shortfall) at level α (historical).

This lets users compare CDaR to more standard measures.

### 3.6 Optimization Problems (Core of “Enhanced” Model)
Use cvxpy to formulate optimization problems.

#### 3.6.1 Basic CDaR Minimization
Given:

- Historical returns matrix R ∈ R^(T×N).

Weights w ∈ R^N, with constraints:

- 1^T w = 1.
- w_i ≥ 0 (no short selling) or set by config.
- Confidence level α.

Goal: minimize CDaR of the portfolio’s drawdowns.

LP-style formulation (Rockafellar & Uryasev–type adaptation to drawdowns):

- Compute portfolio cumulative values V_t(w) and drawdowns D_t(w) as linear functions of returns and weights.
- For linearity, approximate directly in terms of returns and weight effects. Scenario-based linearization (detailed in code comments).

Introduce auxiliary variables:

- η: drawdown threshold (scalar).
- ξ_t ≥ 0: excess drawdown above η for each time t.

Objective:

min_{w,η,ξ} η + 1/((1−α)T) * Σ ξ_t

Subject to:

- Drawdown constraints (for each t):

X_t(w) − η ≤ ξ_t,
ξ_t ≥ 0

where X_t(w) are drawdown magnitudes (approximate or compute as needed).

- Weight constraints: sum to 1, bounds, etc.

Implement function:

```python
optimize_portfolio_cdar(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    no_short: bool = True,
    weight_bounds: tuple[float, float] | None = (0.0, 1.0),
    target_return: float | None = None,
    target_cdar: float | None = None,
    solver: str | None = None
) -> dict
```

Return a dict with:

- weights: np.ndarray
- cdar: float
- expected_return: float (historical mean)
- other_metrics: dict (max drawdown, volatility, VaR, CVaR, etc.)
- status: solver status string.

#### 3.6.2 Mean–CDaR Efficient Frontier
For a grid of target expected returns μ_target, solve minimization of CDaR subject to:

E[R_p] ≥ μ_target

Collect results (weights, CDaR, expected return).

Function:

```python
compute_cdar_efficient_frontier(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    no_short: bool = True,
    n_points: int = 20,
    return_range: tuple[float, float] | None = None
) -> pd.DataFrame
```

Output DataFrame columns:

- target_return
- achieved_return
- cdar
- volatility
- optional: max_drawdown, etc.

#### 3.6.3 Enhanced Multi-Objective (Mean–Variance–CDaR Surface)
Construct a family of portfolios that trade off:

- Expected return.
- Variance (or volatility).
- CDaR.

Use scalarization:

min_w λ1·CDaR(w) + λ2·σ(w) − λ3·μ(w)

with λ_i ≥ 0 weights chosen from a grid.

Implement function:

```python
compute_mean_var_cdar_surface(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    lambda_grid: list[tuple[float, float, float]],
    no_short: bool = True
) -> pd.DataFrame
```

DataFrame columns:

- lambda_cdar, lambda_var, lambda_return
- expected_return, volatility, cdar, max_drawdown
- Possibly the weights as a nested structure or external mapping.

## 4. Visualization Requirements
All visualizations should have:

- Clear axes labels.
- Titles that mention CDaR and key parameters (e.g., confidence level).
- Legends and annotations where useful.
- Ability to save to file (.png, .svg, .html for interactive).

### 4.1 Underwater (Drawdown) Chart
In `viz/underwater.py` implement:

```python
plot_underwater(
    values: pd.Series,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None
)
```

- Top subplot: portfolio value over time.
- Bottom subplot: drawdown (as negative values).
- Optionally shade regions where drawdown exceeds certain thresholds (e.g., beyond CDaR quantile).

### 4.2 Mean–CDaR Efficient Frontier Plot
In `viz/frontier.py`:

```python
plot_cdar_efficient_frontier(
    frontier_df: pd.DataFrame,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None
)
```

- X-axis: CDaR.
- Y-axis: expected return.
- Draw a curve connecting points.
- Highlight:
  - Minimum CDaR portfolio.
  - Maximum return portfolio on the frontier.

### 4.3 Mean–Variance–CDaR Surface
In `viz/surfaces.py`:

```python
plot_mean_variance_cdar_surface(
    surface_df: pd.DataFrame,
    mode: str = "3d",  # "3d" or "2d-projections"
    show: bool = True,
    save_path: str | None = None
)
```

3D mode:

- Axes: (volatility, CDaR, expected return).
- Color: maybe another metric (max drawdown or lambda weights).

2D projections:

- (volatility, expected return) colored by CDaR.

Use Plotly’s 3D scatter/mesh for interactivity.

## 5. Data Loading and Preprocessing
In `data/loaders.py`:

- `load_from_csv(path, date_col, price_cols, parse_dates=True)`.
- `load_from_yfinance(tickers, start, end, interval="1d")`.

In `data/preprocess.py`:

- `align_and_clean_prices(prices_df)`:
  - Drop rows with all NaN.
  - Forward-fill or drop NA depending on config.
- `normalize_prices(prices_df)` (optional).

Configurable options in `config.py` (e.g., using `pydantic.BaseSettings`):

- return_method ("simple" or "log").
- risk_free_rate (for Sharpe-like metrics).
- annualization_factor (e.g., 252).
- default_alpha for CDaR (e.g., 0.95).

## 6. CLI Design
Use typer or click to implement a CLI entry point `enhanced_cdar.cli:app`.

Commands:

- `fetch-data`
  - Args:
    - `--tickers` (comma-separated string).
    - `--start`, `--end`.
    - `--output` (CSV path).
  - Downloads via yfinance and writes CSV.

- `analyze-portfolio`
  - Args:
    - `--prices-csv` (input file).
    - `--weights` (comma-separated floats) or `--weights-json`.
    - `--alpha` (CDaR level).
  - Computes returns, drawdowns, CDaR, and prints a summary.

- `optimize-cdar`
  - Args:
    - `--prices-csv`.
    - `--alpha`.
    - `--no-short` / `--allow-short`.
    - `--target-return` (optional).
  - Runs CDaR optimization and outputs:
    - Optimal weights.
    - Key metrics.
    - Optionally writes results to CSV.

- `frontier`
  - Args:
    - `--prices-csv`.
    - `--alpha`.
    - `--n-points`.
  - Computes frontier, saves CSV and plot.

- `surface`
  - Args:
    - `--prices-csv`.
    - `--alpha`.
    - `--lambda-grid-json` or some preset.
  - Computes surface, saves CSV and plot.

CLI should have help messages and examples.

## 7. Documentation and README
Create `README.md` with:

- Project title and badges (build, license).
- Short explanation of CDaR and why it matters (plain English).
- Install instructions:

```bash
pip install enhanced-cdar
```

(or local `pip install -e .`).

- Quick start example (copy-pasteable):
  - Download data (SPY, AAPL, etc.).
  - Optimize CDaR portfolio.
  - Plot underwater chart and frontier.
- API overview with key functions and classes.
- Links to examples/ notebooks.

Include a `docs/` section optionally or keep it in README for now.

## 8. Licensing and GitHub Details

### 8.1 LICENSE
Use MIT License. Generate a standard LICENSE file with MIT terms and placeholder for author name and year.

### 8.2 .gitignore
Use a Python-standard `.gitignore`, including at least:

- `__pycache__/`
- `*.pyc`
- `.venv/`, `venv/`
- `.ipynb_checkpoints/`
- `.DS_Store`
- `*.egg-info/`
- `dist/`, `build/`

### 8.3 GitHub Actions (Optional Advanced Feature)
Add a minimal CI pipeline:

`.github/workflows/python-tests.yml`:

- On push/pull_request:
  - Set up Python.
  - Install dependencies.
  - Run pytest.

## 9. Testing Requirements
Use pytest to cover:

- Drawdown calculations:
  - Synthetic price series with known max drawdown and CDaR.
- CDaR computation:
  - Edge cases: no drawdowns, constant series.
- Optimization:
  - Check weights sum to 1.
  - Check constraints (no short, bounds).
  - Verify that optimized portfolio CDaR is not worse than a naive equal-weight portfolio on simple test data.
- CLI:
  - Use pytest’s capsys or click.testing.CliRunner / typer test client to test command behavior.

Include fixtures in `tests/conftest.py` for synthetic data.

## 10. Code Quality and Style

- Use type hints throughout (PEP 484).
- Add docstrings for all public functions and classes.
- Use a consistent style (e.g., black, isort, ruff for linting).
- Log key steps (especially in optimization) with Python’s logging module, not print.

## 11. Stretch Goals (Optional, if Codex can handle)
If there is capacity, add:

- Regime analysis:
  - Compare CDaR across subperiods (e.g., pre- and post-crisis).
- Scenario analysis:
  - Stress test CDaR under hypothetical shocks (e.g., –10% shock to equity).
- Interactive web app:
  - Simple FastAPI or Streamlit front-end to:
    - Upload CSV.
    - Run optimization.
    - Show charts.
- Factor-based extension:
  - Decompose portfolio into factor exposures and show CDaR per factor.

But the core requirement is: a robust, well-documented Python library + CLI for enhanced CDaR modeling, optimization, and visualization, structured and commented so high-finance quants can understand, audit, and extend the code.
