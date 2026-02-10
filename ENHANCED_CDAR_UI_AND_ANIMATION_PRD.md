# PROJECT PRD – ENHANCED CDaR VISUAL UI & ANIMATION MODULE

File name suggestion: ENHANCED_CDAR_UI_AND_ANIMATION_PRD.txt

## 1. Overview
Goal: Build a visual front-end and animation system on top of the existing enhanced-cdar-model package.

Two deliverables:

- A Streamlit-based UI ("CDaR Lab") for interactive exploration:
  - Upload or fetch data.
  - Run CDaR optimization.
  - Interactively explore underwater charts, mean–CDaR frontiers, and mean–variance–CDaR surfaces.

- A video-generation module that turns CDaR analytics into animated MP4/GIFs:
  - Underwater (drawdown) animations over time.
  - Time-evolving efficient frontiers.
  - Time-evolving mean–variance–CDaR surfaces.

This UI does not re-implement the math. It calls the existing Python API (analytics + optimization + viz computation) and focuses on interaction, layout, and animation.

## 2. Tech Stack and Structure

### 2.1 UI stack
- Streamlit for the web UI (`streamlit` Python package).
- Plotly for interactive charts inside Streamlit (underwater, frontier, surface).
- Matplotlib + `matplotlib.animation.FuncAnimation` for MP4/GIF export where needed.

### 2.2 Directory structure (in existing repo)
Extend the current repo like:

```text
enhanced-cdar-model/
  src/
    enhanced_cdar/
      ...
      viz/
        __init__.py
        underwater.py
        frontier.py
        surfaces.py
        animation.py        # NEW
  ui/
    streamlit_app.py        # NEW
    __init__.py (optional)
```

`ui/streamlit_app.py`: main Streamlit entrypoint.

`viz/animation.py`: animation and video export utilities.

## 3. Streamlit UI – “CDaR Lab”

### 3.1 High-level UI layout
In `ui/streamlit_app.py`, create a Streamlit app with:

- Sidebar: configuration.
- Main area: tabs for:
  - “Overview”
  - “Underwater”
  - “Frontier”
  - “3D Surface”
  - “Animations / Export”

Use Streamlit standard patterns: `st.sidebar`, `st.tabs`, `st.file_uploader`, etc.

### 3.2 Sidebar controls
Sidebar sections:

- Data Source
  - Option “Use cached example dataset” (default).
  - Option “Fetch with yfinance”:
    - Text input tickers (comma-separated).
    - Date inputs: `start_date`, `end_date`.
  - Option “Upload CSV”:
    - `st.file_uploader` for prices CSV.
    - Inputs to select date column and price columns.
  - Button: Load data.

- Portfolio configuration
  - Dropdown:
    - “Equal-weight”
    - “Optimize CDaR (min CDaR)”
    - “Optimize mean–CDaR frontier point”
  - Alpha (CDaR level) slider: 0.80–0.99 (default 0.95).
  - Long/short toggle:
    - Long-only vs Long-short.
  - Risk-free rate input (annual, default 0.02).
  - Benchmark ticker input (default SPY).

- Rebalancing & backtest
  - Rebalancing frequency:
    - None (static weights)
    - Monthly
    - Quarterly
    - Every N periods (numeric input for N).
  - Rolling CDaR window length input (default 63).

- Visualization options
  - Checkboxes:
    - Show benchmark.
    - Show rolling CDaR.
    - Show mean–variance–CDaR surface.
  - Theme selection:
    - Light / Dark (maps to Plotly templates).

- Run / Export
  - Button: Run analysis.
  - Button: Generate animations.
  - Button: Download latest report (placeholder for future).

### 3.3 Data flow
When user presses Load data:

- Call into `enhanced_cdar.data.loaders` with appropriate config.
- Store `prices_df` in Streamlit session state (`st.session_state`).

When user presses Run analysis:

- Validate presence of `prices_df`.
- Compute returns, portfolio weights, and optimization results using core package.
- Store:
  - `equity_curve`
  - `drawdown_series`
  - `rolling_cdar`
  - `frontier_df`
  - `surface_df`

These are used to populate plots and metrics.

## 4. UI Tabs and Visuals

### 4.1 Overview tab
Show:

- Key metrics in `st.metric` cards:
  - Expected return (annualized).
  - Volatility (annualized).
  - CDaR (α).
  - Max drawdown.
  - Calmar ratio.
  - Sharpe ratio.

- A small table of allocation weights.

- Short explanatory text (plain English) describing what CDaR is.

### 4.2 Underwater tab
Use Plotly for interactive underwater chart:

- Upper chart: cumulative portfolio value (and benchmark if selected).
- Lower chart: drawdown series (0 at peak, negative when underwater).

Optional:

- Overlay rolling CDaR as a line on the lower chart.
- Hover tooltips with time, drawdown %, and rolling CDaR.

Function in `viz/underwater.py` used by Streamlit:

```python
def make_underwater_figure(
    values: pd.Series,
    drawdown: pd.Series,
    benchmark_values: pd.Series | None = None,
    theme: str = "plotly_white",
) -> plotly.graph_objects.Figure:
    ...
```

Streamlit then calls this and renders via `st.plotly_chart(fig, use_container_width=True)`.

### 4.3 Frontier tab
Show mean–CDaR efficient frontier:

- X-axis: CDaR (risk).
- Y-axis: expected return.

Highlight the current chosen portfolio.

Optionally display scatter of other frontier points with color by volatility or max drawdown.

Wrapper:

```python
def make_cdar_frontier_figure(
    frontier_df: pd.DataFrame,
    current_point: dict | None = None,
    theme: str = "plotly_white",
) -> go.Figure:
    ...
```

Expose in Streamlit with hover, zoom, etc.

### 4.4 3D Surface tab
Mean–variance–CDaR surface:

- 3D scatter/mesh:
  - X: volatility
  - Y: CDaR
  - Z: expected return
  - Color: maybe lambda weights or max drawdown.

Optionally, 2D projections as separate subplots.

Wrapper:

```python
def make_mean_variance_cdar_surface_figure(
    surface_df: pd.DataFrame,
    theme: str = "plotly_white",
) -> go.Figure:
    ...
```

## 5. Animation & Video Generation
All animation logic resides in `viz/animation.py`. The UI will just call into these functions.

### 5.1 Underwater animation
Goal: create an MP4/GIF where the equity curve and underwater chart grow over time, showing evolving drawdowns.

Function:

```python
def animate_underwater(
    values: pd.Series,
    drawdown: pd.Series,
    benchmark_values: pd.Series | None = None,
    fps: int = 24,
    dpi: int = 120,
    save_path: str = "underwater.mp4",
) -> str:
    """
    Create an underwater animation and save as MP4.
    Returns the path to the saved file.
    """
```

Requirements:

- Use `matplotlib.animation.FuncAnimation`.
- Top subplot: equity curve up to frame `t`.
- Bottom subplot: drawdown up to frame `t`.
- Optionally shade drawdowns deeper than the CDaR threshold in a different color.
- Support ~200–500 frames (e.g., subsample trading days if needed).

### 5.2 Frontier evolution animation
Goal: show how efficient frontier moves over time (e.g., monthly re-estimated).

Input: list of `(date_label, frontier_df)` snapshots.

```python
def animate_frontier_over_time(
    frontier_snapshots: list[tuple[str, pd.DataFrame]],
    fps: int = 12,
    dpi: int = 120,
    save_path: str = "frontier_over_time.mp4",
) -> str:
    ...
```

Behavior:

- Each frame: frontier at a given date.
- Animate timeline with:
  - X: CDaR
  - Y: expected return
- Optional annotation of date in the plot title.

### 5.3 Mean–variance–CDaR surface evolution
Goal: “KERR/SCHW vibe” – evolving risk surface.

Input: list of `(date_label, surface_df)`.

```python
def animate_surface_over_time(
    surface_snapshots: list[tuple[str, pd.DataFrame]],
    fps: int = 8,
    dpi: int = 120,
    save_path: str = "surface_over_time.mp4",
) -> str:
    ...
```

Approach:

- Use Matplotlib 3D or Plotly animation; choose Matplotlib for easier MP4 writing, or explain using `plotly.io.write_html` for interactive time slider.

For Matplotlib 3D:

- Draw scatter/mesh of `(vol, CDaR, return)` for each snapshot.
- Interpolate camera position lightly to avoid jitter.

## 6. Streamlit UI – Animation Tab
In the “Animations / Export” tab:

Controls:

- Checkboxes:
  - “Generate underwater animation”
  - “Generate frontier animation”
  - “Generate surface animation”
- Inputs for FPS and duration (approx) per animation.
- Button: Generate selected animations.

When clicked:

- Call the corresponding `viz.animation` functions using the data already computed and stored.
- Show progress with `st.progress`.

When done, show:

- Download buttons:
  - `st.download_button(label="Download Underwater MP4", data=..., file_name=...)`.

Implementation details:

- Save exports under `runs/<timestamp>/videos/`.
- Expose run directory path in UI.

## 7. CLI Integration
Add CLI commands in `enhanced_cdar.cli`:

- `enhanced-cdar ui`
  - Run Streamlit app.
  - Equivalent to `streamlit run ui/streamlit_app.py`.
  - Allow custom port and debug options via flags.

- `enhanced-cdar animate-underwater`
  - Args:
    - `--prices-csv` or `--run-dir`.
    - `--fps`, `--dpi`.
  - Outputs `underwater.mp4` in chosen path.

- `enhanced-cdar animate-frontier`
  - Args for frontier snapshots generation and output file.

- `enhanced-cdar animate-surface`
  - Similar to frontier but for surfaces.

## 8. UX and Design Notes
Keep UI simple and clean, not like a trading terminal:

- Neutral colors, readable fonts, no clutter.

Provide hover tooltips explaining:

- CDaR.
- Drawdown.
- Efficient frontier vs mean–variance–CDaR surface.

For “risk porn” feel:

- Smooth lines, consistent color palette (e.g., portfolio = blue, benchmark = gray, underwater = red).
- Optional toggle for dark mode (connect to Plotly theme).

## 9. Testing and Performance
Ensure animation functions can handle:

- ~5–10 years of daily data (with subsampling).

Provide unit tests for:

- Animation helper functions (frame counts, no exceptions).
- UI-level util functions (data loading, config parsing).

Do not unit-test Streamlit output; just test underlying logic.

## 10. Out of Scope for This PRD
- No multi-user auth, databases, or complex backend.
- No live market streaming data; all data is historical / batch.
- No generic trading engine; focus is visualization and animation of CDaR analytics.

END OF UI & ANIMATION PRD FILE
