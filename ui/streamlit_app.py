"""CDaR Lab Streamlit UI."""

from __future__ import annotations

import json
import base64
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from enhanced_cdar.config import AppConfig
from enhanced_cdar.data.loaders import load_from_yfinance
from enhanced_cdar.data.preprocess import align_and_clean_prices
from enhanced_cdar.metrics.cdar import compute_rolling_cdar
from enhanced_cdar.metrics.drawdown import (
    compute_returns,
)
from enhanced_cdar.portfolio.backtest import run_backtest
from enhanced_cdar.portfolio.optimization import (
    compute_cdar_efficient_frontier,
    compute_mean_var_cdar_surface,
    optimize_portfolio_cdar,
)
from enhanced_cdar.viz.animation import (
    animate_frontier_over_time,
    animate_reactive_cdar_model,
    animate_surface_over_time,
    animate_underwater,
    generate_frontier_snapshots,
    generate_surface_snapshots,
)
from enhanced_cdar.viz.frontier import make_cdar_frontier_figure
from enhanced_cdar.viz.surfaces import (
    make_cdar_reactive_model_figure,
    make_mean_variance_cdar_surface_figure,
)
from enhanced_cdar.viz.underwater import make_underwater_figure
try:
    from ui.state import UIState
    from ui.utils import (
        autodetect_csv_columns,
        load_example_prices,
        parse_uploaded_prices,
    )
except ModuleNotFoundError:
    # Supports direct execution via: streamlit run ui/streamlit_app.py
    from state import UIState  # type: ignore
    from utils import (  # type: ignore
        autodetect_csv_columns,
        load_example_prices,
        parse_uploaded_prices,
    )

STATE_KEY = "cdar_ui_state"


def get_state() -> UIState:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = UIState()
    return st.session_state[STATE_KEY]


def _theme_to_template(theme: str) -> str:
    return "plotly_dark" if theme == "Dark" else "plotly_white"


def _gif_to_mp4_for_preview(path: Path) -> Path:
    """Convert GIF to MP4 for playback when ffmpeg is available."""
    if path.suffix.lower() != ".gif":
        return path
    if shutil.which("ffmpeg") is None:
        return path

    mp4_path = path.with_name(f"{path.stem}.preview.mp4")
    if mp4_path.exists() and mp4_path.stat().st_mtime >= path.stat().st_mtime:
        return mp4_path

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-movflags",
                "faststart",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return mp4_path
    except Exception:
        return path


def _render_looping_video(path: Path, height: int = 520) -> None:
    """Render autoplaying, looping HTML5 video with pause controls."""
    suffix = path.suffix.lower()
    mime = "video/mp4" if suffix == ".mp4" else "video/webm"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    html = f"""
    <video controls autoplay loop muted playsinline style="width:100%; border-radius:10px;">
      <source src="data:{mime};base64,{payload}" type="{mime}">
      Your browser does not support embedded video playback.
    </video>
    """
    components.html(html, height=height, scrolling=False)


def _safe_fetch_prices(
    tickers: str,
    start: str,
    end: str,
    frequency: str,
    cfg: AppConfig,
) -> pd.DataFrame | None:
    try:
        result = load_from_yfinance(
            tickers=[t.strip() for t in tickers.split(",") if t.strip()],
            start=start,
            end=end,
            frequency=frequency,
            cache_dir=cfg.data.cache_dir,
            use_cache=cfg.data.use_cache,
            refresh=cfg.data.refresh_cache,
        )
        return result.prices
    except Exception as exc:
        st.error(
            "Could not fetch data from yfinance. "
            "If network access is unavailable, upload a CSV instead. "
            f"Details: {exc}"
        )
        return None


def _choose_weights(
    mode: str,
    returns_df: pd.DataFrame,
    alpha: float,
    no_short: bool,
    gross_limit: float,
    frontier_target_return: float,
    frontier_percentile: int,
    fast_mode: bool,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    n_assets = returns_df.shape[1]
    if mode == "Equal-weight":
        return np.full(n_assets, 1.0 / n_assets), None

    if mode == "Optimize CDaR (min CDaR)":
        result = optimize_portfolio_cdar(
            returns=returns_df,
            alpha=alpha,
            no_short=no_short,
            gross_exposure_limit=None if no_short else gross_limit,
            weight_bounds=None,
        )
        return np.asarray(result["weights"], dtype=float), None

    frontier_df = compute_cdar_efficient_frontier(
        returns=returns_df,
        alpha=alpha,
        no_short=no_short,
        n_points=12 if fast_mode else 30,
        gross_exposure_limit=None if no_short else gross_limit,
        parallel=not fast_mode,
    )
    feasible = frontier_df.dropna(subset=["achieved_return", "cdar"])
    if feasible.empty:
        raise ValueError("No feasible frontier points were produced.")

    target_idx = min(
        max(int(len(feasible) * frontier_percentile / 100.0), 0),
        len(feasible) - 1,
    )
    percentile_row = feasible.iloc[target_idx]
    distance = (feasible["achieved_return"] - frontier_target_return).abs()
    nearest_row = feasible.loc[distance.idxmin()] if len(feasible) else percentile_row

    chosen = nearest_row if np.isfinite(frontier_target_return) else percentile_row
    return np.asarray(chosen["weights"], dtype=float), frontier_df


def _load_or_fetch_benchmark(
    benchmark_ticker: str,
    prices_df: pd.DataFrame,
    start: str,
    end: str,
    cfg: AppConfig,
) -> pd.Series | None:
    if benchmark_ticker in prices_df.columns:
        return prices_df[benchmark_ticker]

    st.warning(
        f"Benchmark '{benchmark_ticker}' is not in uploaded data. "
        "Attempting to fetch benchmark prices from yfinance."
    )
    fetched = _safe_fetch_prices(benchmark_ticker, start, end, cfg.data.frequency, cfg)
    if fetched is None or fetched.empty:
        st.warning("Benchmark fetch failed. Benchmark overlays and metrics are disabled.")
        return None
    return fetched.iloc[:, 0]


def _run_analysis(
    state: UIState,
    cfg: AppConfig,
    alpha: float,
    no_short: bool,
    gross_limit: float,
    risk_free_rate: float,
    mode: str,
    benchmark_ticker: str,
    show_rolling_cdar: bool,
    rolling_window: int,
    rebalance_choice: str,
    rebalance_n: int,
    show_surface: bool,
    compute_frontier: bool,
    fast_mode: bool,
    frontier_target_return: float,
    frontier_percentile: int,
) -> None:
    if state.prices_df is None:
        raise ValueError("No prices loaded. Use 'Load data' first.")

    prices_df = align_and_clean_prices(state.prices_df, cfg.data.missing_data_policy)
    returns_df = compute_returns(prices_df, method=cfg.metrics.return_method)
    weights, frontier_df = _choose_weights(
        mode,
        returns_df,
        alpha,
        no_short,
        gross_limit,
        frontier_target_return,
        frontier_percentile,
        fast_mode,
    )

    start = prices_df.index.min().date().isoformat()
    end = prices_df.index.max().date().isoformat()
    benchmark_values = _load_or_fetch_benchmark(benchmark_ticker, prices_df, start, end, cfg)

    rebalance_calendar = "none"
    rebalance_every_n = None
    if rebalance_choice == "Monthly":
        rebalance_calendar = "M"
    elif rebalance_choice == "Quarterly":
        rebalance_calendar = "Q"
    elif rebalance_choice == "Every N periods":
        rebalance_every_n = rebalance_n

    benchmark_returns = None
    if benchmark_values is not None:
        benchmark_returns = compute_returns(
            benchmark_values.to_frame(benchmark_ticker),
            method=cfg.metrics.return_method,
        ).iloc[:, 0]

    backtest = run_backtest(
        returns=returns_df,
        weights=weights,
        rebalance_calendar=rebalance_calendar,
        rebalance_every_n_periods=rebalance_every_n,
        rebalance_mode="fixed",
        benchmark_returns=benchmark_returns,
        alpha=alpha,
        annualization_factor=cfg.annualization_factor,
        risk_free_rate_annual=risk_free_rate,
        no_short=no_short,
        gross_exposure_limit=None if no_short else gross_limit,
    )

    rolling = None
    if show_rolling_cdar:
        rolling = compute_rolling_cdar(backtest.drawdown, alpha=alpha, window=rolling_window)

    if frontier_df is None and compute_frontier:
        frontier_df = compute_cdar_efficient_frontier(
            returns=returns_df,
            alpha=alpha,
            no_short=no_short,
            n_points=12 if fast_mode else 30,
            gross_exposure_limit=None if no_short else gross_limit,
            parallel=not fast_mode,
        )

    surface_df = None
    if show_surface:
        lambda_grid = [
            (1.0, 0.1, 0.1),
            (1.0, 0.3, 0.3),
            (1.0, 0.5, 0.5),
            (1.0, 0.8, 0.8),
            (1.0, 1.0, 1.0),
            (0.5, 1.0, 1.0),
            (0.3, 1.0, 1.2),
        ]
        surface_df = compute_mean_var_cdar_surface(
            returns_df,
            alpha=alpha,
            lambda_grid=lambda_grid,
            no_short=no_short,
            gross_exposure_limit=None if no_short else gross_limit,
            parallel=not fast_mode,
        )

    state.returns_df = returns_df
    state.weights = weights
    state.equity_curve = backtest.portfolio_values
    state.drawdown_series = backtest.drawdown
    state.rolling_cdar = rolling
    state.frontier_df = frontier_df
    state.surface_df = surface_df

    metrics = dict(backtest.metrics)
    if backtest.benchmark_metrics:
        metrics.update(backtest.benchmark_metrics)
    state.analysis_results = {
        "metrics": metrics,
        "weights": dict(zip(returns_df.columns.tolist(), weights.tolist())),
        "mode": mode,
        "alpha": alpha,
        "no_short": no_short,
        "data_start": str(returns_df.index.min().date()),
        "data_end": str(returns_df.index.max().date()),
    }
    if benchmark_values is not None:
        state.benchmark_prices = benchmark_values
        state.benchmark_returns = benchmark_returns


def _render_sidebar(state: UIState) -> dict[str, Any]:
    st.sidebar.title("CDaR Lab")

    cfg = AppConfig()
    fast_mode = st.sidebar.toggle(
        "Fast mode",
        value=True,
        help=(
            "Prioritizes speed by reducing expensive optimization and animation settings. "
            "Recommended for long date ranges."
        ),
    )

    st.sidebar.subheader("Data Source")
    source = st.sidebar.radio(
        "Select source",
        options=["Use cached example dataset", "Fetch with yfinance", "Upload CSV"],
        index=0,
    )

    tickers = st.sidebar.text_input("Tickers", value="SPY,AGG,GLD,QQQ")
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2026-01-01"))

    uploaded = None
    date_col_override = ""
    price_cols_override = ""
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload prices CSV", type=["csv"])
        date_col_override = st.sidebar.text_input("Date column", value="date")
        price_cols_override = st.sidebar.text_input(
            "Price columns (comma-separated)",
            value="",
        )

    if st.sidebar.button("Load data"):
        if source == "Use cached example dataset":
            state.prices_df = load_example_prices()
            st.sidebar.success("Loaded example dataset.")
        elif source == "Fetch with yfinance":
            fetched = _safe_fetch_prices(
                tickers,
                start_date.isoformat(),
                end_date.isoformat(),
                cfg.data.frequency,
                cfg,
            )
            if fetched is not None:
                state.prices_df = fetched
                st.sidebar.success("Fetched prices from yfinance.")
        else:
            if uploaded is None:
                st.sidebar.error("Please upload a CSV file first.")
            else:
                frame = pd.read_csv(uploaded)
                auto_date, auto_price_cols = autodetect_csv_columns(frame)
                selected_date = date_col_override or auto_date
                if selected_date is None:
                    st.sidebar.error("Could not detect a date column. Enter one manually.")
                else:
                    selected_prices = [
                        c.strip() for c in price_cols_override.split(",") if c.strip()
                    ]
                    if not selected_prices:
                        selected_prices = auto_price_cols
                    try:
                        state.prices_df = parse_uploaded_prices(
                            frame,
                            selected_date,
                            selected_prices,
                        )
                        st.sidebar.success("Loaded prices from uploaded CSV.")
                    except Exception as exc:
                        st.sidebar.error(f"CSV parse error: {exc}")

    st.sidebar.subheader("Portfolio Configuration")
    mode = st.sidebar.selectbox(
        "Portfolio mode",
        options=[
            "Equal-weight",
            "Optimize CDaR (min CDaR)",
            "Optimize mean–CDaR frontier point",
        ],
    )
    alpha = st.sidebar.slider("CDaR alpha", min_value=0.80, max_value=0.99, value=0.95)
    no_short = not st.sidebar.toggle("Allow short", value=False)
    risk_free = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, step=0.005)
    benchmark_ticker = st.sidebar.text_input("Benchmark ticker", value="SPY")

    st.sidebar.subheader("Frontier Selection")
    frontier_target_return = st.sidebar.slider(
        "Target return",
        min_value=-0.10,
        max_value=0.10,
        value=0.01,
        step=0.001,
    )
    frontier_percentile = st.sidebar.slider("Frontier CDaR percentile", 0, 100, 50)

    st.sidebar.subheader("Rebalancing & Backtest")
    rebalance_choice = st.sidebar.selectbox(
        "Rebalancing frequency",
        options=["None (static)", "Monthly", "Quarterly", "Every N periods"],
        index=0,
    )
    rebalance_n = st.sidebar.number_input("N periods", min_value=1, value=21)
    rolling_window = st.sidebar.number_input("Rolling CDaR window", min_value=2, value=63)

    st.sidebar.subheader("Visualization")
    show_benchmark = st.sidebar.checkbox("Show benchmark", value=True)
    show_rolling = st.sidebar.checkbox("Show rolling CDaR", value=not fast_mode)
    show_surface = st.sidebar.checkbox(
        "Show mean–variance–CDaR surface",
        value=not fast_mode,
    )
    compute_frontier = st.sidebar.checkbox("Precompute frontier", value=not fast_mode)
    theme = st.sidebar.selectbox("Theme", options=["Light", "Dark"], index=0)

    st.sidebar.subheader("Animation / Export")
    gen_underwater = st.sidebar.checkbox("Generate underwater animation", value=True)
    gen_frontier = st.sidebar.checkbox("Generate frontier animation", value=not fast_mode)
    gen_surface = st.sidebar.checkbox("Generate surface animation", value=not fast_mode)
    gen_reactive_model = st.sidebar.checkbox("Generate reactive 3D model animation", value=True)
    fps = st.sidebar.number_input(
        "Animation FPS",
        min_value=2,
        value=8 if fast_mode else 12,
        max_value=60,
    )
    max_frames = st.sidebar.number_input(
        "Max frames",
        min_value=20,
        value=120 if fast_mode else 300,
        max_value=1000,
    )
    snapshot_lookback = st.sidebar.number_input(
        "Snapshot lookback (periods)",
        min_value=50,
        value=126 if fast_mode else 252,
        max_value=2000,
    )
    snapshot_step_mode_ui = st.sidebar.selectbox(
        "Snapshot step",
        options=["monthly", "quarterly", "custom"],
        index=1 if fast_mode else 0,
    )
    snapshot_custom_step = st.sidebar.number_input(
        "Custom snapshot step (periods)",
        min_value=1,
        value=63 if fast_mode else 21,
        max_value=500,
    )
    projection_years = st.sidebar.number_input(
        "Projection horizon (years)",
        min_value=1,
        max_value=30,
        value=5,
    )

    if st.sidebar.button("Run analysis"):
        try:
            _run_analysis(
                state=state,
                cfg=cfg,
                alpha=alpha,
                no_short=no_short,
                gross_limit=2.0,
                risk_free_rate=risk_free,
                mode=mode,
                benchmark_ticker=benchmark_ticker,
                show_rolling_cdar=show_rolling,
                rolling_window=rolling_window,
                rebalance_choice=rebalance_choice,
                rebalance_n=int(rebalance_n),
                show_surface=show_surface,
                compute_frontier=compute_frontier,
                fast_mode=fast_mode,
                frontier_target_return=frontier_target_return,
                frontier_percentile=frontier_percentile,
            )
            st.sidebar.success("Analysis complete.")
        except Exception as exc:
            st.sidebar.error(f"Analysis failed: {exc}")

    if st.sidebar.button("Generate animations"):
        _generate_animations(
            state,
            fps=int(fps),
            max_frames=int(max_frames),
            theme=theme,
            snapshot_lookback=int(snapshot_lookback),
            snapshot_step_mode=str(snapshot_step_mode_ui),
            snapshot_custom_step=int(snapshot_custom_step),
            generate_underwater=gen_underwater,
            generate_frontier=gen_frontier,
            generate_surface=gen_surface,
            generate_reactive_model=gen_reactive_model,
            fast_mode=fast_mode,
        )

    return {
        "show_benchmark": show_benchmark,
        "show_rolling": show_rolling,
        "show_surface": show_surface,
        "theme": theme,
        "projection_years": int(projection_years),
    }


def _generate_animations(
    state: UIState,
    fps: int,
    max_frames: int,
    theme: str,
    snapshot_lookback: int,
    snapshot_step_mode: str,
    snapshot_custom_step: int,
    generate_underwater: bool,
    generate_frontier: bool,
    generate_surface: bool,
    generate_reactive_model: bool,
    fast_mode: bool,
) -> None:
    if state.returns_df is None or state.equity_curve is None or state.drawdown_series is None:
        st.sidebar.error("Run analysis first before generating animations.")
        return

    run_dir = Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    progress = st.sidebar.progress(0.0)
    tasks = sum(
        int(x)
        for x in [
            generate_underwater,
            generate_frontier,
            generate_surface,
            generate_reactive_model,
        ]
    )
    done = 0

    if generate_underwater:
        path = animate_underwater(
            values=state.equity_curve,
            drawdown=state.drawdown_series,
            benchmark_values=state.benchmark_prices if state.benchmark_prices is not None else None,
            fps=fps,
            save_path=str(videos_dir / "underwater.mp4"),
            max_frames=max_frames,
        )
        files.append({"type": "underwater", "path": path})
        done += 1
        progress.progress(done / max(tasks, 1))

    if generate_frontier and state.returns_df is not None:
        snapshots = generate_frontier_snapshots(
            returns=state.returns_df,
            alpha=float(state.analysis_results.get("alpha", 0.95)),
            no_short=bool(state.analysis_results.get("no_short", True)),
            lookback=snapshot_lookback,
            step_mode=snapshot_step_mode,
            step_n=snapshot_custom_step,
            max_snapshots=max_frames,
        )
        if snapshots:
            path = animate_frontier_over_time(
                frontier_snapshots=snapshots,
                fps=max(4, min(24, fps)),
                save_path=str(videos_dir / "frontier_over_time.mp4"),
                max_frames=max_frames,
            )
            files.append({"type": "frontier", "path": path})
        done += 1
        progress.progress(done / max(tasks, 1))

    if generate_surface and state.returns_df is not None:
        snapshots = generate_surface_snapshots(
            returns=state.returns_df,
            alpha=float(state.analysis_results.get("alpha", 0.95)),
            no_short=bool(state.analysis_results.get("no_short", True)),
            lookback=snapshot_lookback,
            step_mode=snapshot_step_mode,
            step_n=snapshot_custom_step,
            max_snapshots=max_frames,
        )
        if snapshots:
            path = animate_surface_over_time(
                surface_snapshots=snapshots,
                fps=max(4, min(16, fps)),
                save_path=str(videos_dir / "surface_over_time.mp4"),
                max_frames=max_frames,
            )
            files.append({"type": "surface", "path": path})
        done += 1
        progress.progress(done / max(tasks, 1))

    if generate_reactive_model:
        ret_series = None
        if state.returns_df is not None and state.weights is not None:
            ret_series = (state.returns_df * state.weights).sum(axis=1)
        path = animate_reactive_cdar_model(
            drawdown=state.drawdown_series,
            returns=ret_series,
            fps=max(6, min(12 if fast_mode else 14, fps)),
            save_path=str(videos_dir / "reactive_cdar_model.mp4"),
            max_frames=max(180 if fast_mode else 300, max_frames),
        )
        files.append({"type": "reactive_model", "path": path})
        done += 1
        progress.progress(done / max(tasks, 1))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "fps": fps,
        "max_frames": max_frames,
        "theme": theme,
        "snapshot_lookback": snapshot_lookback,
        "snapshot_step_mode": snapshot_step_mode,
        "snapshot_custom_step": snapshot_custom_step,
    }
    manifest_path = videos_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state.run_dir = str(run_dir)
    st.sidebar.success(f"Animations generated in: {videos_dir}")


def _render_overview_tab(state: UIState, projection_years: int = 5) -> None:
    st.subheader("Overview")
    metrics = state.analysis_results.get("metrics", {}) if state.analysis_results else {}

    cols = st.columns(3)
    cols[0].metric("Expected Return", _fmt_pct(metrics.get("expected_return")))
    cols[1].metric("Volatility", _fmt_pct(metrics.get("volatility")))
    cols[2].metric("CDaR", _fmt_pct(metrics.get("cdar")))

    cols2 = st.columns(3)
    cols2[0].metric("Max Drawdown", _fmt_pct(metrics.get("max_drawdown")))
    cols2[1].metric("Calmar", _fmt_num(metrics.get("calmar")))
    cols2[2].metric("Sharpe", _fmt_num(metrics.get("sharpe")))
    expected_return = metrics.get("expected_return")
    projected_value = None
    if expected_return is not None and np.isfinite(float(expected_return)):
        projected_value = (1.0 + float(expected_return)) ** projection_years
    cols3 = st.columns(2)
    cols3[0].metric(
        f"Projected Value ({projection_years}Y, start=1.00)",
        _fmt_value(projected_value),
    )
    if state.analysis_results:
        cols3[1].metric(
            "Data Window",
            (
                f"{state.analysis_results.get('data_start', '--')} to "
                f"{state.analysis_results.get('data_end', '--')}"
            ),
        )

    st.caption(
        "CDaR measures the average size of the worst drawdowns, rather than day-to-day volatility."
    )

    if state.weights is not None and state.returns_df is not None:
        wdf = pd.DataFrame(
            {
                "asset": state.returns_df.columns,
                "weight": state.weights,
            }
        )
        st.dataframe(wdf, use_container_width=True)
    else:
        st.info("Load data and run analysis to populate metrics and weights.")


def _render_underwater_tab(
    state: UIState,
    show_benchmark: bool,
    show_rolling: bool,
    theme: str,
) -> None:
    st.subheader("Underwater")
    if state.equity_curve is None or state.drawdown_series is None:
        st.info("Run analysis to populate underwater visualization.")
        return

    benchmark_values = state.benchmark_prices if show_benchmark else None
    fig = make_underwater_figure(
        values=state.equity_curve,
        drawdown=state.drawdown_series,
        benchmark_values=benchmark_values,
        rolling_cdar=state.rolling_cdar if show_rolling else None,
        theme=_theme_to_template(theme),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_frontier_tab(state: UIState, theme: str) -> None:
    st.subheader("Frontier")
    if state.frontier_df is None:
        st.info("Run analysis to populate frontier visualization.")
        return

    current = None
    if state.analysis_results.get("metrics"):
        current = {
            "cdar": state.analysis_results["metrics"].get("cdar"),
            "expected_return": state.analysis_results["metrics"].get("expected_return"),
        }
    fig = make_cdar_frontier_figure(
        frontier_df=state.frontier_df,
        current_point=current,
        theme=_theme_to_template(theme),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_surface_tab(state: UIState, show_surface: bool, theme: str) -> None:
    st.subheader("3D Surface")
    if not show_surface:
        st.info("Enable 'Show mean–variance–CDaR surface' and run analysis.")
        return
    if state.surface_df is None:
        st.info("Run analysis to populate surface visualization.")
        return

    fig = make_mean_variance_cdar_surface_figure(
        surface_df=state.surface_df,
        theme=_theme_to_template(theme),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_reactive_model_tab(state: UIState, theme: str) -> None:
    st.subheader("Reactive 3D Model")
    if state.drawdown_series is None:
        st.info("Run analysis to populate the reactive 3D model.")
        return

    ret_series = None
    if state.returns_df is not None and state.weights is not None:
        ret_series = (state.returns_df * state.weights).sum(axis=1)

    fig = make_cdar_reactive_model_figure(
        drawdown=state.drawdown_series,
        returns=ret_series,
        theme=_theme_to_template(theme),
        max_frames=260,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Model deformation intensity increases as drawdowns become deeper and more volatile."
    )

    if state.run_dir is None:
        return
    manifest_path = Path(state.run_dir) / "videos" / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reactive_entries = [
        entry for entry in manifest.get("files", []) if entry.get("type") == "reactive_model"
    ]
    if not reactive_entries:
        return

    latest_path = Path(str(reactive_entries[-1].get("path", "")))
    if not latest_path.exists():
        return

    st.markdown("### Reactive Model Video")
    preview_path = _gif_to_mp4_for_preview(latest_path)
    suffix = preview_path.suffix.lower()
    if suffix == ".mp4":
        _render_looping_video(preview_path, height=560)
    elif suffix == ".gif":
        st.image(latest_path.read_bytes(), caption=latest_path.name)
        st.info("GIF fallback detected. Install ffmpeg to enable MP4 export.")


def _render_animation_tab(state: UIState) -> None:
    st.subheader("Animations / Export")
    if state.run_dir is None:
        st.info("Generate animations from sidebar to populate exports.")
        return

    videos_dir = Path(state.run_dir) / "videos"
    st.write(f"Run directory: `{state.run_dir}`")
    manifest_path = videos_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        st.json(manifest)
        for entry in manifest.get("files", []):
            file_path = Path(entry["path"])
            if file_path.exists():
                st.markdown(f"### {entry['type'].replace('_', ' ').title()} Preview")
                preview_path = _gif_to_mp4_for_preview(file_path)
                suffix = preview_path.suffix.lower()
                if suffix == ".mp4":
                    _render_looping_video(preview_path, height=500)
                elif suffix == ".gif":
                    st.image(file_path.read_bytes(), caption=file_path.name)
                    st.info("GIF fallback detected. Install ffmpeg to enable MP4 export.")
                else:
                    st.info(f"No inline preview for {suffix} files yet.")
                st.download_button(
                    label=f"Download {entry['type']} ({file_path.suffix})",
                    data=file_path.read_bytes(),
                    file_name=file_path.name,
                    mime="application/octet-stream",
                )


def _fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{100.0 * float(value):.2f}%"


def _fmt_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{float(value):.3f}"


def _fmt_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{float(value):.2f}x"


def main() -> None:
    st.set_page_config(page_title="CDaR Lab", layout="wide")
    st.title("CDaR Lab")
    st.caption("Interactive enhanced CDaR analytics and animation studio.")

    state = get_state()
    opts = _render_sidebar(state)

    tab_overview, tab_underwater, tab_frontier, tab_surface, tab_reactive, tab_animation = (
        st.tabs(
            [
                "Overview",
                "Underwater",
                "Frontier",
                "3D Surface",
                "Reactive 3D Model",
                "Animations / Export",
            ]
        )
    )

    with tab_overview:
        _render_overview_tab(state, projection_years=int(opts["projection_years"]))
    with tab_underwater:
        _render_underwater_tab(
            state,
            show_benchmark=bool(opts["show_benchmark"]),
            show_rolling=bool(opts["show_rolling"]),
            theme=str(opts["theme"]),
        )
    with tab_frontier:
        _render_frontier_tab(state, theme=str(opts["theme"]))
    with tab_surface:
        _render_surface_tab(
            state,
            show_surface=bool(opts["show_surface"]),
            theme=str(opts["theme"]),
        )
    with tab_reactive:
        _render_reactive_model_tab(state, theme=str(opts["theme"]))
    with tab_animation:
        _render_animation_tab(state)


if __name__ == "__main__":
    main()
