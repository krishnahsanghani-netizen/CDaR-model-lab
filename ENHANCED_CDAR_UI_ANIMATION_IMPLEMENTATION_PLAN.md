# Enhanced CDaR UI + Animation Implementation Plan (v0.2.0)

## Scope Baseline
- Build Streamlit UI (`ui/streamlit_app.py`) and animation module (`src/enhanced_cdar/viz/animation.py`) in parallel.
- Reuse existing analytics/optimization APIs; no math reimplementation.
- Ship as `v0.2.0` after quality gates pass.
- Out of scope: auth, DB, live streaming, websockets, multi-user sync.

## Execution Rules
- Commit and push after each major phase completion.
- Keep UI local-first with Streamlit Community Cloud compatibility.
- If no internet/yfinance fails: show actionable fallback to CSV upload.

---

## Phase 1: UI/App Skeleton + State Model
### Deliverables
- Create `ui/streamlit_app.py` scaffold.
- Add typed UI state container (dataclass) for loaded data/results.
- Add tab structure: Overview, Underwater, Frontier, 3D Surface, Animations/Export.
- Add minimal sidebar sections and session initialization.

### Acceptance Criteria
- App runs via `streamlit run ui/streamlit_app.py`.
- All tabs render without runtime errors before any data is loaded.
- Session state object persists across reruns.

---

## Phase 2: Data Input Layer (Example / yfinance / CSV)
### Deliverables
- Sidebar data source controls with:
  - cached example dataset (default),
  - yfinance fetch (tickers/start/end),
  - CSV upload with date/price column auto-detection + manual override.
- Offline-safe yfinance behavior with clear warning/hint to CSV upload.
- Load Data action stores canonical `prices_df` in session state.

### Acceptance Criteria
- Example dataset loads in one click.
- CSV auto-detection works for normal date+numeric files and can be overridden manually.
- yfinance failure path provides user-readable fallback guidance.

---

## Phase 3: Analysis Orchestrator (Run Analysis)
### Deliverables
- Add Run Analysis action that computes and stores:
  - returns,
  - weights (equal-weight / min-CDaR / frontier-selected point),
  - equity curve + drawdown,
  - rolling CDaR (optional),
  - frontier and surface frames (if enabled),
  - benchmark series/metrics.
- Portfolio mode selectors:
  - target return slider,
  - CDaR percentile slider along frontier.
- Rebalancing in UI: static + periodic (M/Q/every-N), no dynamic re-opt in UI.

### Acceptance Criteria
- Clicking Run Analysis updates all tab data artifacts consistently.
- Selected portfolio mode changes metrics and plots deterministically.
- Rebalancing settings affect backtest outputs as expected.

---

## Phase 4: Overview Tab (Metrics + Allocation)
### Deliverables
- `st.metric` cards for expected return, vol, CDaR, max drawdown, Calmar, Sharpe.
- Allocation table for current weights.
- beginner-friendly explanatory text for CDaR.

### Acceptance Criteria
- Metrics match backend outputs from stored result objects.
- Table and cards update after each analysis run.
- Empty-state guidance shown when analysis hasn’t run yet.

---

## Phase 5: Underwater Interactive View
### Deliverables
- Add `make_underwater_figure(...)` in `viz/underwater.py`.
- Render equity + drawdown with optional benchmark and rolling CDaR overlay.
- Theme toggle (light/dark) mapped to Plotly template.

### Acceptance Criteria
- Underwater tab shows two-pane interactive chart with hover info.
- Benchmark/rolling toggles work without reloading data.
- Figure can be reused by non-UI callers.

---

## Phase 6: Frontier + Surface Interactive Views
### Deliverables
- Add `make_cdar_frontier_figure(...)` in `viz/frontier.py`.
- Add `make_mean_variance_cdar_surface_figure(...)` in `viz/surfaces.py`.
- Highlight selected portfolio point on frontier.
- 3D surface rendering with optional 2D projection mode.

### Acceptance Criteria
- Frontier tab displays curve + selected point + hover details.
- Surface tab displays valid 3D plot from computed surface dataframe.
- Plots respect light/dark theme.

---

## Phase 7: Animation Core Module
### Deliverables
- Implement `viz/animation.py` with:
  - `animate_underwater(...)`,
  - `animate_frontier_over_time(...)`,
  - `animate_surface_over_time(...)`.
- MP4 preferred via ffmpeg; fallback GIF/image-sequence when ffmpeg unavailable.
- Automatic frame downsampling to configurable max frame count (default 300).

### Acceptance Criteria
- Each function returns saved output path.
- If ffmpeg missing, generation still succeeds via fallback with warning.
- Functions handle long histories without excessive runtime/memory.

---

## Phase 8: Snapshot Generation Engine (Frontier/Surface over Time)
### Deliverables
- Implement rolling-window snapshot generation (default 252 lookback, monthly step).
- Step options: monthly, quarterly, custom N-period step.
- Optional consume-precomputed snapshots support for advanced users.

### Acceptance Criteria
- Snapshot generation works from raw prices without manual preprocessing.
- Output shape is valid for animation functions.
- Step and lookback controls are respected.

---

## Phase 9: Animations/Export Streamlit Tab
### Deliverables
- Add UI controls for animation selection, FPS, duration-ish controls, max frames.
- Generate selected animations using computed data/snapshots.
- Save to `runs/<timestamp>/videos/` and produce manifest JSON.
- Show progress bar and download buttons.

### Acceptance Criteria
- User can generate one or multiple animations from UI session data.
- Manifest includes filenames, format, fps, frame count, timestamps.
- Downloads function correctly from Streamlit.

---

## Phase 10: CLI Expansion for UI + Animations
### Deliverables
- Add CLI commands:
  - `enhanced-cdar ui` (pass-through streamlit flags),
  - `enhanced-cdar animate-underwater`,
  - `enhanced-cdar animate-frontier`,
  - `enhanced-cdar animate-surface`.
- `animate-*` accepts both `--run-dir` and `--prices-csv` (`--run-dir` wins).
- `animate-frontier/surface` can generate snapshots from prices and optionally consume snapshot files.

### Acceptance Criteria
- Commands execute with clear help text.
- `enhanced-cdar ui --port ... --server-headless ...` forwards correctly.
- `animate-*` writes outputs + manifest into expected directories.

---

## Phase 11: Testing + Performance Hardening
### Deliverables
- Unit tests for animation helpers (frame handling, fallback behavior, no-exception paths).
- UI utility tests (config parsing, data source normalization, auto-detection helpers).
- Smoke integration test for tiny animation artifact generation.
- CI skips heavy rendering tests when ffmpeg unavailable.

### Acceptance Criteria
- New tests pass on 3.10/3.11 CI matrix.
- No hard CI dependency on ffmpeg installation.
- Runtime for smoke animation test remains CI-friendly.

---

## Phase 12: Documentation, Release Prep, and v0.2.0
### Deliverables
- Expand README with:
  - UI launch instructions,
  - animation usage examples,
  - troubleshooting (offline/yfinance/ffmpeg fallback),
  - Streamlit Cloud deployment notes.
- Add/update example config + command snippets.
- Update changelog for UI/animation release.
- Bump version to `0.2.0`, tag release.

### Acceptance Criteria
- End-to-end quickstart works for local user.
- Changelog and release notes map to shipped features.
- Git tag `v0.2.0` pushed after green CI.

---

## Commit Milestones (Planned)
1. Phase 1-2 scaffold + data inputs
2. Phase 3-4 analysis + overview
3. Phase 5-6 interactive plots
4. Phase 7-8 animation core + snapshots
5. Phase 9-10 export tab + animation CLI
6. Phase 11-12 tests/docs/release and `v0.2.0`

## Risks and Mitigations
- ffmpeg availability inconsistency:
  - Mitigation: fallback chain (MP4 -> GIF -> PNG sequence) with warnings.
- Streamlit rerun/state complexity:
  - Mitigation: typed session container + strict state transitions.
- Long-history animation performance:
  - Mitigation: configurable frame cap + automatic downsampling.
- yfinance network fragility:
  - Mitigation: robust exception path + CSV-first fallback UX.
