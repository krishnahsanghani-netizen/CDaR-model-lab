# Enhanced CDaR Model Implementation Plan

## Milestone A: Foundation + Core Metrics
- Phase 0: Repository foundation
  - Create package layout and `src/` structure.
  - Add `pyproject.toml` with setuptools and CLI entry point `enhanced-cdar`.
  - Add pinned `requirements.txt` and development tooling configuration.
  - Add `LICENSE`, `.gitignore`, and `CHANGELOG.md`.
- Phase 1: Config, data loading, preprocessing, caching, metadata
- Phase 2: Core metrics (returns, drawdown, CDaR, VaR/CVaR, Sharpe/Sortino/Calmar, rolling CDaR)

## Milestone B: Portfolio + Optimization
- Phase 3: Constraints, bounds, backtesting, rebalancing, benchmark-relative analytics
- Phase 4: CDaR optimization with cvxpy and solver fallback
- Phase 5: Efficient frontier and mean-variance-CDaR surface with parallelization

## Milestone C: Visualization + CLI
- Phase 6: Plotly-first plots with matplotlib fallback
- Phase 7: Typer CLI commands as thin API wrappers

## Milestone D: Quality + Release
- Phase 8: Tests (unit, property, integration), CI, coverage target, docs hardening
- Release `0.1.0`

## Tracking Method
- Progress is tracked in `CHANGELOG.md` under an `Unreleased` section.
- I will check in after each major PRD feature/milestone completion.
