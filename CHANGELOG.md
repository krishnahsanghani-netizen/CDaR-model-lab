# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Initial planning document in `IMPLEMENTATION_PLAN.md`.
- Repository scaffold for the enhanced CDaR package.
- Packaging and tooling setup via `pyproject.toml`, pinned `requirements.txt`, and CI workflow for Python 3.10/3.11.
- Typed configuration models with YAML loading and override merge behavior.
- Data loading utilities for CSV and yfinance with cache support and metadata capture.
- Price preprocessing utilities with configurable missing-data policy and normalization.
- Core metrics implementation for returns, drawdowns, empirical CDaR, rolling CDaR, VaR/CVaR, and Sharpe/Sortino/Calmar.
- Portfolio constraint utilities including default long-only/long-short bounds and per-asset bounds CSV loader.
- Backtesting engine with close-to-close execution assumption, fixed/dynamic rebalancing modes, and benchmark-relative analytics.
- Initial unit tests for drawdown, CDaR, and backtest behavior.
- CDaR optimization module with ECOS primary / SCS fallback, frontier generation, and mean-variance-CDaR surface generation.
- Plotting modules for underwater charts, Mean-CDaR frontier, and Mean-Variance-CDaR surfaces (Plotly primary, Matplotlib fallback).
- Typer-based CLI with commands: `fetch-data`, `analyze-portfolio`, `optimize-cdar`, `frontier`, `surface`, and `run-pipeline`.
- CLI support for `--format text|json`, `--config`, `--verbose`, and `--quiet`.
- Initial fixture and integration dataset under `tests/data/` plus integration-style optimization test.
- Hypothesis-based metric property tests for CDaR and max drawdown non-negativity.
- Expanded README with mini-docs style quickstart and CLI usage.

### Changed
- Corrected module naming to `data/preprocess.py` in scaffold.

### Fixed
- Replaced invalid empty notebook placeholder with a valid JSON notebook at `examples/example_basic_pipeline.ipynb` to satisfy Ruff notebook parsing.
- Wrapped long CLI lines in `src/enhanced_cdar/cli.py` to satisfy Ruff `E501` line-length checks.
