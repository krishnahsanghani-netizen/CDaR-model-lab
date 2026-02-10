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
- Nested config merge helper `merge_config` for reliable override precedence.
- `analyze-portfolio` enhancements for optional benchmark-relative metrics, rolling CDaR, and optional parametric metric outputs.
- `optimize-cdar` enhancements for optional per-asset bounds CSV, gross limit override, and solver override.
- Example reproducibility config at `examples/config.example.yaml`.
- CI-runnable Python example script at `examples/example_basic_pipeline.py`.
- CLI benchmark-path test coverage for tracking error and information ratio output.
- Package build dependency and CI build step (`python -m build`).
- Optimization status guidance messages for infeasible, unbounded, target-miss, and solver-error outcomes.
- Frontier/surface outputs now carry status guidance messages in addition to status codes.
- Infeasibility-path optimization test coverage.
- New `backtest` CLI command with:
  - fixed/dynamic rebalancing mode support,
  - calendar (`none|M|Q`) and every-N rebalance controls,
  - optional benchmark-relative analytics output,
  - optional backtest series export to CSV.
- CLI test coverage for the `backtest` command path with benchmark and rebalancing options.
- `frontier` CLI enhancements:
  - explicit `--no-short/--allow-short`,
  - optional `--gross-limit`,
  - optional `--return-min/--return-max`,
  - `--parallel/--no-parallel` and `--n-jobs`.
- `surface` CLI enhancements:
  - optional `--lambda-grid-json`,
  - preset grids via `--lambda-preset small|medium`,
  - explicit shorting/gross/parallel controls.
- `run-pipeline` enhancement:
  - explicit `--no-short/--allow-short`,
  - writes `results/summary.json` artifact.
- Additional CLI tests for advanced frontier and surface command paths.
- Added scenario analysis API in `portfolio/scenario.py` for:
  - additive global return shifts,
  - per-asset shock maps,
  - baseline vs scenario metric comparison.
- Added `scenario` CLI command with:
  - preset scenarios (`basic`),
  - optional custom `--scenarios-json`,
  - optional CSV export of scenario results.
- Added scenario unit and CLI test coverage.
- Added regime analysis API in `portfolio/regime.py` for yearly/quarterly/monthly subperiod metrics.
- Added `regime` CLI command for cross-regime CDaR comparison with CSV export.
- Added regime unit and CLI test coverage.
- Added Phase 8 hardening tests for:
  - weight defaults and gross-exposure constraint validation,
  - per-asset bounds CSV loading,
  - frontier status/message behavior under extreme target-return ranges,
  - scenario validation edge cases,
  - strict missing-data preprocessing policy behavior.
- Added hardening tests for rebalance schedule behavior (`M` boundaries and every-N rules).
- Added regime validation tests for non-datetime index rejection and minimum-period filtering.
- Removed `type: ignore` in regime CLI by explicit `RegimeFrequency` narrowing/cast.
- Updated README quality-gate section to reflect current CI checks.

### Changed
- Corrected module naming to `data/preprocess.py` in scaffold.
- Switched CLI override application to validated nested config merge behavior.
- CI lint gate now runs `ruff` only; import ordering and Black checks are not separately gated.
- Ruff rule selection updated to focus on active error classes (`E`, `F`, `UP`, `B`).

### Fixed
- Replaced invalid empty notebook placeholder with a valid JSON notebook at `examples/example_basic_pipeline.ipynb` to satisfy Ruff notebook parsing.
- Wrapped long CLI lines in `src/enhanced_cdar/cli.py` to satisfy Ruff `E501` line-length checks.
- Wrapped all remaining >100-character Python lines across `src/` and `tests/` to satisfy repository line-length linting.
- Rewrote `examples/example_basic_pipeline.ipynb` cell sources with valid newline encoding to fix Ruff notebook parsing/syntax failures.
- Resolved Ruff `UP035`, `F401`, and `E741` issues in data/backtest/optimization modules.
- Fixed Python 3.10 compatibility in data metadata timestamps by replacing `datetime.UTC` with `timezone.utc`.
- Added `types-PyYAML` to development dependencies to satisfy mypy typed-import checks.
- Fixed mypy typing issues in CDaR/loss helper return types, dynamic optimizer callback narrowing, and CLI lambda-grid tuple typing.
- Fixed Python 3.10 mypy scalar inference in historical VaR/CVaR quantile handling.
