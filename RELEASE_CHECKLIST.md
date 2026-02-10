# Release Checklist (v0.1.0)

## Pre-Release
- Ensure `CHANGELOG.md` includes all user-visible changes.
- Verify package metadata in `pyproject.toml` (version, license, author placeholder).
- Confirm README examples and CLI commands are current.

## Quality Gates
- Run lint: `ruff check .`
- Run type check: `mypy src`
- Run tests with coverage: `pytest`
- Build package: `python -m build`

## Validation
- Smoke test CLI commands:
  - `enhanced-cdar analyze-portfolio`
  - `enhanced-cdar optimize-cdar`
  - `enhanced-cdar frontier`
  - `enhanced-cdar surface`
  - `enhanced-cdar backtest`
  - `enhanced-cdar scenario`
  - `enhanced-cdar regime`

## Publish Prep
- Update version tag according to SemVer.
- Create release notes from `CHANGELOG.md`.
- Push tag and publish artifact (when ready).
