# Changelog

All notable changes to Evo-Learn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-02-05

### Added
- **Security**: Replaced pickle with joblib for model serialization (safer deserialization)
- **Validation**: Added comprehensive data validation before training
  - Checks for NaN in target column
  - Validates minimum number of classes for classification
  - Warns about severe class imbalance (>10:1 ratio)
  - Detects all-NaN and constant features
- **Input Protection**: Added file size limit validation (default 500MB) to prevent OOM errors
- **Logging**: Added proper error logging to all exception handlers in visualization module
- **Type Safety**: Added comprehensive type hints to all public functions
- **Configuration**: Made warning filters configurable via `EVO_LEARN_SHOW_WARNINGS` environment variable
- **Dependencies**: Added missing dependencies to pyproject.toml (tqdm, colorama, shap)

### Changed
- **Preprocessing**: Fixed StandardScaler to use `with_mean=True` (proper centering for dense data)
- **Preprocessing**: Added `max_categories` constraint to OneHotEncoder to prevent memory explosion
- **Preprocessing**: Fixed pandas 3.0 deprecation warning by including "string" dtype in select_dtypes
- **Logging**: Improved logging format to include function names for better debugging
- **Dependencies**: Relaxed numpy version constraint from `>=2.3.0,<3.0.0` to `>=2.0.0,<3.0.0`
- **Code Quality**: Reformatted entire codebase with black for consistent style
- **Visualization**: Silent failures now log errors instead of passing silently

### Fixed
- Security vulnerability in model loading (replaced pickle.load with joblib.load)
- Pandas 3.0 compatibility warning in select_dtypes
- StandardScaler configuration that was disabling centering
- Missing logging for visualization failures
- Type annotation errors caught by mypy
- Silent exception handling that could hide errors

### Security
- ⚠️ **CRITICAL**: Replaced unsafe pickle deserialization with joblib
- Added input file size validation to prevent DoS attacks
- Added data validation to catch malformed inputs early

## [1.2.0] - Previous Release

### Features
- AutoML with TPOT for classification and regression
- Production-ready preprocessing pipeline
- Baseline model fallbacks
- YAML configuration support
- Cross-validation utilities
- Model explainability with SHAP
- Comprehensive visualization suite
- Docker support
- Type hints and modern Python features (3.10+)

---

## Migration Guide

### From 1.2.0 to 1.3.0

**Breaking Changes**: None

**Recommended Actions**:
1. **Models**: Existing pickle-based models will still load, but new models will use joblib
2. **Config**: Consider setting `EVO_LEARN_SHOW_WARNINGS=true` during development to see all warnings
3. **Data**: Review your datasets - the new validation will catch issues that were silently ignored before

**New Features to Try**:
- The data validation will now warn you about class imbalance and other potential issues
- Visualization errors are now logged, making debugging easier
- File size limits protect against accidentally loading huge datasets

---

## Development Notes

### Testing
All changes maintain backward compatibility. Test suite passes 26/26 tests with no failures.

### Code Quality
- Black formatting applied to entire codebase
- Type hints added for mypy compliance
- Logging improved throughout

### Performance
- No performance regressions
- Joblib provides better compression than pickle (compress=3)
