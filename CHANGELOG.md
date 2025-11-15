# Changelog

All notable changes to the Evo-Learn project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed broken unit tests in `tests/test_core.py` and `tests/test_preprocessing.py` that were using StringIO incorrectly
- Fixed OneHotEncoder deprecation warning by changing `sparse=False` to `sparse_output=False` for scikit-learn 1.4+ compatibility
- Fixed smoke test to support standalone execution with proper path handling
- Updated `verify_evo_learn.py` to check for correct module names instead of non-existent ones
- Fixed example_usage.py to use correct imports and modern API

### Improved
- **Enhanced error handling**: Added comprehensive error messages with actionable suggestions
  - Better file validation with detailed error messages
  - Minimum column requirements checking
  - Data quality warnings when too much data is removed
- **Input validation**: Added validation for file extensions, minimum data requirements, and edge cases
- **Code quality**: Cleaner, more maintainable code structure throughout
- **Backward compatibility**: Made `mloptimizer` package a compatibility layer that wraps root modules
  - Old code using `from mloptimizer.core import run_automl` still works
  - Deprecation warnings guide users to new import style
- **Documentation**: Improved docstrings and error messages for better user experience

### Changed
- Converted `mloptimizer/` directory to backward compatibility layer
- Updated all example code to use root-level imports
- Enhanced load_data function with better validation and error messages
- Updated .gitignore to exclude generated model files

### Tested
- All 14 tests pass successfully ✓
- Verification script passes ✓
- CLI commands (train, predict, evaluate) work correctly ✓
- End-to-end testing confirms full functionality ✓

## [1.2.0] - Previous Release

### Added
- Full preprocessing pipeline with imputation, encoding, and scaling
- Baseline models (LogisticRegression/Ridge) when TPOT fails or times out
- Config-driven runs via YAML configuration
- Stratified splits for classification tasks
- Rich visualizations (ROC/PR curves, residual plots)
- Docker support for containerized deployment
- Comprehensive test suite with 80%+ coverage
- CI/CD workflows for testing, linting, and releases

### Features
- Automated model search using TPOT
- Support for both classification and regression tasks
- Type-safe code with comprehensive type hints
- Production-ready pipeline structure

## [1.0.0] - Initial Release

### Added
- Basic AutoML functionality using TPOT
- Data preprocessing capabilities
- Model training and evaluation
- Command-line interface
- Documentation and examples
