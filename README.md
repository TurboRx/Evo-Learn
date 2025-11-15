# Evo-Learn

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/TurboRx/Evo-Learn/workflows/Tests/badge.svg)](https://github.com/TurboRx/Evo-Learn/actions)
[![codecov](https://codecov.io/gh/TurboRx/Evo-Learn/branch/main/graph/badge.svg)](https://codecov.io/gh/TurboRx/Evo-Learn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A robust AutoML toolkit built on TPOT with production-friendly preprocessing, config-driven runs, baseline fallbacks, and comprehensive CI/CD.

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Contributing](#-contributing)

</div>

---

## ✨ Features

- **Automated Model Search**: Leverages TPOT for intelligent model selection (classification & regression)
- **Production-Ready Pipeline**: Full preprocessing with imputation, encoding, and optional scaling
- **Smart Fallbacks**: Baseline models (LogisticRegression/Ridge) when TPOT fails or times out
- **Config-Driven**: YAML configuration for reproducible experiments
- **Stratified Splits**: Automatic stratification for classification tasks
- **Rich Visualizations**: ROC/PR curves, residual plots, actual vs predicted
- **Docker Support**: Containerized deployment ready
- **Type-Safe**: Comprehensive type hints throughout
- **Well-Tested**: Extensive test suite with 80%+ coverage
- **CI/CD Ready**: GitHub Actions workflows for testing, linting, and releases

## 🚀 Installation

### From PyPI (Coming Soon)

```bash
pip install evo-learn
```

### From Source

**Requirements**: Python 3.10 or 3.11

```bash
# Clone the repository
git clone https://github.com/TurboRx/Evo-Learn.git
cd Evo-Learn

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python verify_evo_learn.py
```

### Using Docker

```bash
# Build and run
docker-compose up evo-learn

# For development
docker-compose run evo-learn-dev bash

# Run tests
docker-compose run evo-learn-test
```

## 💡 Quick Start

### Python API

```python
from enhanced_core import run_automl

# Classification example
result = run_automl(
    data_path="your_dataset.csv",
    target_column="target",
    task="classification",
    generations=5,
    population_size=20,
    config_path="evo_config.yaml"  # optional
)

print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
print(f"F1 Score: {result['metrics']['f1']:.4f}")
```

### Command Line Interface

```bash
# Train a model
python cli.py train \
    --data dataset.csv \
    --target target_column \
    --task classification \
    --config evo_config.yaml

# Use baseline model (faster, no TPOT)
python cli.py train \
    --data dataset.csv \
    --target target_column \
    --baseline

# Make predictions
python cli.py predict \
    --model mloptimizer/models/model.pkl \
    --data new_data.csv \
    --output predictions.csv

# Evaluate model with visualizations
python cli.py evaluate \
    --model mloptimizer/models/model.pkl \
    --data test_data.csv \
    --target target_column \
    --output-dir evaluation_results
```

## ⚙️ Configuration

Create `evo_config.yaml` to customize behavior:

```yaml
# Task settings
default_task: classification  # or 'regression'
random_state: 42
test_size: 0.2

# TPOT parameters
generations: 5
population_size: 20
max_time_mins: 10
max_eval_time_mins: 5

# Preprocessing
handle_categoricals: true
impute_strategy: median  # 'mean', 'median', 'most_frequent'
scale_numeric: true

# Model selection
baseline: false  # Set true to skip TPOT and use baseline model

# Output
output_dir: mloptimizer/models
```

CLI flags override config settings when provided.

## 📊 Output Artifacts

### Training Outputs

```
mloptimizer/models/
├── model.pkl                    # Trained pipeline (preprocessing + model)
├── classification_metadata.json # Metrics, parameters, timestamps
├── tpot_pipeline.py             # TPOT-exported code (if available)
└── feature_importance.png       # Feature importance plot (if supported)
```

### Evaluation Outputs

```
evaluation_results/
├── evaluation_report.json       # Detailed metrics
├── roc_curve.png                # ROC curve (binary classification)
├── pr_curve.png                 # Precision-Recall curve
├── residuals.png                # Residual plot (regression)
└── actual_vs_pred.png           # Actual vs Predicted (regression)
```

## 📚 Documentation

### Examples

Check the [`examples/`](examples/) directory for:
- Handling imbalanced datasets
- Custom preprocessing pipelines
- Model interpretation
- Cross-validation workflows
- Hyperparameter tuning

### API Reference

Detailed API documentation is available in the [docs/](docs/) directory (coming soon).

## 🧪 Troubleshooting

| Issue | Solution |
|-------|----------|
| TPOT fails or times out | Use `--baseline` flag or set `baseline: true` in config |
| Python version errors | Ensure Python 3.10 or 3.11 is installed |
| Missing dependencies | Run `pip install -r requirements.txt` |
| Import errors | Verify installation with `python verify_evo_learn.py` |
| Class imbalance | Stratified splits are applied automatically |
| Mixed data types | Automatic imputation and encoding handles this |

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov

# Format code
black .
isort .

# Type checking
mypy .
```

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Specific test file
pytest tests/test_core.py -v

# Integration tests only
pytest tests/test_integration.py -v

# Parallel execution
pytest tests/ -n auto
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest tests/`
5. Commit with conventional commits: `feat: add amazing feature`
6. Push and create a Pull Request

## 📄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**TurboRx**
- GitHub: [@TurboRx](https://github.com/TurboRx)
- Repository: [Evo-Learn](https://github.com/TurboRx/Evo-Learn)

## ⭐ Star History

If you find Evo-Learn useful, please consider starring the repository!

---

<div align="center">

Made with ❤️ by [TurboRx](https://github.com/TurboRx)

</div>
