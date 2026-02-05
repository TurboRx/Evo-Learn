# Evo-Learn

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready AutoML toolkit built on TPOT**

[Installation](#installation) • [Quick Start](#quick-start) • [Configuration](#configuration)

</div>

---

## Features

- **Automated Model Selection**: TPOT-powered search for classification & regression
- **Production Pipeline**: Preprocessing with imputation, encoding, and scaling
- **Smart Fallbacks**: Baseline models when TPOT fails or times out
- **Config-Driven**: YAML configuration for reproducible runs
- **Data Validation**: Catches NaN, single-class, and imbalance issues
- **Visualizations**: ROC/PR curves, residuals, feature distributions
- **Type-Safe**: Complete type hints throughout
- **Secure**: Joblib serialization, path traversal protection

## Installation

### From Source

```bash
git clone https://github.com/TurboRx/Evo-Learn.git
cd Evo-Learn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_installation.py
```

### Docker

```bash
docker-compose up evo-learn
docker-compose run evo-learn-test  # Run tests
```

## Quick Start

### Python API

```python
from core import run_automl

result = run_automl(
    data_path="data.csv",
    target_column="target",
    task="classification",
    generations=5,
    population_size=20
)

print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
print(f"Model saved: {result['model_path']}")
```

### Command Line

```bash
python cli.py train data.csv target --task classification --generations 5
python cli.py predict model.pkl test_data.csv --target target
python cli.py evaluate model.pkl test_data.csv target
```

## Configuration

Create `config.yaml`:

```yaml
handle_categoricals: true
impute_strategy: median
scale_numeric: true
output_dir: models
n_jobs: -1
```

Use with:

```python
result = run_automl(
    data_path="data.csv",
    target_column="target",
    config_path="config.yaml"
)
```

## Project Structure

```
.
├── core.py              # Main AutoML logic
├── preprocessing.py     # Data preprocessing
├── visualization.py     # Plots and dashboards
├── cli.py              # Command-line interface
├── utils.py            # Utility functions
├── examples/           # Usage examples
└── tests/             # Test suite
```

## Examples

See `examples/` directory:
- `01_basic_classification.py`
- `02_regression_with_missing_data.py`
- `03_imbalanced_classification.py`
- `04_categorical_features.py`
- `05_new_features_demo.py`

## Testing

```bash
pytest tests/
pytest tests/ --cov=. --cov-report=html
```

## Development

```bash
pip install -e .
pre-commit install
black .
flake8 .
mypy .
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Pull requests welcome. Please ensure:
- Tests pass
- Code formatted with Black
- Type hints included
- Documentation updated

---

**Note**: Requires Python 3.10+
