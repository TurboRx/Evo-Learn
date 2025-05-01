# Evo-Learn

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/TurboRx/Evo-Learn/evo_learn_workflow.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A sophisticated machine learning automation and verification platform leveraging TPOT for intelligent model selection and performance optimization.

## Core Capabilities

- **Automated Machine Learning**: Intelligent model exploration and selection using TPOT
- **Comprehensive Verification**: Package and module validation for reliable deployments
- **Advanced ML Workflows**: Streamlined pipelines for data preprocessing, model training, and evaluation
- **Hyperparameter Tuning**: Dynamic optimization of model parameters

## Quick Start

### Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tpot
```

### Basic Usage

```python
from enhanced_core import run_automl

# Run automated machine learning
results = run_automl(
    data_path="your_dataset.csv",
    target_column="target",
    task="classification",
    generations=5,
    population_size=20
)

# Check model performance
print(f"Model accuracy: {results['metrics']['accuracy']:.4f}")
```

## GitHub Integration

This repository includes powerful GitHub integration scripts:

### Manual Push

Push specific files with proper attribution:

```bash
node push_as_turborx.js --files "file1.py,file2.py" --message "Your commit message"
```

### Automated Push

Automatically detect and push changed files:

```bash
node auto-push.js --token "$GITHUB_TOKEN" --name "TurboRx" --email "your.email@example.com"
```

### GitHub Actions

The repository uses GitHub Actions for CI/CD:

- Automatic testing across multiple Python versions
- Verification of core functionality
- Comprehensive test coverage

## Verification

You can verify your Evo-Learn installation using:

```bash
python verify_evo_learn.py
```

## License

MIT

## Author

Built by [TurboRx](https://github.com/TurboRx)