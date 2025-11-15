# Evo-Learn Examples

This directory contains example scripts demonstrating various use cases and features of Evo-Learn.

## Examples Overview

### 1. Basic Classification (`01_basic_classification.py`)
Simple binary classification example showing the fundamental workflow.

**Run it:**
```bash
python examples/01_basic_classification.py
python cli.py train --data examples/sample_classification.csv --target target --baseline
```

### 2. Regression with Missing Data (`02_regression_with_missing_data.py`)
Demonstrates handling datasets with missing values using different imputation strategies.

**Run it:**
```bash
python examples/02_regression_with_missing_data.py
python cli.py train --data examples/sample_regression_missing.csv --target target --config examples/regression_config.yaml
```

### 3. Imbalanced Classification (`03_imbalanced_classification.py`)
Shows how to work with imbalanced datasets and interpret appropriate metrics.

**Run it:**
```bash
python examples/03_imbalanced_classification.py
python cli.py train --data examples/imbalanced_classification.csv --target target --baseline
```

### 4. Categorical Features (`04_categorical_features.py`)
Handling mixed numeric and categorical features with automatic encoding.

**Run it:**
```bash
python examples/04_categorical_features.py
python cli.py train --data examples/categorical_features.csv --target target --baseline
```

## Running Examples

### Prerequisites
Ensure Evo-Learn is installed:
```bash
pip install -r ../requirements.txt
```

### Generate Sample Data
Run any example script to generate its dataset:
```bash
python examples/01_basic_classification.py
```

### Train Models
Use the CLI to train models on generated data:
```bash
# Quick baseline model
python cli.py train --data examples/sample_classification.csv --target target --baseline

# Full TPOT optimization (slower)
python cli.py train --data examples/sample_classification.csv --target target --config evo_config.yaml
```

### Make Predictions
```bash
python cli.py predict --model mloptimizer/models/model.pkl --data examples/sample_classification.csv --output predictions.csv
```

### Evaluate Models
```bash
python cli.py evaluate --model mloptimizer/models/model.pkl --data examples/sample_classification.csv --target target --output-dir evaluation_results
```

## Tips

- **Start with baseline models** (`--baseline` flag) for quick prototyping
- **Use TPOT** for final model optimization (remove `--baseline`)
- **Adjust config** files for different hyperparameters
- **Check output artifacts** in `mloptimizer/models/` directory
- **Review visualizations** generated during evaluation

## Adding Your Own Examples

1. Create a new Python script in this directory
2. Generate or load your dataset
3. Document the use case and features demonstrated
4. Update this README with your example

## Common Issues

**Issue**: Import errors when running examples
**Solution**: Ensure you're in the project root directory

**Issue**: TPOT takes too long
**Solution**: Reduce `generations` and `population_size` in config, or use `--baseline`

**Issue**: Memory errors with large datasets
**Solution**: Reduce dataset size or use sampling

## Further Resources

- [Main README](../README.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [API Documentation](../docs/) (coming soon)
