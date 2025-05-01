# Evo-Learn: Automated Machine Learning Tool

Evo-Learn is an open-source automated machine learning tool designed to streamline the model selection and hyperparameter tuning process. Leveraging the power of TPOT (Tree-based Pipeline Optimization Tool), Evo-Learn empowers users to quickly identify optimal machine learning pipelines, saving valuable time and resources.

## Features

### Automated Model Selection
- Automatically searches for the best machine learning models and hyperparameters
- Supports both classification and regression tasks
- Uses genetic programming to optimize machine learning pipelines
- Provides easy-to-use interfaces for training, evaluation, and prediction

### Data Preprocessing
- Handles missing values with customizable imputation strategies
- Automatically encodes categorical features
- Scales numerical features for improved model performance
- Supports custom preprocessing pipelines

### Advanced Visualization
- Generate feature distribution plots to understand your data
- Create correlation matrices to identify relationships between features
- Visualize confusion matrices for classification models
- Plot ROC curves and precision-recall curves for model evaluation
- Generate feature importance plots to identify influential features

### Model Management
- Save trained models for future use
- Export optimized pipelines as Python code
- Track model metadata and performance metrics
- Evaluate models on new datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/TurboRx/Evo-Learn.git
cd Evo-Learn

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

Evo-Learn provides a user-friendly command-line interface for all operations:

```bash
# Training a classification model
python evo_learn_cli.py train --data your_data.csv --target target_column --generations 10 --population 30 --visualize

# Making predictions with a trained model
python evo_learn_cli.py predict --model models/your_model.pkl --data new_data.csv --output predictions.csv

# Evaluating a model
python evo_learn_cli.py evaluate --model models/your_model.pkl --data test_data.csv --target target_column

# Creating visualizations
python evo_learn_cli.py visualize --data your_data.csv --target target_column
```

### Python API

You can also use Evo-Learn as a Python library in your code:

```python
from mloptimizer.core import run_automl

# Run automated machine learning
result = run_automl(
    data_path='your_data.csv',
    target_column='target',
    task='classification',
    generations=5,
    population_size=20
)

# Access the results
print(f"Accuracy: {result['metrics']['accuracy']}")
print(f"Model saved to: {result['model_path']}")
```

## Advanced Usage

### Customizing the Optimization Process

```python
from mloptimizer.core import run_automl

# Run with custom parameters
result = run_automl(
    data_path='your_data.csv',
    target_column='target',
    task='classification',
    generations=10,        # More generations for better results
    population_size=50,    # Larger population for broader search
    max_time_mins=60,      # Limit runtime to 60 minutes
    max_eval_time_mins=5   # Limit each pipeline evaluation to 5 minutes
)
```

### Custom Data Preprocessing

```python
from mloptimizer.preprocessing import preprocess_data

# Custom preprocessing
processed_data = preprocess_data(
    data_path='your_data.csv',
    categorical_features=['feature1', 'feature2'],
    numerical_features=['feature3', 'feature4'],
    impute_strategy='median'  # Use median for imputing missing values
)
```

### Creating Visualizations

```python
from mloptimizer.visualization import plot_feature_distributions, plot_correlation_matrix

# Create visualizations
plot_feature_distributions(
    data=your_dataframe,
    target_column='target',
    output_dir='visualizations'
)

plot_correlation_matrix(
    data=your_dataframe,
    output_path='visualizations/correlation.png'
)
```

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- TPOT
- numpy
- matplotlib
- seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2025 Evo-Learn Contributors

## Acknowledgments

- TPOT: Tree-based Pipeline Optimization Tool
- scikit-learn: Machine learning in Python
- pandas: Data manipulation and analysis

## Citation

If you use Evo-Learn in your research, please cite:

```
@software{evolearn,
  author = {TurboRx},
  title = {Evo-Learn: An Automated Machine Learning Tool},
  url = {https://github.com/TurboRx/Evo-Learn},
  year = {2025},
}
```