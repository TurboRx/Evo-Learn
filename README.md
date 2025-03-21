# Evo Learn

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evo Learn is an open-source automated machine learning (AutoML) tool designed to streamline the model selection and hyperparameter tuning process. Leveraging the power of TPOT (Tree-based Pipeline Optimization Tool), Evo Learn empowers users to quickly identify optimal machine learning pipelines, saving valuable time and resources.

## Features

* **Automated Model Selection:** Automatically searches for the best machine learning models and their hyperparameters.
* **Data Preprocessing:** Includes functions for handling missing values and encoding categorical features.
* **TPOT Integration:** Seamlessly integrates with TPOT for robust pipeline optimization.
* **Easy-to-Use Interface:** Provides an advanced Python API for quick integration into existing workflows.
* **Model Export:** Exports the best-performing model as a Python script for easy deployment.
* **Unit Testing:** Includes comprehensive unit tests to ensure code reliability.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/TurboRx/Evo-Learn.git
    cd Evo-Learn
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Prepare Your Dataset

Ensure your dataset is in CSV format and place it in the `examples/data/` directory.

### Run the Example Script

```python
from mloptimizer.core import run_automl
from mloptimizer.preprocessing import preprocess_data

# Example usage
data_path = 'examples/data/your_dataset.csv'
target_column = 'target'
categorical_features = ['categorical_feature1', 'categorical_feature2']  # Replace with your actual features

# Preprocess the data
preprocessed_data = preprocess_data(data_path, categorical_features)
preprocessed_data.to_csv('examples/data/preprocessed_data.csv', index=False)

# Run AutoML
accuracy = run_automl('examples/data/preprocessed_data.csv', target_column, generations=5, population_size=20)

print(f"AutoML Accuracy: {accuracy}")
```

Replace `'target'`, `'categorical_feature1'`, and `'categorical_feature2'` with your actual column names.

### View the Exported Model

The best-performing model will be exported to `mloptimizer/models/best_model.py`.

## Advanced Usage

### Customizing the Search Space

You can customize the search space for the TPOT optimization process by modifying the configuration dictionary.

```python
from tpot import TPOTClassifier

custom_config = {
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    # Add more custom configurations here
}

tpot = TPOTClassifier(config_dict=custom_config, generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
```

### Using Custom Scoring Metrics

You can define and use custom scoring metrics for the optimization process.

```python
from sklearn.metrics import make_scorer, f1_score

custom_scorer = make_scorer(f1_score, average='weighted')

tpot = TPOTClassifier(generations=5, population_size=20, scoring=custom_scorer, verbosity=2)
tpot.fit(X_train, y_train)
```

## Running Tests

To run the unit tests, navigate to the `tests` directory and execute:

```bash
python -m unittest
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Commit your changes.
4.  Push to your branch.
5.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more information.
