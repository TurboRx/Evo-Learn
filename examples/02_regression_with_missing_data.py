"""Regression example with missing data handling.

Demonstrates:
- Handling datasets with missing values
- Regression task
- Different imputation strategies
- Custom configuration
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Generate regression data
print("Generating regression dataset with missing values...")
X, y = make_regression(
    n_samples=400,
    n_features=8,
    n_informative=6,
    noise=10,
    random_state=42
)

# Create DataFrame
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Introduce missing values (10% missing in random cells)
np.random.seed(42)
for col in feature_names[:5]:  # Add missing values to first 5 features
    missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_idx, col] = np.nan

print(f"Missing values per column:")
print(df.isnull().sum())

# Save to CSV
df.to_csv('examples/sample_regression_missing.csv', index=False)
print("\nSaved to examples/sample_regression_missing.csv")

# Create custom config for this example
config = """
default_task: regression
random_state: 42
test_size: 0.25
generations: 3
population_size: 15
handle_categoricals: true
impute_strategy: median  # Try 'mean' or 'most_frequent' too
scale_numeric: true
baseline: true
output_dir: examples/models
"""

with open('examples/regression_config.yaml', 'w') as f:
    f.write(config)

print("\nCreated config: examples/regression_config.yaml")
print("\nTo train the model, run:")
print("python cli.py train --data examples/sample_regression_missing.csv --target target --config examples/regression_config.yaml")
