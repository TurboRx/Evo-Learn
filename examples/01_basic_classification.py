"""Basic classification example with Evo-Learn.

This example demonstrates:
- Loading a dataset
- Training a classification model
- Making predictions
- Evaluating results
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
print("Generating sample classification dataset...")
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Create DataFrame
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save to CSV
df.to_csv('examples/sample_classification.csv', index=False)
print("Saved to examples/sample_classification.csv")

# Train with Evo-Learn (uncomment when core module is properly structured)
"""
from core import run_automl

result = run_automl(
    data_path='examples/sample_classification.csv',
    target_column='target',
    task='classification',
    generations=3,
    population_size=15,
    baseline=True,  # Use baseline for quick demo
    output_dir='examples/models'
)

print("\nResults:")
print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
print(f"F1 Score: {result['metrics']['f1']:.4f}")
print(f"Precision: {result['metrics']['precision']:.4f}")
print(f"Recall: {result['metrics']['recall']:.4f}")
"""

print("\nTo train the model, run:")
print("python cli.py train --data examples/sample_classification.csv --target target --baseline")
