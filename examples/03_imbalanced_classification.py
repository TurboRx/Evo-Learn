"""Handling imbalanced classification datasets.

Demonstrates:
- Working with imbalanced classes
- Stratified splitting
- Appropriate evaluation metrics
- Class distribution analysis
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate highly imbalanced dataset (10:1 ratio)
print("Generating imbalanced classification dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=10,
    n_redundant=2,
    n_classes=2,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    flip_y=0.01,
    random_state=42
)

# Create DataFrame
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("\nClass distribution:")
print(df['target'].value_counts())
print(f"\nClass balance: {(y.sum() / len(y)) * 100:.1f}% positive class")

# Save to CSV
df.to_csv('examples/imbalanced_classification.csv', index=False)
print("\nSaved to examples/imbalanced_classification.csv")

print("\nNote: Evo-Learn automatically uses stratified splits for classification.")
print("The evaluation will include metrics suitable for imbalanced data:")
print("- F1 Score (harmonic mean of precision and recall)")
print("- Precision (minimize false positives)")
print("- Recall (minimize false negatives)")
print("- ROC-AUC (overall discriminative ability)")

print("\nTo train the model, run:")
print("python cli.py train --data examples/imbalanced_classification.csv --target target --baseline")
