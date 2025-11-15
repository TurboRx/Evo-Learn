"""Working with categorical features.

Demonstrates:
- Mixed numeric and categorical features
- Automatic categorical encoding
- High cardinality handling
"""

import pandas as pd
import numpy as np

# Generate dataset with categorical features
print("Generating dataset with categorical features...")
np.random.seed(42)
n_samples = 600

df = pd.DataFrame({
    # Numeric features
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    
    # Low cardinality categorical
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'employment': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
    
    # Medium cardinality categorical
    'city': np.random.choice([f'City_{i}' for i in range(20)], n_samples),
    
    # Ordinal categorical
    'experience': np.random.choice(['Entry', 'Mid', 'Senior', 'Expert'], n_samples),
})

# Create target based on features
df['target'] = (
    (df['age'] > 30) & 
    (df['income'] > 45000) & 
    (df['education'].isin(['Master', 'PhD']))
).astype(int)

# Add some noise
noise_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[noise_idx, 'target'] = 1 - df.loc[noise_idx, 'target']

print("\nDataset info:")
print(df.info())
print("\nCategorical columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"  {col}: {df[col].nunique()} unique values")

# Save to CSV
df.to_csv('examples/categorical_features.csv', index=False)
print("\nSaved to examples/categorical_features.csv")

print("\nNote: Evo-Learn automatically handles categorical features.")
print("Set handle_categoricals: true in config (default).")
print("\nTo train the model, run:")
print("python cli.py train --data examples/categorical_features.csv --target target --baseline")
