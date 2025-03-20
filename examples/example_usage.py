# examples/example_usage.py

from mloptimizer.core import run_automl
from mloptimizer.preprocessing import preprocess_data

# Example usage
data_path = 'examples/data/your_dataset.csv'
target_column = 'target'  # Replace with your actual target column name
categorical_features = ['categorical_feature1', 'categorical_feature2'] # Replace with your actual categorical features or remove if none.

# Preprocess the data
preprocessed_data = preprocess_data(data_path, categorical_features)
preprocessed_data.to_csv('examples/data/preprocessed_data.csv', index=False) #saves preprocessed data.

# Run AutoML
accuracy = run_automl('examples/data/preprocessed_data.csv', target_column, generations=5, population_size=20)

print(f"AutoML Accuracy: {accuracy}")
