# mloptimizer/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data_path, categorical_features=None):
    """
    Preprocesses the dataset.

    Args:
        data_path (str): Path to the dataset CSV file.
        categorical_features (list): List of categorical feature names.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    data = pd.read_csv(data_path)
    data = data.dropna()

    if categorical_features:
        for feature in categorical_features:
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature])

    return data
