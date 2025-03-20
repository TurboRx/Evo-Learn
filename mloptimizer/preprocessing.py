import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(data_path, categorical_features=None):
    """
    Preprocesses the dataset.

    Args:
        data_path (str): Path to the dataset CSV file. The file should exist and be accessible.
        categorical_features (list): List of categorical feature names. Features should be present in the dataset.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    # Validate input parameters
    if not os.path.exists(data_path):
        logging.error("The data file does not exist.")
        raise FileNotFoundError(f"The file at {data_path} was not found.")

    if categorical_features and not all(isinstance(feature, str) for feature in categorical_features):
        logging.error("categorical_features should be a list of strings.")
        raise ValueError("categorical_features should be a list of strings.")

    try:
        logging.info("Reading the data file...")
        data = pd.read_csv(data_path)
        logging.info("Data file read successfully.")

        # Handle missing values by dropping rows with NaNs (consider imputing in future improvements)
        data = data.dropna()
        logging.info("Missing values dropped.")

        # Encode categorical features if provided
        if categorical_features:
            for feature in categorical_features:
                if feature in data.columns:
                    le = LabelEncoder()
                    data[feature] = le.fit_transform(data[feature])
                    logging.info(f"Encoded categorical feature: {feature}")
                else:
                    logging.warning(f"Feature {feature} not found in the dataset.")

        return data

    except Exception as e:
        logging.error("An error occurred during preprocessing.")
        logging.exception(e)
        raise
