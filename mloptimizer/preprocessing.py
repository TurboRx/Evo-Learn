import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(data_path, categorical_features=None, numerical_features=None, impute_strategy='mean'):
    """
    Preprocesses the dataset.

    Args:
        data_path (str): Path to the dataset CSV file. The file should exist and be accessible.
        categorical_features (list): List of categorical feature names. Features should be present in the dataset.
        numerical_features (list): List of numerical feature names. Features should be present in the dataset.
        impute_strategy (str): Strategy to impute missing values. Options are 'mean', 'median', 'most_frequent', or 'constant'.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    # Validate input parameters
    if not os.path.isfile(data_path) or not data_path.endswith('.csv'):
        logging.error("Invalid data file path or not a CSV file.")
        raise ValueError("The data_path should be a valid path to a CSV file.")

    if categorical_features and not all(isinstance(feature, str) for feature in categorical_features):
        logging.error("categorical_features should be a list of strings.")
        raise ValueError("categorical_features should be a list of strings.")

    if numerical_features and not all(isinstance(feature, str) for feature in numerical_features):
        logging.error("numerical_features should be a list of strings.")
        raise ValueError("numerical_features should be a list of strings.")
    
    try:
        logging.info("Reading the data file...")
        data = pd.read_csv(data_path)
        logging.info("Data file read successfully.")

        # Handle missing values by imputing with the specified strategy
        imputer = SimpleImputer(strategy=impute_strategy)
        if numerical_features:
            data[numerical_features] = imputer.fit_transform(data[numerical_features])
            logging.info(f"Missing values in numerical features imputed with {impute_strategy} strategy.")

        # Encode categorical features if provided
        if categorical_features:
            for feature in categorical_features:
                if feature in data.columns:
                    le = LabelEncoder()
                    data[feature] = le.fit_transform(data[feature])
                    logging.info(f"Encoded categorical feature: {feature}")
                else:
                    logging.warning(f"Feature {feature} not found in the dataset.")

        # Scale numerical features
        if numerical_features:
            scaler = StandardScaler()
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            logging.info("Numerical features scaled.")

        return data

    except FileNotFoundError:
        logging.error("The data file does not exist.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("The data file is empty.")
        raise
    except pd.errors.ParserError:
        logging.error("Error parsing the data file.")
        raise
    except Exception as e:
        logging.error("An error occurred during preprocessing.")
        logging.exception(e)
        raise

# Example usage:
# processed_data = preprocess_data('path/to/data.csv', categorical_features=['cat1', 'cat2'], numerical_features=['num1', 'num2'], impute_strategy='median')
