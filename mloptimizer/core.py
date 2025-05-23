import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data shape after loading: {data.shape}")
        data = data.dropna()
        logging.info(f"Data shape after dropping NAs: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"Error parsing CSV file: {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def split_data(data: pd.DataFrame, target_column: str, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""
    try:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except KeyError:
        logging.error(f"Target column '{target_column}' not found in data")
        raise
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def run_automl(data_path: str, target_column: str, generations: int = 5, population_size: int = 20, test_size: float = 0.2, random_state: int = 42) -> float:
    """
    Runs automated machine learning using TPOT.

    Args:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target variable column.
        generations (int): Number of generations for TPOT.
        population_size (int): Population size for TPOT.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.

    Returns:
        float: Accuracy of the best model on the test set.
    """
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column, test_size, random_state)

    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, random_state=random_state)
    tpot.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, tpot.predict(X_test))
    tpot.export('mloptimizer/models/best_model.py')
    logging.info(f"AutoML Accuracy: {accuracy}")
    return accuracy
