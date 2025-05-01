import time
import logging
import numpy as np
from typing import Callable, Any, Dict, List, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def timer(unit: str = "seconds", log_level: int = logging.INFO) -> Callable:
    """
    Advanced decorator to measure the execution time of a function.

    Args:
        unit (str): Unit of time to display ('seconds' or 'milliseconds'). Default is 'seconds'.
        log_level (int): Logging level for the timing information. Default is logging.INFO.

    Returns:
        Callable: A decorator for measuring execution time.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000 if unit == "milliseconds" else (end_time - start_time)
                unit_label = "ms" if unit == "milliseconds" else "s"

                logger.log(log_level, f"Function '{func.__name__}' executed in {elapsed_time:.2f} {unit_label}")
                
                # Only log arguments in debug mode to avoid exposing sensitive data in production
                if log_level == logging.DEBUG:
                    logger.debug(f"Arguments: args={args}, kwargs={kwargs}")

                return result
            except Exception as e:
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000 if unit == "milliseconds" else (end_time - start_time)
                unit_label = "ms" if unit == "milliseconds" else "s"

                logger.error(f"Function '{func.__name__}' failed after {elapsed_time:.2f} {unit_label}")
                logger.exception(e)
                raise
        return wrapper
    return decorator

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate various classification metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_proba (np.ndarray, optional): Predicted probabilities for positive class. Defaults to None.

    Returns:
        Dict[str, float]: Dictionary containing various metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Calculate ROC AUC if probabilities are provided
    if y_proba is not None:
        # Check if binary classification (2 classes)
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics

def validate_input(data: Any, expected_type: Any, var_name: str) -> None:
    """
    Validate input parameters for type and basic properties.

    Args:
        data (Any): The data to validate.
        expected_type (Any): The expected type of the data.
        var_name (str): Name of the variable (for error messages).

    Raises:
        TypeError: If the data is not of the expected type.
        ValueError: If the data does not meet validation criteria.
    """
    if not isinstance(data, expected_type):
        raise TypeError(f"{var_name} must be of type {expected_type.__name__}, got {type(data).__name__}")
    
    if expected_type in (str, list, dict, np.ndarray) and not data:
        raise ValueError(f"{var_name} cannot be empty")
    
    if expected_type in (int, float) and data < 0:
        raise ValueError(f"{var_name} must be non-negative")

def save_model_metadata(model_info: Dict[str, Any], path: str) -> None:
    """
    Save model metadata to a file.

    Args:
        model_info (Dict[str, Any]): Dictionary containing model metadata.
        path (str): Path to save the metadata.
    """
    import json
    import os
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        cleaned_info = convert_numpy(model_info)
        
        with open(path, 'w') as f:
            json.dump(cleaned_info, f, indent=4)
            
        logger.info(f"Model metadata saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model metadata: {e}")
        raise

def cross_validate_model(model, X, y, cv=5, random_state=42) -> Dict[str, List[float]]:
    """
    Perform cross-validation on a model and return detailed metrics.

    Args:
        model: The model to validate.
        X: Feature data.
        y: Target data.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        Dict[str, List[float]]: Dictionary of metrics with list of values for each fold.
    """
    from sklearn.model_selection import StratifiedKFold
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    if hasattr(model, 'predict_proba'):
        metrics['roc_auc'] = []
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        fold_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if available
        if 'roc_auc' in metrics and len(np.unique(y)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            fold_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        # Add metrics to results
        for metric, value in fold_metrics.items():
            metrics[metric].append(value)
    
    return metrics