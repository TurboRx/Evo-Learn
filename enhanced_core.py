import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    
    Args:
        data_path (str): Path to the CSV file containing the dataset.
        
    Returns:
        pd.DataFrame: Loaded and preprocessed DataFrame.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        pd.errors.ParserError: If there's an error parsing the CSV file.
    """
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data shape after loading: {data.shape}")
        data = data.dropna()
        logger.info(f"Data shape after dropping NAs: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV file: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def split_data(data: pd.DataFrame, target_column: str, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target variable column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test.
        
    Raises:
        KeyError: If the target column is not in the data.
    """
    try:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except KeyError:
        logger.error(f"Target column '{target_column}' not found in data")
        raise
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def run_automl(data_path: str, target_column: str, task: str = 'classification',
              generations: int = 5, population_size: int = 20, 
              test_size: float = 0.2, random_state: int = 42,
              output_dir: str = 'mloptimizer/models',
              max_time_mins: Optional[int] = None,
              max_eval_time_mins: Optional[int] = 5) -> Dict[str, Any]:
    """
    Run automated machine learning using TPOT for classification or regression tasks.
    
    Args:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target variable column.
        task (str): Type of ML task - 'classification' or 'regression'.
        generations (int): Number of generations for TPOT.
        population_size (int): Population size for TPOT.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.
        output_dir (str): Directory to save the model and metadata.
        max_time_mins (int, optional): Maximum time in minutes for optimization.
        max_eval_time_mins (int, optional): Maximum time in minutes for evaluating a single pipeline.
        
    Returns:
        Dict[str, Any]: Dictionary with model information and evaluation metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column, test_size, random_state)
    
    # Configure TPOT based on the task
    if task.lower() == 'classification':
        tpot = TPOTClassifier(
            generations=generations,
            population_size=population_size,
            verbosity=2,
            random_state=random_state,
            max_time_mins=max_time_mins,
            max_eval_time_mins=max_eval_time_mins
        )
    elif task.lower() == 'regression':
        tpot = TPOTRegressor(
            generations=generations,
            population_size=population_size,
            verbosity=2,
            random_state=random_state,
            max_time_mins=max_time_mins,
            max_eval_time_mins=max_eval_time_mins
        )
    else:
        raise ValueError(f"Task must be 'classification' or 'regression', got '{task}'")
    
    # Run TPOT optimization
    try:
        logger.info(f"Starting {task} model optimization with TPOT...")
        tpot.fit(X_train, y_train)
        logger.info("TPOT optimization completed successfully")
        
        # Make predictions and calculate performance metrics
        y_pred = tpot.predict(X_test)
        
        # Calculate metrics based on task
        metrics = {}
        if task.lower() == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Calculate ROC AUC if binary classification
            if len(np.unique(y_test)) == 2 and hasattr(tpot, 'predict_proba'):
                y_proba = tpot.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                
            logger.info(f"Classification Metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            
            logger.info(f"Regression Metrics: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        # Save the model and metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{task}_{timestamp}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        
        # Export the optimized pipeline as Python code
        python_script_path = os.path.join(output_dir, f"{model_name}_pipeline.py")
        tpot.export(python_script_path)
        logger.info(f"Exported optimized pipeline to {python_script_path}")
        
        # Save the actual model
        with open(model_path, 'wb') as f:
            pickle.dump(tpot.fitted_pipeline_, f)
        logger.info(f"Saved model to {model_path}")
        
        # Prepare the result information
        result = {
            'model_name': model_name,
            'task': task,
            'model_path': model_path,
            'pipeline_path': python_script_path,
            'metrics': metrics,
            'feature_names': X_train.columns.tolist(),
            'target_column': target_column,
            'training_samples': X_train.shape[0],
            'testing_samples': X_test.shape[0],
            'timestamp': timestamp,
            'tpot_config': {
                'generations': generations,
                'population_size': population_size,
                'max_time_mins': max_time_mins,
                'max_eval_time_mins': max_eval_time_mins
            }
        }
        
        # Save metadata to JSON
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved model metadata to {metadata_path}")
        
        # Visualize feature importance if available
        if hasattr(tpot.fitted_pipeline_, 'feature_importances_'):
            try:
                plt.figure(figsize=(10, 6))
                feature_importance = tpot.fitted_pipeline_.feature_importances_
                sorted_idx = np.argsort(feature_importance)
                plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                plt.yticks(range(len(sorted_idx)), np.array(X_train.columns)[sorted_idx])
                plt.xlabel('Feature Importance')
                plt.title('Feature Importance of Optimized Model')
                plt.tight_layout()
                
                # Save the figure
                fig_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
                plt.savefig(fig_path)
                plt.close()
                logger.info(f"Saved feature importance plot to {fig_path}")
                
                # Add to result
                result['feature_importance_plot'] = fig_path
            except Exception as e:
                logger.warning(f"Could not generate feature importance plot: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in TPOT optimization: {e}")
        raise

def load_model(model_path: str) -> Any:
    """
    Load a saved model from disk.
    
    Args:
        model_path (str): Path to the saved model file.
        
    Returns:
        Any: The loaded model.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict(model: Any, data: Union[pd.DataFrame, str], target_column: str = None) -> np.ndarray:
    """
    Make predictions using a loaded model.
    
    Args:
        model: The trained model.
        data (Union[pd.DataFrame, str]): DataFrame or path to CSV with input data.
        target_column (str, optional): Target column name if in data. Default is None.
        
    Returns:
        np.ndarray: Model predictions.
    """
    # Load the data if it's a string (file path)
    if isinstance(data, str):
        data = load_data(data)
        
    # Remove target column if present
    if target_column and target_column in data.columns:
        X = data.drop(target_column, axis=1)
    else:
        X = data
        
    # Make predictions
    try:
        predictions = model.predict(X)
        logger.info(f"Made predictions on {len(predictions)} samples")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise