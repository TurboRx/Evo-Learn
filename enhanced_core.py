import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier, TPOTRegressor
from typing import Tuple, Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
import json
from datetime import datetime
import yaml
from preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some common warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV data with comprehensive error handling and validation.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and cleaned data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.ParserError: If CSV parsing fails
        ValueError: If data is empty or invalid
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data: {data_path} shape={data.shape}")
        
        if data.empty:
            raise ValueError(f"Data file is empty: {data_path}")
            
        # Check for and handle mixed dtypes that might cause issues with numpy 2.x
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert numeric strings to proper numeric types
                numeric_data = pd.to_numeric(data[col], errors='ignore')
                if not numeric_data.equals(data[col]):
                    data[col] = numeric_data
        
        original_shape = data.shape
        data = data.dropna()
        logger.info(f"After dropna: shape={data.shape} (removed {original_shape[0] - data.shape[0]} rows)")
        
        if data.empty:
            raise ValueError("No data remaining after removing NaN values")
            
        return data
        
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {data_path}. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def split_data(
    data: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2,
    random_state: int = 42, 
    task: str = 'classification'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets with stratification for classification.
    
    Args:
        data: Input dataframe
        target_column: Name of target column
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        task: 'classification' or 'regression'
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        KeyError: If target column not found
        ValueError: If insufficient data for splitting
    """
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in data columns: {list(data.columns)}")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    if len(data) < 4:  # Minimum for train/test split
        raise ValueError(f"Insufficient data for splitting: {len(data)} rows (minimum 4 required)")
    
    # Determine stratification strategy
    strat = None
    if task.lower() == 'classification':
        unique_classes = y.nunique()
        if unique_classes > 1 and unique_classes <= len(y) // 2:
            # Only stratify if we have reasonable class distribution
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:  # Need at least 2 samples per class
                strat = y
                logger.info(f"Using stratified split with {unique_classes} classes")
            else:
                logger.warning(f"Skipping stratification: minimum class has only {min_class_count} samples")
        else:
            logger.warning(f"Skipping stratification: {unique_classes} unique classes in {len(y)} samples")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
        
        logger.info(
            f"Data split completed - Train: X={X_train.shape}, y={y_train.shape}; "
            f"Test: X={X_test.shape}, y={y_test.shape}"
        )
        
        return X_train, X_test, y_train, y_test
        
    except ValueError as e:
        logger.error(f"Error during data splitting: {e}")
        raise

def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from YAML file with error handling.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dict containing configuration parameters
    """
    if not config_path:
        return {}
    
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
            
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in config {config_path}: {e}, using defaults")
        return {}
    except Exception as e:
        logger.warning(f"Failed to read config {config_path}: {e}, using defaults")
        return {}

def _compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dict containing computed metrics
    """
    try:
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Add ROC AUC for binary classification if probabilities available
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        # Return basic metrics in case of error  
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

def _compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict containing computed metrics
    """
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': rmse,
            'mae': float(mae),
            'r2': float(r2)
        }
        
    except Exception as e:
        logger.error(f"Error computing regression metrics: {e}")
        return {
            'mse': float('inf'),
            'rmse': float('inf'), 
            'mae': float('inf'),
            'r2': -float('inf')
        }

def run_automl(
    data_path: str, 
    target_column: str, 
    task: str = 'classification',
    generations: int = 5, 
    population_size: int = 20,
    test_size: float = 0.2, 
    random_state: int = 42,
    output_dir: str = 'mloptimizer/models',
    max_time_mins: Optional[int] = None,
    max_eval_time_mins: Optional[int] = 5,
    config_path: Optional[str] = None,
    always_baseline: bool = False
) -> Dict[str, Any]:
    """
    Run automated machine learning with TPOT or baseline models.
    
    Args:
        data_path: Path to CSV data file
        target_column: Name of target column
        task: Either 'classification' or 'regression'
        generations: Number of TPOT generations
        population_size: TPOT population size
        test_size: Fraction for test split
        random_state: Random seed
        output_dir: Directory for saving models
        max_time_mins: Max time for TPOT optimization
        max_eval_time_mins: Max time per model evaluation
        config_path: Path to YAML config file
        always_baseline: If True, skip TPOT and use baseline
        
    Returns:
        Dict containing model info, metrics, and paths
        
    Raises:
        ValueError: For invalid parameters
        Exception: For various processing errors
    """
    # Load and merge configuration
    cfg = _load_config(config_path)
    handle_categoricals = bool(cfg.get('handle_categoricals', True))
    impute_strategy = cfg.get('impute_strategy', 'median')
    scale_numeric = bool(cfg.get('scale_numeric', True))
    output_dir = cfg.get('output_dir', output_dir)
    
    if not always_baseline:
        always_baseline = bool(cfg.get('baseline', False))
    
    # Validate inputs
    if task.lower() not in ['classification', 'regression']:
        raise ValueError(f"Task must be 'classification' or 'regression', got '{task}'")
    
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise
    
    # Load and split data
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column, test_size, random_state, task)
    
    # Build preprocessing pipeline
    try:
        preprocessor, feature_names = build_preprocessor(
            df=data, target_column=target_column,
            impute_strategy=impute_strategy,
            handle_categoricals=handle_categoricals,
            scale_numeric=scale_numeric
        )
        logger.info("Preprocessing pipeline built successfully")
    except Exception as e:
        logger.error(f"Failed to build preprocessing pipeline: {e}")
        raise
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _fit_and_package(final_estimator, model_tag: str) -> Dict[str, Any]:
        """
        Fit model with preprocessing and package results.
        
        Args:
            final_estimator: The ML estimator to use
            model_tag: Tag for naming the model
            
        Returns:
            Dict containing model information and results
        """
        try:
            # Create and fit pipeline
            model = Pipeline(steps=[("preprocess", preprocessor), ("est", final_estimator)])
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Compute metrics
            if task.lower() == 'classification':
                y_proba = None
                if hasattr(model.named_steps['est'], 'predict_proba') and len(np.unique(y_test)) == 2:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                    except Exception as e:
                        logger.warning(f"Could not get prediction probabilities: {e}")
                metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
            else:
                metrics = _compute_regression_metrics(y_test, y_pred)
            
            # Save model
            model_name = f"{model_tag}_{task}_{timestamp}"
            model_path = os.path.join(output_dir, f"{model_name}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Prepare result dictionary
            result = {
                'model_name': model_name,
                'task': task,
                'model_path': model_path,
                'pipeline_path': None,
                'metrics': metrics,
                'feature_names': feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None),
                'target_column': target_column,
                'training_samples': X_train.shape[0] if hasattr(X_train, 'shape') else None,
                'testing_samples': X_test.shape[0] if hasattr(X_test, 'shape') else None,
                'timestamp': timestamp,
                'tpot_config': None,
                'preprocessing': {
                    'handle_categoricals': handle_categoricals,
                    'impute_strategy': impute_strategy,
                    'scale_numeric': scale_numeric
                },
                'model_type': model_tag
            }
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            logger.info(f"Model {model_name} saved successfully with metrics: {metrics}")
            return result
            
        except Exception as e:
            logger.error(f"Error in _fit_and_package: {e}")
            raise
    
    # Use baseline if requested
    if always_baseline:
        logger.info("Using baseline model (TPOT optimization skipped)")
        if task.lower() == 'classification':
            return _fit_and_package(LogisticRegression(max_iter=200, random_state=random_state), "baseline_logreg")
        else:
            return _fit_and_package(Ridge(alpha=1.0, random_state=random_state), "baseline_ridge")
    
    # Try TPOT optimization
    try:
        logger.info(f"Starting TPOT {task} optimization...")
        
        # Create TPOT instance
        if task.lower() == 'classification':
            tpot = TPOTClassifier(
                generations=generations,
                population_size=population_size,
                verbosity=2,
                random_state=random_state,
                max_time_mins=max_time_mins,
                max_eval_time_mins=max_eval_time_mins,
                config_dict=None,
                n_jobs=1  # Safer for stability
            )
        else:
            tpot = TPOTRegressor(
                generations=generations,
                population_size=population_size,
                verbosity=2,
                random_state=random_state,
                max_time_mins=max_time_mins,
                max_eval_time_mins=max_eval_time_mins,
                config_dict=None,
                n_jobs=1  # Safer for stability
            )
        
        # Create pipeline with preprocessing
        model = Pipeline(steps=[("preprocess", preprocessor), ("tpot", tpot)])
        
        # Fit model
        model.fit(X_train, y_train)
        logger.info("TPOT optimization completed successfully")
        
        # Make predictions and compute metrics
        y_pred = model.predict(X_test)
        
        if task.lower() == 'classification':
            y_proba = None
            if hasattr(model.named_steps['tpot'], 'predict_proba') and len(np.unique(y_test)) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                except Exception as e:
                    logger.warning(f"Could not get TPOT prediction probabilities: {e}")
            metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
        else:
            metrics = _compute_regression_metrics(y_test, y_pred)
        
        # Save model and export pipeline
        model_name = f"tpot_{task}_{timestamp}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        
        # Try to export TPOT pipeline
        python_script_path = None
        try:
            python_script_path = os.path.join(output_dir, f"{model_name}_pipeline.py")
            tpot.export(python_script_path)
            logger.info(f"TPOT pipeline exported to: {python_script_path}")
        except Exception as e:
            logger.warning(f"Could not export TPOT pipeline: {e}")
            python_script_path = None
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Prepare results
        result = {
            'model_name': model_name,
            'task': task,
            'model_path': model_path,
            'pipeline_path': python_script_path,
            'metrics': metrics,
            'feature_names': feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None),
            'target_column': target_column,
            'training_samples': X_train.shape[0] if hasattr(X_train, 'shape') else None,
            'testing_samples': X_test.shape[0] if hasattr(X_test, 'shape') else None,
            'timestamp': timestamp,
            'tpot_config': {
                'generations': generations,
                'population_size': population_size,
                'max_time_mins': max_time_mins,
                'max_eval_time_mins': max_eval_time_mins
            },
            'preprocessing': {
                'handle_categoricals': handle_categoricals,
                'impute_strategy': impute_strategy,
                'scale_numeric': scale_numeric
            },
            'model_type': 'tpot'
        }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        # Try to create feature importance plot
        try:
            final_est = model.named_steps['tpot'].fitted_pipeline_
            if hasattr(final_est, 'feature_importances_'):
                importances = final_est.feature_importances_
                if len(importances) > 0:
                    order = np.argsort(importances)
                    plt.figure(figsize=(10, 6))
                    names = (np.array(result['feature_names'])[order] 
                           if result['feature_names'] and len(result['feature_names']) == len(order)
                           else [f'Feature_{i}' for i in order])
                    plt.barh(range(len(order)), importances[order])
                    plt.yticks(range(len(order)), names)
                    plt.xlabel('Feature Importance')
                    plt.title('Feature Importance of Optimized Model')
                    plt.tight_layout()
                    fig_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
                    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    result['feature_importance_plot'] = fig_path
                    logger.info(f"Feature importance plot saved: {fig_path}")
        except Exception as e:
            logger.info(f"Skipping feature importance plot: {e}")
        
        logger.info(f"TPOT model completed with metrics: {metrics}")
        return result
    
    except Exception as e:
        logger.error(f"TPOT optimization failed: {e}")
        logger.info("Falling back to baseline model")
        
        # Fallback to baseline
        if task.lower() == 'classification':
            return _fit_and_package(LogisticRegression(max_iter=200, random_state=random_state), "baseline_logreg")
        else:
            return _fit_and_package(Ridge(alpha=1.0, random_state=random_state), "baseline_ridge")

def load_model(model_path: str) -> Any:
    """
    Load a saved model pipeline from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model pipeline
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For other loading errors
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def predict(model: Any, data: Union[pd.DataFrame, str], target_column: Optional[str] = None) -> np.ndarray:
    """
    Make predictions with a saved pipeline.
    
    Args:
        model: Loaded model pipeline
        data: Input data (DataFrame or path to CSV)
        target_column: Name of target column to exclude (if present)
        
    Returns:
        Array of predictions
        
    Raises:
        Exception: For prediction errors
    """
    try:
        # Load data if path provided
        if isinstance(data, str):
            data = load_data(data)
        
        # Remove target column if present
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            logger.info(f"Removed target column '{target_column}' from prediction data")
        else:
            X = data.copy()
        
        # Make predictions
        predictions = model.predict(X)
        logger.info(f"Predictions completed: {len(predictions)} samples")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise