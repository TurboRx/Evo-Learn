"""Core AutoML functionality using TPOT with modern Python features."""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tpot import TPOTClassifier, TPOTRegressor

from preprocessing import build_preprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some common warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def load_data(data_path: str | Path) -> pd.DataFrame:
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
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Please ensure the file path is correct and the file exists."
            )
        
        # Check file extension
        if path.suffix.lower() != '.csv':
            logger.warning(f"File does not have .csv extension: {data_path}")
            
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data: {data_path} shape={data.shape}")
        
        if data.empty:
            raise ValueError(
                f"Data file is empty: {data_path}\n"
                f"The CSV file contains no data rows."
            )
        
        if len(data.columns) < 2:
            raise ValueError(
                f"Data file must have at least 2 columns (features + target): {data_path}\n"
                f"Found only {len(data.columns)} column(s)."
            )
            
        # Check for and handle mixed dtypes that might cause issues
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert numeric strings to proper numeric types
                numeric_data = pd.to_numeric(data[col], errors='ignore')
                if not numeric_data.equals(data[col]):
                    data[col] = numeric_data
        
        # NOTE: Aggressive dropna removed. Imputation is handled in the preprocessor.
        # This prevents accidental data loss when the pipeline can handle NaNs.
        
        return data
        
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {data_path}. Error: {e}")
        raise ValueError(
            f"CSV parsing failed for {data_path}\n"
            f"Please ensure the file is a valid CSV format.\n"
            f"Error: {e}"
        )
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

def split_data(
    data: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2,
    random_state: int = 42, 
    task: str = 'classification'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:  # Need at least 2 samples per class for split
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

def _load_config(config_path: str | None) -> dict[str, Any]:
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
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
            
        with path.open('r') as f:
            config = yaml.safe_load(f) or {}
            
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in config {config_path}: {e}, using defaults")
        return {}
    except Exception as e:
        logger.warning(f"Failed to read config {config_path}: {e}, using defaults")
        return {}

def _compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
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
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

def _compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
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
    data_path: str | Path, 
    target_column: str, 
    task: str = 'classification',
    generations: int = 5, 
    population_size: int = 20,
    test_size: float = 0.2, 
    random_state: int = 42,
    output_dir: str | Path = 'models',
    max_time_mins: int | None = None,
    max_eval_time_mins: int | None = 5,
    n_jobs: int = -1,
    config_path: str | None = None,
    always_baseline: bool = False
) -> dict[str, Any]:
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
        n_jobs: Number of CPU cores to use for TPOT (-1 for all)
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
    output_dir = Path(cfg.get('output_dir', output_dir))
    n_jobs = int(cfg.get('n_jobs', n_jobs))
    
    if not always_baseline:
        always_baseline = bool(cfg.get('baseline', False))
    
    match task.lower():
        case 'classification' | 'regression':
            pass
        case _:
            raise ValueError(f"Task must be 'classification' or 'regression', got '{task}'")
    
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise
    
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column, test_size, random_state, task)
    
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
    
    def _fit_and_package(final_estimator, model_tag: str) -> dict[str, Any]:
        """Fit model with preprocessing and package results."""
        try:
            model = Pipeline(steps=[("preprocess", preprocessor), ("est", final_estimator)])
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            match task.lower():
                case 'classification':
                    y_proba = None
                    if hasattr(model.named_steps['est'], 'predict_proba') and len(np.unique(y_test)) == 2:
                        try:
                            y_proba = model.predict_proba(X_test)[:, 1]
                        except Exception as e:
                            logger.warning(f"Could not get prediction probabilities: {e}")
                    metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
                case _:
                    metrics = _compute_regression_metrics(y_test, y_pred)
            
            model_name = f"{model_tag}_{task}_{timestamp}"
            model_path = output_dir / f"{model_name}.pkl"
            
            with model_path.open('wb') as f:
                pickle.dump(model, f)
            
            result = {
                'model_name': model_name,
                'task': task,
                'model_path': str(model_path),
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
            
            metadata_path = output_dir / f"{model_name}_metadata.json"
            with metadata_path.open('w') as f:
                json.dump(result, f, indent=4)
            
            logger.info(f"Model {model_name} saved successfully with metrics: {metrics}")
            return result
            
        except Exception as e:
            logger.error(f"Error in _fit_and_package: {e}")
            raise
    
    if always_baseline:
        logger.info("Using baseline model (TPOT optimization skipped)")
        match task.lower():
            case 'classification':
                return _fit_and_package(LogisticRegression(max_iter=200, random_state=random_state), "baseline_logreg")
            case _:
                return _fit_and_package(Ridge(alpha=1.0, random_state=random_state), "baseline_ridge")
    
    try:
        logger.info(f"Starting TPOT {task} optimization...")
        
        match task.lower():
            case 'classification':
                tpot = TPOTClassifier(
                    generations=generations,
                    population_size=population_size,
                    verbosity=2,
                    random_state=random_state,
                    max_time_mins=max_time_mins,
                    max_eval_time_mins=max_eval_time_mins,
                    config_dict=None,
                    n_jobs=n_jobs
                )
            case _:
                tpot = TPOTRegressor(
                    generations=generations,
                    population_size=population_size,
                    verbosity=2,
                    random_state=random_state,
                    max_time_mins=max_time_mins,
                    max_eval_time_mins=max_eval_time_mins,
                    config_dict=None,
                    n_jobs=n_jobs
                )
        
        model = Pipeline(steps=[("preprocess", preprocessor), ("tpot", tpot)])
        model.fit(X_train, y_train)
        logger.info("TPOT optimization completed successfully")
        
        y_pred = model.predict(X_test)
        
        match task.lower():
            case 'classification':
                y_proba = None
                if hasattr(model.named_steps['tpot'], 'predict_proba') and len(np.unique(y_test)) == 2:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                    except Exception as e:
                        logger.warning(f"Could not get TPOT prediction probabilities: {e}")
                metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
            case _:
                metrics = _compute_regression_metrics(y_test, y_pred)
        
        model_name = f"tpot_{task}_{timestamp}"
        model_path = output_dir / f"{model_name}.pkl"
        
        python_script_path = None
        try:
            python_script_path = output_dir / f"{model_name}_pipeline.py"
            tpot.export(str(python_script_path))
            logger.info(f"TPOT pipeline exported to: {python_script_path}")
        except Exception as e:
            logger.warning(f"Could not export TPOT pipeline: {e}")
            python_script_path = None
        
        with model_path.open('wb') as f:
            pickle.dump(model, f)
        
        result = {
            'model_name': model_name,
            'task': task,
            'model_path': str(model_path),
            'pipeline_path': str(python_script_path) if python_script_path else None,
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
                'max_eval_time_mins': max_eval_time_mins,
                'n_jobs': n_jobs
            },
            'preprocessing': {
                'handle_categoricals': handle_categoricals,
                'impute_strategy': impute_strategy,
                'scale_numeric': scale_numeric
            },
            'model_type': 'tpot'
        }
        
        metadata_path = output_dir / f"{model_name}_metadata.json"
        with metadata_path.open('w') as f:
            json.dump(result, f, indent=4)
        
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
                    fig_path = output_dir / f"{model_name}_feature_importance.png"
                    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    result['feature_importance_plot'] = str(fig_path)
                    logger.info(f"Feature importance plot saved: {fig_path}")
        except Exception as e:
            logger.info(f"Skipping feature importance plot: {e}")
        
        logger.info(f"TPOT model completed with metrics: {metrics}")
        return result
    
    except Exception as e:
        logger.error(f"TPOT optimization failed: {e}")
        logger.info("Falling back to baseline model")
        
        match task.lower():
            case 'classification':
                return _fit_and_package(LogisticRegression(max_iter=200, random_state=random_state), "baseline_logreg")
            case _:
                return _fit_and_package(Ridge(alpha=1.0, random_state=random_state), "baseline_ridge")

def load_model(model_path: str | Path) -> Any:
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
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with path.open('rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def predict(model: Any, data: pd.DataFrame | str | Path, target_column: str | None = None) -> np.ndarray:
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
        if isinstance(data, (str, Path)):
            data = load_data(data)
        
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            logger.info(f"Removed target column '{target_column}' from prediction data")
        else:
            X = data.copy()
        
        predictions = model.predict(X)
        logger.info(f"Predictions completed: {len(predictions)} samples")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
