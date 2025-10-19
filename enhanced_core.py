import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier, TPOTRegressor
from typing import Tuple, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import json
from datetime import datetime
import yaml
from preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV and drop NA with logging and error handling."""
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data: {data_path} shape={data.shape}")
        data = data.dropna()
        logger.info(f"After dropna: shape={data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Failed to parse CSV: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def split_data(data: pd.DataFrame, target_column: str, test_size: float = 0.2,
               random_state: int = 42, task: str = 'classification') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test; stratify for classification when possible."""
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found")
    X = data.drop(columns=[target_column])
    y = data[target_column]
    strat = None
    if task.lower() == 'classification' and y.nunique() > 1:
        strat = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
    logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}; Test shapes: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test

def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.warning(f"Failed to read config {config_path}: {e}")
        return {}

def _compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception:
            pass
    return metrics

def _compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def run_automl(data_path: str, target_column: str, task: str = 'classification',
              generations: int = 5, population_size: int = 20,
              test_size: float = 0.2, random_state: int = 42,
              output_dir: str = 'mloptimizer/models',
              max_time_mins: Optional[int] = None,
              max_eval_time_mins: Optional[int] = 5,
              config_path: Optional[str] = None,
              always_baseline: bool = False) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    handle_categoricals = bool(cfg.get('handle_categoricals', True))
    impute_strategy = cfg.get('impute_strategy', 'median')
    scale_numeric = bool(cfg.get('scale_numeric', True))
    output_dir = cfg.get('output_dir', output_dir)
    if not always_baseline:
        always_baseline = bool(cfg.get('baseline', False))

    os.makedirs(output_dir, exist_ok=True)

    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column, test_size, random_state, task)

    preprocessor, feature_names = build_preprocessor(
        df=data, target_column=target_column,
        impute_strategy=impute_strategy,
        handle_categoricals=handle_categoricals,
        scale_numeric=scale_numeric
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _fit_and_package(final_estimator, model_tag: str) -> Dict[str, Any]:
        model = Pipeline(steps=[("preprocess", preprocessor), ("est", final_estimator)])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task.lower() == 'classification':
            y_proba = None
            if hasattr(model.named_steps['est'], 'predict_proba') and len(np.unique(y_test)) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                except Exception:
                    y_proba = None
            metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
        else:
            metrics = _compute_regression_metrics(y_test, y_pred)
        model_name = f"{model_tag}_{task}_{timestamp}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        result = {
            'model_name': model_name,
            'task': task,
            'model_path': model_path,
            'pipeline_path': None,
            'metrics': metrics,
            'feature_names': feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None),
            'target_column': target_column,
            'training_samples': getattr(X_train, 'shape', [None])[0],
            'testing_samples': getattr(X_test, 'shape', [None])[0],
            'timestamp': timestamp,
            'tpot_config': None,
            'preprocessing': {
                'handle_categoricals': handle_categoricals,
                'impute_strategy': impute_strategy,
                'scale_numeric': scale_numeric
            },
            'model_type': model_tag
        }
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=4)
        return result

    if always_baseline:
        logger.info("always_baseline=True: Skipping TPOT and training baseline.")
        if task.lower() == 'classification':
            return _fit_and_package(LogisticRegression(max_iter=200), "baseline_logreg")
        else:
            return _fit_and_package(Ridge(alpha=1.0, random_state=random_state), "baseline_ridge")

    try:
        if task.lower() == 'classification':
            tpot = TPOTClassifier(
                generations=generations,
                population_size=population_size,
                verbosity=2,
                random_state=random_state,
                max_time_mins=max_time_mins,
                max_eval_time_mins=max_eval_time_mins,
                config_dict=None
            )
        elif task.lower() == 'regression':
            tpot = TPOTRegressor(
                generations=generations,
                population_size=population_size,
                verbosity=2,
                random_state=random_state,
                max_time_mins=max_time_mins,
                max_eval_time_mins=max_eval_time_mins,
                config_dict=None
            )
        else:
            raise ValueError(f"Task must be 'classification' or 'regression', got '{task}'")

        model = Pipeline(steps=[("preprocess", preprocessor), ("tpot", tpot)])
        logger.info(f"Starting {task} optimization (with preprocessing) ...")
        model.fit(X_train, y_train)
        logger.info("TPOT optimization completed successfully")

        y_pred = model.predict(X_test)
        if task.lower() == 'classification':
            y_proba = None
            if hasattr(model.named_steps['tpot'], 'predict_proba') and len(np.unique(y_test)) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                except Exception:
                    pass
            metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
        else:
            metrics = _compute_regression_metrics(y_test, y_pred)

        model_name = f"tpot_{task}_{timestamp}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        python_script_path = None
        try:
            python_script_path = os.path.join(output_dir, f"{model_name}_pipeline.py")
            tpot.export(python_script_path)
        except Exception as e:
            python_script_path = None
            logger.warning(f"Could not export TPOT pipeline: {e}")

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        result = {
            'model_name': model_name,
            'task': task,
            'model_path': model_path,
            'pipeline_path': python_script_path,
            'metrics': metrics,
            'feature_names': feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None),
            'target_column': target_column,
            'training_samples': getattr(X_train, 'shape', [None])[0],
            'testing_samples': getattr(X_test, 'shape', [None])[0],
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
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=4)
        try:
            final_est = model.named_steps['tpot'].fitted_pipeline_
            if hasattr(final_est, 'feature_importances_'):
                importances = final_est.feature_importances_
                order = np.argsort(importances)
                plt.figure(figsize=(10, 6))
                names = np.array(result['feature_names'])[order] if result['feature_names'] else range(len(order))
                plt.barh(range(len(order)), importances[order])
                plt.yticks(range(len(order)), names)
                plt.xlabel('Feature Importance')
                plt.title('Feature Importance of Optimized Model')
                plt.tight_layout()
                fig_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
                plt.savefig(fig_path)
                plt.close()
                result['feature_importance_plot'] = fig_path
        except Exception as e:
            logger.info(f"Skipping feature importance plot: {e}")

        return result

    except Exception as e:
        logger.error(f"TPOT optimization failed, falling back to baseline. Reason: {e}")
        if task.lower() == 'classification':
            return _fit_and_package(LogisticRegression(max_iter=200), "baseline_logreg")
        else:
            return _fit_and_package(Ridge(alpha=1.0, random_state=random_state), "baseline_ridge")

def load_model(model_path: str) -> Any:
    """Load a saved model pipeline from disk."""
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
    """Make predictions with a saved pipeline; drops target if present."""
    if isinstance(data, str):
        data = load_data(data)
    X = data.drop(columns=[target_column]) if target_column and target_column in data.columns else data
    try:
        preds = model.predict(X)
        logger.info(f"Predictions made: n={len(preds)}")
        return preds
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
