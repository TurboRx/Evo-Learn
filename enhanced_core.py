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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    ...

def split_data(data: pd.DataFrame, target_column: str, test_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ...


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


def run_automl(data_path: str, target_column: str, task: str = 'classification',
              generations: int = 5, population_size: int = 20,
              test_size: float = 0.2, random_state: int = 42,
              output_dir: str = 'mloptimizer/models',
              max_time_mins: Optional[int] = None,
              max_eval_time_mins: Optional[int] = 5,
              config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run automated machine learning using TPOT with optional preprocessing and config overrides.
    """
    # Load optional config
    cfg = _load_config(config_path)

    handle_categoricals = bool(cfg.get('handle_categoricals', True))
    impute_strategy = cfg.get('impute_strategy', 'median')
    scale_numeric = bool(cfg.get('scale_numeric', True))
    output_dir = cfg.get('output_dir', output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load and split data
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column, test_size, random_state)

    # Preprocessing: build a ColumnTransformer
    preprocessor, feature_names = build_preprocessor(
        df=data, target_column=target_column,
        impute_strategy=impute_strategy,
        handle_categoricals=handle_categoricals,
        scale_numeric=scale_numeric
    )

    # Configure TPOT based on the task
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

    # Wrap TPOT inside a pipeline with preprocessing
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("tpot", tpot)
    ])

    try:
        logger.info(f"Starting {task} optimization (with preprocessing) ...")
        model.fit(X_train, y_train)
        logger.info("Optimization completed successfully")

        y_pred = model.predict(X_test)
        metrics = {}
        if task.lower() == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            if len(np.unique(y_test)) == 2 and hasattr(model.named_steps['tpot'], 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                except Exception:
                    pass
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{task}_{timestamp}"
        model_path = os.path.join(output_dir, f"{model_name}.pkl")

        # Export TPOT discovered pipeline code
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
            }
        }

        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=4)

        # Attempt feature importance if model exposes it after preprocessing
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
        logger.error(f"Error in optimization: {e}")
        raise


def load_model(model_path: str) -> Any:
    ...

def predict(model: Any, data: Union[pd.DataFrame, str], target_column: str = None) -> np.ndarray:
    ...
