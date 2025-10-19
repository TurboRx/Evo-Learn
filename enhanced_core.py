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


def _compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    ...

def _compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    ...


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
    # baseline from config unless CLI explicitly passed always_baseline True
    if not always_baseline:
        always_baseline = bool(cfg.get('baseline', False))

    os.makedirs(output_dir, exist_ok=True)
    ...
