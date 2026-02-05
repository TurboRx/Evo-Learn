"""Core AutoML functionality using TPOT with modern Python features."""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
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
    format="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress some warnings for cleaner output (can be configured via environment)
if os.getenv("EVO_LEARN_SHOW_WARNINGS", "").lower() != "true":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def load_data(data_path: str | Path, max_size_mb: int = 500) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with pre-checks and light type coercion.
    
    Performs existence and file-size validation, reads the CSV, ensures the file contains at least two columns, and attempts to convert object-typed columns that contain numeric strings to numeric dtype.
    
    Parameters:
        data_path (str | Path): Path to the CSV file to load.
        max_size_mb (int): Maximum allowed file size in megabytes; a ValueError is raised if the file exceeds this size.
    
    Returns:
        pd.DataFrame: The loaded DataFrame with attempted numeric coercions applied to object columns.
    
    Raises:
        FileNotFoundError: If the file at data_path does not exist.
        ValueError: If the file is larger than max_size_mb, the CSV is empty, has fewer than two columns, or CSV parsing fails (parsing errors are reported as ValueError).
        Exception: Re-raises unexpected exceptions encountered during loading.
    """
    try:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Please ensure the file path is correct and the file exists."
            )

        # Check file size to prevent OOM
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)\n"
                f"Consider using a smaller dataset or increasing max_size_mb parameter."
            )
        logger.info(f"Loading file: {data_path} ({file_size_mb:.2f}MB)")

        # Check file extension
        if path.suffix.lower() != ".csv":
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
            if data[col].dtype == "object":
                # Try to convert numeric strings to proper numeric types
                numeric_data = pd.to_numeric(data[col], errors="ignore")
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
    task: str = "classification",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataframe into train and test sets, applying stratified sampling for classification when feasible.
    
    Parameters:
        data (pd.DataFrame): Input dataframe containing features and the target column.
        target_column (str): Name of the target column to separate from features.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random operations to ensure reproducibility.
        task (str): Either "classification" or "regression"; determines whether stratification is considered.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: (X_train, X_test, y_train, y_test).
    
    Raises:
        KeyError: If the specified target column is not present in the dataframe.
        ValueError: If the dataset is too small to perform a train/test split or if sklearn raises a splitting error.
    """
    if target_column not in data.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in data columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if len(data) < 4:  # Minimum for train/test split
        raise ValueError(
            f"Insufficient data for splitting: {len(data)} rows (minimum 4 required)"
        )

    # Determine stratification strategy
    strat = None
    if task.lower() == "classification":
        unique_classes = y.nunique()
        if unique_classes > 1 and unique_classes <= len(y) // 2:
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:  # Need at least 2 samples per class for split
                strat = y
                logger.info(f"Using stratified split with {unique_classes} classes")
            else:
                logger.warning(
                    f"Skipping stratification: minimum class has only {min_class_count} samples"
                )
        else:
            logger.warning(
                f"Skipping stratification: {unique_classes} unique classes in {len(y)} samples"
            )

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
    Load YAML configuration from a file and return its contents as a dictionary.
    
    If `config_path` is None, the file does not exist, or the file contains invalid YAML,
    an empty dictionary is returned and a warning is logged.
    
    Parameters:
        config_path (str | None): Path to a YAML config file, or None to use defaults.
    
    Returns:
        dict[str, Any]: Parsed configuration mapping, or an empty dict when no valid config is available.
    """
    if not config_path:
        return {}

    try:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        with path.open("r") as f:
            config = yaml.safe_load(f) or {}

        logger.info(f"Loaded configuration from: {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in config {config_path}: {e}, using defaults")
        return {}
    except Exception as e:
        logger.warning(f"Failed to read config {config_path}: {e}, using defaults")
        return {}


def validate_data_for_training(
    data: pd.DataFrame, target_column: str, task: str
) -> None:
    """
    Validate a dataset for training by checking the target column and flagging common data issues.
    
    Checks performed:
    - If the target column exists, ensures it contains no NaN values and (for classification) at least two classes.
    - For classification, computes class distribution and logs a warning for severe imbalance.
    - Detects columns with all NaN values and constant (single-valued) features and logs warnings that they will likely be dropped during preprocessing.
    
    Parameters:
        data (pd.DataFrame): Input dataset containing features and the target column.
        target_column (str): Name of the target column to validate.
        task (str): Task type; must be either 'classification' or 'regression'.
    
    Raises:
        ValueError: If the target column contains NaN values, or if a classification target has fewer than two unique classes.
    """
    # Check for NaN in target column
    if target_column in data.columns:
        nan_count = data[target_column].isna().sum()
        if nan_count > 0:
            raise ValueError(
                f"Target column '{target_column}' contains {nan_count} NaN values. "
                f"Please remove or impute these values before training."
            )

        # Check for single class in classification
        if task.lower() == "classification":
            unique_classes = data[target_column].nunique()
            if unique_classes < 2:
                raise ValueError(
                    f"Classification requires at least 2 classes, but target column '{target_column}' "
                    f"has only {unique_classes} unique value(s): {data[target_column].unique()}"
                )

            # Warn about class imbalance
            class_counts = data[target_column].value_counts()
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            imbalance_ratio = (
                max_class_count / min_class_count
                if min_class_count > 0
                else float("inf")
            )

            if imbalance_ratio > 10:
                logger.warning(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    f"Consider using techniques like SMOTE or class weights."
                )
                logger.warning(f"Class distribution: {class_counts.to_dict()}")

    # Check for all-NaN features
    all_nan_cols = data.columns[data.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(
            f"Found {len(all_nan_cols)} columns with all NaN values: {all_nan_cols[:5]}. "
            f"These will be dropped during preprocessing."
        )

    # Check for constant features
    constant_cols = []
    for col in data.columns:
        if col != target_column and data[col].nunique() == 1:
            constant_cols.append(col)

    if constant_cols:
        logger.warning(
            f"Found {len(constant_cols)} constant features: {constant_cols[:5]}. "
            f"These provide no information and may be dropped."
        )

    logger.info(
        f"Data validation passed for {len(data)} samples with {len(data.columns)} features"
    )


def _compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute standard classification metrics from true labels and predictions.
    
    Parameters:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_proba: Predicted probabilities. For binary tasks this may be a 1D array of probabilities for the positive class or a 2D array of class probability estimates; when provided and the problem is binary, ROC AUC will be attempted.
    
    Returns:
        dict[str, float]: Mapping of metric names to float values. Always includes `accuracy`, `precision`, `recall`, and `f1_score`. Includes `roc_auc` when `y_proba` is provided and ROC AUC can be computed for a binary classification problem.
    """
    try:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }

        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not compute ROC AUC: {e}")

        return metrics

    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }


def _compute_regression_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """
    Compute standard regression evaluation metrics for true and predicted values.
    
    Returns:
        A dict with:
            - `mse`: Mean squared error as a float.
            - `rmse`: Root mean squared error as a float.
            - `mae`: Mean absolute error as a float.
            - `r2`: R² (coefficient of determination) as a float.
        On internal failure, returns `mse`, `rmse`, and `mae` as `inf` and `r2` as `-inf`.
    """
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2)}

    except Exception as e:
        logger.error(f"Error computing regression metrics: {e}")
        return {
            "mse": float("inf"),
            "rmse": float("inf"),
            "mae": float("inf"),
            "r2": -float("inf"),
        }


def run_automl(
    data_path: str | Path,
    target_column: str,
    task: str = "classification",
    generations: int = 5,
    population_size: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str | Path = "models",
    max_time_mins: int | None = None,
    max_eval_time_mins: int | None = 5,
    n_jobs: int = -1,
    config_path: str | None = None,
    always_baseline: bool = False,
) -> dict[str, Any]:
    """
    Run an automated ML workflow on a CSV dataset using TPOT or a baseline model.
    
    Loads and validates data, builds a preprocessing pipeline, splits data, trains either a TPOT-optimized pipeline (when available) or a baseline estimator, evaluates on a test set, saves the fitted pipeline and metadata, and returns a result summary.
    
    Parameters:
        data_path (str | Path): Path to the input CSV file.
        target_column (str): Name of the target column in the dataset.
        task (str): "classification" or "regression".
        generations (int): Number of TPOT generations to run.
        population_size (int): TPOT population size.
        test_size (float): Fraction of data reserved for testing (must be >0 and <1).
        random_state (int): Seed for reproducibility.
        output_dir (str | Path): Directory where models, artifacts, and metadata are saved.
        max_time_mins (int | None): Optional overall time limit (minutes) for TPOT optimization.
        max_eval_time_mins (int | None): Optional per-model evaluation time limit (minutes) for TPOT.
        n_jobs (int): Number of CPU cores to use by TPOT (-1 uses all available).
        config_path (str | None): Optional path to a YAML configuration file to override defaults.
        always_baseline (bool): If True, skip TPOT and train only the baseline model.
    
    Returns:
        dict: Result dictionary containing model metadata and evaluation, including keys such as:
            - "model_name": generated model identifier
            - "task": task type ("classification" or "regression")
            - "model_path": filesystem path to the serialized model
            - "pipeline_path": path to exported TPOT Python pipeline (if available)
            - "metrics": evaluation metrics (classification or regression)
            - "feature_names": list of feature names used
            - "target_column": provided target column name
            - "training_samples", "testing_samples": sample counts
            - "timestamp": run timestamp
            - "tpot_config": TPOT run configuration (when applicable)
            - "preprocessing": preprocessing settings applied
            - "model_type": "tpot" or baseline tag
            - optionally "feature_importance_plot": path to a saved feature importance image
    
    Raises:
        ValueError: For invalid parameter values (e.g., unsupported task or invalid test_size).
        Exception: For other failures during directory creation, data loading, preprocessing, training, or serialization.
    """
    # Load and merge configuration
    cfg = _load_config(config_path)
    handle_categoricals = bool(cfg.get("handle_categoricals", True))
    impute_strategy = cfg.get("impute_strategy", "median")
    scale_numeric = bool(cfg.get("scale_numeric", True))
    output_dir = Path(cfg.get("output_dir", output_dir))
    n_jobs = int(cfg.get("n_jobs", n_jobs))

    if not always_baseline:
        always_baseline = bool(cfg.get("baseline", False))

    match task.lower():
        case "classification" | "regression":
            pass
        case _:
            raise ValueError(
                f"Task must be 'classification' or 'regression', got '{task}'"
            )

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    data = load_data(data_path)

    # Validate data before training
    validate_data_for_training(data, target_column, task)

    X_train, X_test, y_train, y_test = split_data(
        data, target_column, test_size, random_state, task
    )

    try:
        preprocessor, feature_names = build_preprocessor(
            df=data,
            target_column=target_column,
            impute_strategy=impute_strategy,
            handle_categoricals=handle_categoricals,
            scale_numeric=scale_numeric,
        )
        logger.info("Preprocessing pipeline built successfully")
    except Exception as e:
        logger.error(f"Failed to build preprocessing pipeline: {e}")
        raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _fit_and_package(final_estimator: Any, model_tag: str) -> dict[str, Any]:
        """
        Fit a preprocessing-wrapped estimator on the training set, evaluate it on the test set, serialize the fitted pipeline, and write accompanying metadata to disk.
        
        Returns:
            result (dict[str, Any]): Metadata and artefacts for the fitted model, including:
                - "model_name": generated model identifier
                - "task": task name (e.g., "classification" or "regression")
                - "model_path": filesystem path to the serialized pipeline file
                - "pipeline_path": optional path to exported pipeline script (or None)
                - "metrics": evaluation metrics computed on the test set
                - "feature_names": list of feature names used (or None)
                - "target_column": name of the target column
                - "training_samples": number of training samples (or None)
                - "testing_samples": number of testing samples (or None)
                - "timestamp": timestamp string used in naming
                - "tpot_config": TPOT configuration payload if applicable (or None)
                - "preprocessing": dict describing preprocessing settings
                - "model_type": tag identifying the model type used
        
        Side effects:
            - Writes a serialized model file to disk.
            - Writes a JSON metadata file alongside the model.
        """
        try:
            model = Pipeline(
                steps=[("preprocess", preprocessor), ("est", final_estimator)]
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            match task.lower():
                case "classification":
                    y_proba = None
                    if (
                        hasattr(model.named_steps["est"], "predict_proba")
                        and len(np.unique(y_test)) == 2
                    ):
                        try:
                            y_proba = model.predict_proba(X_test)[:, 1]
                        except Exception as e:
                            logger.warning(
                                f"Could not get prediction probabilities: {e}"
                            )
                    metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
                case _:
                    metrics = _compute_regression_metrics(y_test, y_pred)

            model_name = f"{model_tag}_{task}_{timestamp}"
            model_path = output_dir / f"{model_name}.pkl"

            # Use joblib for safer model serialization
            joblib.dump(model, model_path, compress=3)

            result = {
                "model_name": model_name,
                "task": task,
                "model_path": str(model_path),
                "pipeline_path": None,
                "metrics": metrics,
                "feature_names": feature_names
                or (X_train.columns.tolist() if hasattr(X_train, "columns") else None),
                "target_column": target_column,
                "training_samples": (
                    X_train.shape[0] if hasattr(X_train, "shape") else None
                ),
                "testing_samples": (
                    X_test.shape[0] if hasattr(X_test, "shape") else None
                ),
                "timestamp": timestamp,
                "tpot_config": None,
                "preprocessing": {
                    "handle_categoricals": handle_categoricals,
                    "impute_strategy": impute_strategy,
                    "scale_numeric": scale_numeric,
                },
                "model_type": model_tag,
            }

            metadata_path = output_dir / f"{model_name}_metadata.json"
            with metadata_path.open("w") as f:
                json.dump(result, f, indent=4)

            logger.info(
                f"Model {model_name} saved successfully with metrics: {metrics}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in _fit_and_package: {e}")
            raise

    if always_baseline:
        logger.info("Using baseline model (TPOT optimization skipped)")
        match task.lower():
            case "classification":
                return _fit_and_package(
                    LogisticRegression(max_iter=200, random_state=random_state),
                    "baseline_logreg",
                )
            case _:
                return _fit_and_package(
                    Ridge(alpha=1.0, random_state=random_state), "baseline_ridge"
                )

    try:
        logger.info(f"Starting TPOT {task} optimization...")

        match task.lower():
            case "classification":
                tpot = TPOTClassifier(
                    generations=generations,
                    population_size=population_size,
                    verbosity=2,
                    random_state=random_state,
                    max_time_mins=max_time_mins,
                    max_eval_time_mins=max_eval_time_mins,
                    config_dict=None,
                    n_jobs=n_jobs,
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
                    n_jobs=n_jobs,
                )

        model = Pipeline(steps=[("preprocess", preprocessor), ("tpot", tpot)])
        model.fit(X_train, y_train)
        logger.info("TPOT optimization completed successfully")

        y_pred = model.predict(X_test)

        match task.lower():
            case "classification":
                y_proba = None
                if (
                    hasattr(model.named_steps["tpot"], "predict_proba")
                    and len(np.unique(y_test)) == 2
                ):
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                    except Exception as e:
                        logger.warning(
                            f"Could not get TPOT prediction probabilities: {e}"
                        )
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

        # Use joblib for safer model serialization
        joblib.dump(model, model_path, compress=3)

        result = {
            "model_name": model_name,
            "task": task,
            "model_path": str(model_path),
            "pipeline_path": str(python_script_path) if python_script_path else None,
            "metrics": metrics,
            "feature_names": feature_names
            or (X_train.columns.tolist() if hasattr(X_train, "columns") else None),
            "target_column": target_column,
            "training_samples": X_train.shape[0] if hasattr(X_train, "shape") else None,
            "testing_samples": X_test.shape[0] if hasattr(X_test, "shape") else None,
            "timestamp": timestamp,
            "tpot_config": {
                "generations": generations,
                "population_size": population_size,
                "max_time_mins": max_time_mins,
                "max_eval_time_mins": max_eval_time_mins,
                "n_jobs": n_jobs,
            },
            "preprocessing": {
                "handle_categoricals": handle_categoricals,
                "impute_strategy": impute_strategy,
                "scale_numeric": scale_numeric,
            },
            "model_type": "tpot",
        }

        metadata_path = output_dir / f"{model_name}_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(result, f, indent=4)

        try:
            final_est = model.named_steps["tpot"].fitted_pipeline_
            if hasattr(final_est, "feature_importances_"):
                importances = final_est.feature_importances_
                if len(importances) > 0:
                    order = np.argsort(importances)
                    plt.figure(figsize=(10, 6))
                    names = (
                        np.array(result["feature_names"])[order]
                        if result["feature_names"]
                        and len(result["feature_names"]) == len(order)
                        else [f"Feature_{i}" for i in order]
                    )
                    plt.barh(range(len(order)), importances[order])
                    plt.yticks(range(len(order)), names)
                    plt.xlabel("Feature Importance")
                    plt.title("Feature Importance of Optimized Model")
                    plt.tight_layout()
                    fig_path = output_dir / f"{model_name}_feature_importance.png"
                    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    result["feature_importance_plot"] = str(fig_path)
                    logger.info(f"Feature importance plot saved: {fig_path}")
        except Exception as e:
            logger.info(f"Skipping feature importance plot: {e}")

        logger.info(f"TPOT model completed with metrics: {metrics}")
        return result

    except Exception as e:
        logger.error(f"TPOT optimization failed: {e}")
        logger.info("Falling back to baseline model")

        match task.lower():
            case "classification":
                return _fit_and_package(
                    LogisticRegression(max_iter=200, random_state=random_state),
                    "baseline_logreg",
                )
            case _:
                return _fit_and_package(
                    Ridge(alpha=1.0, random_state=random_state), "baseline_ridge"
                )


def load_model(model_path: str | Path) -> Any:
    """
    Load a saved model pipeline from disk.
    
    Returns:
        The deserialized model pipeline.
    
    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For other deserialization or I/O errors.
    """
    try:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Use joblib for safer deserialization
        model = joblib.load(path)

        logger.info(f"Model loaded successfully from: {model_path}")
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def predict(
    model: Any, data: pd.DataFrame | str | Path, target_column: str | None = None
) -> np.ndarray:
    """
    Produce predictions from a fitted pipeline on the provided dataset.
    
    Parameters:
        data (pd.DataFrame | str | Path): Feature data as a DataFrame or a path to a CSV; if a path is given, the file will be loaded.
        target_column (str | None): Name of the target column to exclude from features if present.
    
    Returns:
        np.ndarray: Array of model predictions.
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