"""Core AutoML functionality using TPOT."""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if os.getenv("EVO_LEARN_SHOW_WARNINGS", "").lower() != "true":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def load_data(data_path: str | Path, max_size_mb: int = 500) -> pd.DataFrame:
    """Load CSV file with validation and type coercion."""
    try:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
            )
        logger.info(f"Loading: {data_path} ({file_size_mb:.2f}MB)")

        if path.suffix.lower() != ".csv":
            logger.warning(f"File does not have .csv extension: {data_path}")

        data = pd.read_csv(data_path)
        logger.info(f"Loaded: shape={data.shape}")

        if data.empty:
            raise ValueError(f"Empty file: {data_path}")

        if len(data.columns) < 2:
            raise ValueError(f"Need at least 2 columns, found {len(data.columns)}")

        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = pd.to_numeric(data[col], errors="ignore")

        return data

    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty file: {data_path}")
    except pd.errors.ParserError as e:
        logger.error(f"CSV parse error: {e}")
        raise ValueError(f"Invalid CSV format: {e}")
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise


def split_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    task: str = "classification",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets with stratification for classification."""
    if target_column not in data.columns:
        raise KeyError(f"Target '{target_column}' not found in columns")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if len(data) < 4:
        raise ValueError(f"Insufficient data: need at least 4 rows, got {len(data)}")

    strat = None
    if task.lower() == "classification":
        unique_classes = y.nunique()
        if unique_classes > 1 and unique_classes <= len(y) // 2:
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:
                strat = y
                logger.info(f"Stratified split: {unique_classes} classes")
            else:
                logger.warning(
                    f"Skipping stratification: min class has {min_class_count} samples"
                )
        else:
            logger.warning(f"Skipping stratification: {unique_classes} classes")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
        logger.info(f"Split: train={X_train.shape}, test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    except ValueError as e:
        if strat is not None:
            logger.warning(f"Stratified split failed ({e}), retrying without stratification")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=None
            )
            logger.info(f"Split: train={X_train.shape}, test={X_test.shape}")
            return X_train, X_test, y_train, y_test
        logger.error(f"Split failed: {e}")
        raise


def _load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load YAML config or return empty dict."""
    if not config_path:
        return {}

    try:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found: {config_path}")
            return {}

        with path.open("r") as f:
            config = yaml.safe_load(f) or {}

        logger.info(f"Loaded config: {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Config load error: {e}")
        return {}


def validate_data_for_training(
    data: pd.DataFrame, target_column: str, task: str
) -> None:
    """Validate data before training."""
    if target_column not in data.columns:
        available_cols = list(data.columns)
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}"
        )

    nan_count = data[target_column].isna().sum()
    if nan_count > 0:
        raise ValueError(f"Target '{target_column}' contains {nan_count} NaN values")

    if task.lower() == "classification":
        unique_classes = data[target_column].nunique()
        if unique_classes < 2:
            raise ValueError(
                f"Need at least 2 classes, got {unique_classes}: {data[target_column].unique()}"
            )

        class_counts = data[target_column].value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = (
            max_class_count / min_class_count if min_class_count > 0 else float("inf")
        )

        if imbalance_ratio > 10:
            logger.warning(f"Severe class imbalance: {imbalance_ratio:.1f}:1")
            logger.warning(f"Distribution: {class_counts.to_dict()}")

    all_nan_cols = data.columns[data.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(f"{len(all_nan_cols)} columns with all NaN values: {all_nan_cols[:5]}")

    constant_cols = []
    for col in data.columns:
        if col != target_column and data[col].nunique() == 1:
            constant_cols.append(col)

    if constant_cols:
        logger.warning(f"{len(constant_cols)} constant features: {constant_cols[:5]}")

    logger.info(f"Validation passed: {len(data)} samples, {len(data.columns)} features")


def _compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics."""
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
                logger.warning(f"ROC AUC failed: {e}")

        return metrics

    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }


def _compute_regression_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """Compute regression metrics."""
    try:
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
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
    """Run AutoML with TPOT or baseline models."""
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
        logger.info(f"Output dir: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output dir: {e}")
        raise

    data = load_data(data_path)
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
        logger.info("Preprocessor built")
    except Exception as e:
        logger.error(f"Preprocessor build failed: {e}")
        raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _fit_and_package(final_estimator: Any, model_tag: str) -> dict[str, Any]:
        """Fit model with preprocessing and package results."""
        try:
            model = Pipeline(
                steps=[("preprocess", preprocessor), ("est", final_estimator)]
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            match task.lower():
                case "classification":
                    y_proba = None
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)[:, 1]
                    metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
                case _:
                    metrics = _compute_regression_metrics(y_test, y_pred)

            model_name = f"{model_tag}_{task}_{timestamp}"
            model_path = output_dir / f"{model_name}.pkl"

            joblib.dump(model, model_path, compress=3)

            result = {
                "model_name": model_name,
                "model_type": model_tag,
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
            }

            logger.info(f"Model saved: {model_path}")
            logger.info(f"Metrics: {metrics}")
            return result

        except Exception as e:
            logger.error(f"Fit/package error: {e}")
            raise

    if always_baseline:
        logger.info("Using baseline model")
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
        logger.info("Starting TPOT optimization")

        tpot_params = {
            "generations": generations,
            "population_size": population_size,
            "random_state": random_state,
            "verbosity": 2,
            "n_jobs": n_jobs,
        }

        if max_time_mins:
            tpot_params["max_time_mins"] = max_time_mins
        if max_eval_time_mins:
            tpot_params["max_eval_time_mins"] = max_eval_time_mins

        match task.lower():
            case "classification":
                tpot = TPOTClassifier(**tpot_params)
            case _:
                tpot = TPOTRegressor(**tpot_params)

        tpot.fit(X_train, y_train)

        y_pred = tpot.predict(X_test)

        match task.lower():
            case "classification":
                y_proba = None
                if hasattr(tpot, "predict_proba"):
                    y_proba = tpot.predict_proba(X_test)[:, 1]
                metrics = _compute_classification_metrics(y_test, y_pred, y_proba)
            case _:
                metrics = _compute_regression_metrics(y_test, y_pred)

        model_name = f"tpot_{task}_{timestamp}"
        model_path = output_dir / f"{model_name}.pkl"

        model = Pipeline(steps=[("preprocess", preprocessor), ("tpot", tpot)])
        model.fit(X_train, y_train)

        python_script_path = output_dir / f"{model_name}_pipeline.py"
        try:
            tpot.export(str(python_script_path))
            logger.info(f"Pipeline exported: {python_script_path}")
        except Exception as e:
            logger.warning(f"Export failed: {e}")
            python_script_path = None

        joblib.dump(model, model_path, compress=3)

        result = {
            "model_name": model_name,
            "model_type": "tpot",
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
            },
        }

        try:
            metadata_path = output_dir / f"{model_name}_metadata.json"
            with metadata_path.open("w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Metadata saved: {metadata_path}")
        except Exception as e:
            logger.warning(f"Metadata save failed: {e}")

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
                    plt.title("Feature Importance")
                    plt.tight_layout()
                    fig_path = output_dir / f"{model_name}_feature_importance.png"
                    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    result["feature_importance_plot"] = str(fig_path)
                    logger.info(f"Feature importance: {fig_path}")
        except Exception as e:
            logger.info(f"Feature importance skipped: {e}")

        logger.info(f"TPOT complete: {metrics}")
        return result

    except Exception as e:
        logger.error(f"TPOT failed: {e}")
        logger.info("Falling back to baseline")

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
    """Load model from disk with pickle fallback."""
    try:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            model = joblib.load(path)
        except Exception as joblib_error:
            logger.warning(f"Joblib failed ({joblib_error}), trying pickle")
            import pickle

            with path.open("rb") as f:
                model = pickle.load(f)

        logger.info(f"Model loaded: {model_path}")
        return model

    except FileNotFoundError:
        logger.error(f"Model not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise


def predict(
    model: Any, data: pd.DataFrame | str | Path, target_column: str | None = None
) -> np.ndarray:
    """Generate predictions from model."""
    if isinstance(data, (str, Path)):
        data = load_data(data)

    X = data.drop(columns=[target_column]) if target_column else data

    try:
        predictions = model.predict(X)
        logger.info(f"Predictions generated: {len(predictions)} samples")
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


def save_model(model: Any, model_path: str | Path) -> None:
    """Save model to disk."""
    try:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)
        logger.info(f"Model saved: {path}")
    except Exception as e:
        logger.error(f"Save error: {e}")
        raise
