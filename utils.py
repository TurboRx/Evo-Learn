"""Utility functions."""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def timer(unit: str = "seconds", log_level: int = logging.INFO) -> Callable:
    """Decorator to measure execution time."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = (
                    (end_time - start_time) * 1000
                    if unit == "milliseconds"
                    else (end_time - start_time)
                )
                unit_label = "ms" if unit == "milliseconds" else "s"
                logger.log(
                    log_level,
                    f"'{func.__name__}' executed in {elapsed_time:.2f} {unit_label}",
                )
                if log_level == logging.DEBUG:
                    logger.debug(f"Args: {args}, kwargs: {kwargs}")
                return result
            except Exception as e:
                end_time = time.time()
                elapsed_time = (
                    (end_time - start_time) * 1000
                    if unit == "milliseconds"
                    else (end_time - start_time)
                )
                unit_label = "ms" if unit == "milliseconds" else "s"
                logger.error(
                    f"'{func.__name__}' failed after {elapsed_time:.2f} {unit_label}"
                )
                logger.exception(e)
                raise

        return wrapper

    return decorator


def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def validate_input(data: Any, expected_type: Any, var_name: str) -> None:
    """Validate input type and emptiness."""
    if not isinstance(data, expected_type):
        raise TypeError(
            f"{var_name} must be {expected_type.__name__}, got {type(data).__name__}"
        )
    try:
        if len(data) == 0:
            raise ValueError(f"{var_name} cannot be empty")
    except TypeError:
        pass  # data doesn't support len()


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def save_json(data: dict[str, Any], filepath: str | Path) -> None:
    """Save dictionary to JSON file."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to {filepath}")
    except Exception as e:
        logger.error(f"Save failed: {e}")
        raise


def load_json(filepath: str | Path) -> dict[str, Any]:
    """Load JSON file to dictionary."""
    try:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        with path.open("r") as f:
            data = json.load(f)
        logger.info(f"Loaded from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_size_mb(filepath: str | Path) -> float:
    """Get file size in MB."""
    return Path(filepath).stat().st_size / (1024 * 1024)
