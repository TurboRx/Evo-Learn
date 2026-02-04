"""Package wrapper for top-level core module."""

from core import load_model, predict, run_automl

__all__ = [
    "run_automl",
    "load_model",
    "predict",
]
