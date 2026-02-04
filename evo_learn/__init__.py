"""Evo-Learn: Professional AutoML toolkit built on TPOT.

Evo-Learn provides automated machine learning with production-ready preprocessing,
config-driven runs, baseline fallbacks, and comprehensive evaluation.
"""

__version__ = "1.2.0"
__author__ = "TurboRx"
__email__ = "turborx@example.com"

from .core import load_model, predict, run_automl
from .logging_utils import setup_logging
from .preprocessing import build_preprocessor

__all__ = [
    "run_automl",
    "load_model",
    "predict",
    "build_preprocessor",
    "setup_logging",
]
