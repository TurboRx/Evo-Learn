"""
Evo-Learn MLOptimizer Compatibility Layer

This module provides backward compatibility for code using the old
mloptimizer import structure. New code should import directly from
the root modules (core, preprocessing, utils, etc.).

Usage (legacy):
    from mloptimizer.core import run_automl
    from mloptimizer.preprocessing import preprocess_data

Usage (recommended):
    from core import run_automl
    from preprocessing import build_preprocessor

Version:
    1.2.0 (compatibility layer)

License:
    MIT License
"""

__version__ = "1.2.0"

import logging
import sys
import os

# Add parent directory to path to import root modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note about compatibility
logger.debug("MLOptimizer compatibility layer loaded. Consider importing from root modules directly.")

