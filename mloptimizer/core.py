"""
Backward compatibility wrapper for mloptimizer.core

This module wraps the root-level core module for backward compatibility.
New code should import directly from core module at the root.

Deprecated:
    from mloptimizer.core import run_automl  # Old way

Recommended:
    from core import run_automl  # New way
"""

import sys
import os
import warnings

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from root core module
from core import *

warnings.warn(
    "Importing from mloptimizer.core is deprecated. "
    "Please import directly from core module instead.",
    DeprecationWarning,
    stacklevel=2
)

