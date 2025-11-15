"""
Backward compatibility wrapper for mloptimizer.preprocessing

This module wraps the root-level preprocessing module for backward compatibility.
New code should import directly from preprocessing module at the root.

Deprecated:
    from mloptimizer.preprocessing import preprocess_data  # Old way

Recommended:
    from preprocessing import build_preprocessor  # New way
"""

import sys
import os
import warnings

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from root preprocessing module
from preprocessing import *

warnings.warn(
    "Importing from mloptimizer.preprocessing is deprecated. "
    "Please import directly from preprocessing module instead.",
    DeprecationWarning,
    stacklevel=2
)

