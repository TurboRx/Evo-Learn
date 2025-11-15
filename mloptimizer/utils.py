"""
Backward compatibility wrapper for mloptimizer.utils

This module wraps the root-level utils module for backward compatibility.
New code should import directly from utils module at the root.

Deprecated:
    from mloptimizer.utils import timer  # Old way

Recommended:
    from utils import timer  # New way
"""

import sys
import os
import warnings

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from root utils module
from utils import *

warnings.warn(
    "Importing from mloptimizer.utils is deprecated. "
    "Please import directly from utils module instead.",
    DeprecationWarning,
    stacklevel=2
)

