"""Tests for utility helpers."""
import numpy as np
import pytest

from utils import validate_input


def test_validate_input_empty_ndarray():
    """Ensure empty numpy arrays are rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input(np.array([]), np.ndarray, "data")

    validate_input(np.array([1, 2, 3]), np.ndarray, "data")
