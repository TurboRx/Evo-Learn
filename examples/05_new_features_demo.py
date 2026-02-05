"""Demonstrate key features with concise examples."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from core import run_automl


def example_valid_data():
    """Valid classification dataset."""
    print("\n" + "=" * 50)
    print("Valid Classification")
    print("=" * 50)

    data = pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "target": np.random.choice([0, 1], 100),
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = Path(f.name)
        data.to_csv(path, index=False)

    try:
        result = run_automl(
            data_path=path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
        )
        print(f"✓ Success! Accuracy: {result['metrics']['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    finally:
        path.unlink(missing_ok=True)


def example_nan_in_target():
    """NaN in target column - validation catches it."""
    print("\n" + "=" * 50)
    print("NaN in Target (Should Fail)")
    print("=" * 50)

    data = pd.DataFrame({
        "f1": np.random.randn(100),
        "target": [0, 1, np.nan, 0, 1] * 20,
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = Path(f.name)
        data.to_csv(path, index=False)

    try:
        _ = run_automl(path, "target", "classification", always_baseline=True)
        print("✗ Should have failed")
    except ValueError as e:
        print(f"✓ Caught: {str(e)[:80]}...")
    finally:
        path.unlink(missing_ok=True)


def example_single_class():
    """Single class - validation catches it."""
    print("\n" + "=" * 50)
    print("Single Class (Should Fail)")
    print("=" * 50)

    data = pd.DataFrame({
        "f1": np.random.randn(100),
        "target": [0] * 100,
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = Path(f.name)
        data.to_csv(path, index=False)

    try:
        _ = run_automl(path, "target", "classification", always_baseline=True)
        print("✗ Should have failed")
    except ValueError as e:
        print(f"✓ Caught: {str(e)[:80]}...")
    finally:
        path.unlink(missing_ok=True)


def example_imbalanced():
    """Class imbalance - warns but continues."""
    print("\n" + "=" * 50)
    print("Severe Imbalance (Warning)")
    print("=" * 50)

    data = pd.DataFrame({
        "f1": np.random.randn(200),
        "target": [0] * 190 + [1] * 10,
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = Path(f.name)
        data.to_csv(path, index=False)

    try:
        result = run_automl(
            path, "target", "classification", generations=2, always_baseline=True
        )
        print(f"✓ Success with warning. Acc: {result['metrics']['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Evo-Learn Feature Demonstrations")
    print("=" * 50)

    example_valid_data()
    example_nan_in_target()
    example_single_class()
    example_imbalanced()

    print("\n" + "=" * 50)
    print("Complete")
    print("=" * 50)
